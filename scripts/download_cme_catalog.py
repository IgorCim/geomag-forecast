"""Скачивание каталога CME (2010-2024) с классификацией Earth-directed.

Источники (в порядке приоритета):
  1. NASA DONKI API: https://kauai.ccmc.gsfc.nasa.gov/DONKI/WSM/CME/
  2. CDAW/LASCO каталог (единый текстовый файл):
     https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt

Классификация (Этап 2.5):
  * is_halo        -- width == 360 или "Halo" в Remarks;
  * source_lat/lon -- широта/долгота из Remarks (если координаты есть);
  * is_earth_directed -- halo (width >= 300) ИЛИ (|source_lon| <= 30 и width >= 60).

Выход: data/raw/cme_catalog.csv
columns: [cme_id, start_time, v_linear, angular_width, acceleration,
          source_location, is_halo, source_lat, source_lon, is_earth_directed]

Использование:
    python scripts/download_cme_catalog.py --start-year 2010 --end-year 2024
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_CSV = os.path.join(RAW_DIR, "cme_catalog.csv")

DONKI_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME"
CDAW_TEXT_URL = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/text_ver/univ_all.txt"

_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_DATE = re.compile(r"^\s*(\d{4})/(\d{2})/(\d{2})\s+(\d{2}:\d{2}:\d{2})\s+(.*)$")
_COORD = re.compile(r"\b([NS])\s*(\d{1,2})\s*([EW])\s*(\d{1,3})\b", flags=re.I)

HALO_WIDTH_THRESHOLD = 300.0
HALO_FULL_WIDTH = 360.0
FRONT_LON_LIMIT = 30.0
MIN_WIDTH_FRONT = 60.0


def _first_number(text, default=np.nan) -> float:
    match = _NUM.search(str(text) or "")
    return float(match.group()) if match else float(default)


def parse_coordinates(remarks: str) -> tuple[float, float]:
    """Извлекает (широта, долгота) из строки Remarks. Иначе (NaN, NaN)."""
    match = _COORD.search(remarks or "")
    if not match:
        return float("nan"), float("nan")
    lat_sign = 1.0 if match.group(1).upper() == "N" else -1.0
    lon_sign = 1.0 if match.group(3).upper() == "E" else -1.0
    lat = lat_sign * float(match.group(2))
    lon = lon_sign * float(match.group(4))
    return lat, lon


def is_halo(width: float, remarks: str = "") -> bool:
    """Halo: полный (width == 360) или явное упоминание Halo в Remarks."""
    if np.isfinite(width) and width >= HALO_FULL_WIDTH:
        return True
    return "halo" in (remarks or "").lower()


def is_earth_directed(width: float, source_lon: float = float("nan")) -> bool:
    """Earth-directed: halo (width >= 300) ИЛИ фронтальное CME (|lon| <= 30, width >= 60)."""
    if np.isfinite(width) and width >= HALO_WIDTH_THRESHOLD:
        return True
    if np.isfinite(source_lon) and abs(source_lon) <= FRONT_LON_LIMIT and np.isfinite(width) and width >= MIN_WIDTH_FRONT:
        return True
    return False


def classify(width: float, remarks: str) -> dict:
    """Возвращает словарь классификации CME."""
    lat, lon = parse_coordinates(remarks)
    return {
        "is_halo": is_halo(width, remarks),
        "source_lat": lat,
        "source_lon": lon,
        "is_earth_directed": is_earth_directed(width, lon),
    }


def filter_earth_directed(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только Earth-directed CME (по колонке или пересчитывая по width/lon)."""
    df = df.copy()
    if "is_earth_directed" not in df.columns:
        widths = df["angular_width"].to_numpy(dtype=float)
        lons = (
            df["source_lon"].to_numpy(dtype=float)
            if "source_lon" in df.columns
            else np.full(widths.shape, np.nan)
        )
        df["is_earth_directed"] = [is_earth_directed(w, lo) for w, lo in zip(widths, lons)]
    return df[df["is_earth_directed"].to_numpy(dtype=bool)].reset_index(drop=True)


def _try_donki(start_date: str, end_date: str):
    """Пытается получить CME из DONKI. Возвращает список dict или None."""
    try:
        response = requests.get(
            DONKI_URL,
            params={"startDate": start_date, "endDate": end_date},
            timeout=60,
        )
        if response.status_code != 200 or "json" not in response.headers.get("Content-Type", ""):
            return None
        data = response.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, list):
        return None

    records = []
    for event in data:
        analyses = event.get("cmeAnalyses") or []
        best = next((a for a in analyses if a.get("isMostAccurate")), analyses[0] if analyses else None)
        if best is None or best.get("speed") is None:
            continue
        half_angle = best.get("halfAngle")
        width = 2.0 * half_angle if half_angle is not None else float("nan")
        lat, lon = best.get("latitude"), best.get("longitude")
        remarks = event.get("note", "") or ""
        info = classify(float(width) if np.isfinite(width) else float("nan"), remarks)
        location = f"lat={lat},lon={lon}" if lat is not None and lon is not None else (event.get("sourceLocation") or "unknown")
        records.append(
            {
                "cme_id": event.get("cmeID", ""),
                "start_time": event.get("startTime", ""),
                "v_linear": float(best["speed"]),
                "angular_width": float(width) if np.isfinite(width) else 0.0,
                "acceleration": float(best["acceleration"]) if best.get("acceleration") is not None else 0.0,
                "source_location": location,
                **info,
            }
        )
    return records


def _parse_cdaw_text(text: str, start_year: int, end_year: int) -> list[dict]:
    """Разбирает единый текстовый каталог CDAW (univ_all.txt)."""
    records = []
    for line in text.splitlines():
        match = _DATE.match(line)
        if not match:
            continue
        year = int(match.group(1))
        if year < start_year or year > end_year:
            continue
        date_part = f"{match.group(1)}/{match.group(2)}/{match.group(3)}"
        time_part = match.group(4)
        remainder = match.group(5)
        tokens = remainder.split()
        if len(tokens) < 12:
            continue

        try:
            ts = pd.to_datetime(f"{date_part} {time_part}", format="%Y/%m/%d %H:%M:%S")
        except (ValueError, TypeError):
            continue

        cpa = _first_number(tokens[0])
        width = _first_number(tokens[1])
        v_linear = _first_number(tokens[2])
        accel = _first_number(tokens[5])
        if not np.isfinite(v_linear):
            continue

        remarks = " ".join(tokens[12:])
        info = classify(width, remarks)
        if np.isfinite(info["source_lat"]) and np.isfinite(info["source_lon"]):
            location = f"lat={info['source_lat']:.0f},lon={info['source_lon']:.0f}"
        elif np.isfinite(cpa):
            location = f"PA={cpa:.0f}"
        else:
            location = "Halo"

        records.append(
            {
                "start_time": ts,
                "v_linear": v_linear,
                "angular_width": width if np.isfinite(width) else 0.0,
                "acceleration": accel,
                "source_location": location,
                **info,
            }
        )
    return records


def _download_cdaw_text(start_year: int, end_year: int) -> list[dict]:
    print(f"[cme] Загрузка единого каталога CDAW: {CDAW_TEXT_URL}")
    response = requests.get(CDAW_TEXT_URL, timeout=120)
    response.raise_for_status()
    text = response.content.decode("ISO-8859-1", errors="replace")
    records = _parse_cdaw_text(text, start_year, end_year)
    print(f"[cme] CDAW: извлечено {len(records)} событий за {start_year}-{end_year}")
    return records


def _finalize(records: list[dict], source: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df = df.sort_values("start_time").reset_index(drop=True)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["cme_id"] = df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%S") + "-CME"
    dup = df.groupby("cme_id").cumcount()
    df.loc[dup > 0, "cme_id"] = df.loc[dup > 0, "cme_id"] + "-" + (dup[dup > 0] + 1).astype(str)
    df["angular_width"] = df["angular_width"].fillna(0.0)
    df["acceleration"] = df["acceleration"].fillna(0.0)
    df["source_location"] = df["source_location"].fillna("unknown")
    cols = [
        "cme_id",
        "start_time",
        "v_linear",
        "angular_width",
        "acceleration",
        "source_location",
        "is_halo",
        "source_lat",
        "source_lon",
        "is_earth_directed",
    ]
    df = df[cols]
    df["start_time"] = df["start_time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[cme] Источник: {source}")
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачивание каталога CME.")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--source", choices=["auto", "donki", "cdaw"], default="auto")
    args = parser.parse_args(argv)

    os.makedirs(RAW_DIR, exist_ok=True)

    records = None
    source_used = None

    if args.source in ("auto", "donki"):
        probe = _try_donki(f"{args.start_year}-01-01", f"{args.start_year}-01-31")
        if probe is not None:
            print("[cme] DONKI доступен, скачивание по годам...")
            records = []
            for year in range(args.start_year, args.end_year + 1):
                year_records = _try_donki(f"{year}-01-01", f"{year}-12-31") or []
                print(f"[cme] DONKI {year}: {len(year_records)} событий")
                records.extend(year_records)
            source_used = "DONKI"
        else:
            print("[cme] DONKI недоступен (не-JSON), переход на CDAW.")

    if records is None:
        records = _download_cdaw_text(args.start_year, args.end_year)
        source_used = "CDAW"

    if not records:
        print("[cme] Не удалось получить ни одного события.")
        return 1

    df = _finalize(records, source_used)
    df.to_csv(OUT_CSV, index=False)

    n_total = len(df)
    n_halo = int((df["angular_width"] >= HALO_WIDTH_THRESHOLD).sum())
    n_full_halo = int((df["angular_width"] >= HALO_FULL_WIDTH).sum())
    n_earth = int(df["is_earth_directed"].sum())
    n_side = n_total - n_earth

    print(f"[cme] Всего CME: {n_total}")
    print(f"[cme] Halo (width >= 300): {n_halo} (из них полных width=360: {n_full_halo})")
    print(f"[cme] Earth-directed: {n_earth}")
    print(f"[cme] Side CME (исключены): {n_side}")
    print(f"[cme] Диапазон V0: {df['v_linear'].min():.0f} - {df['v_linear'].max():.0f} км/с")
    print(f"[cme] Период: {df['start_time'].iloc[0]} .. {df['start_time'].iloc[-1]}")
    print(f"[cme] Сохранено: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
