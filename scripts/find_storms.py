"""Поиск геомагнитных бурь (Dst < -50 нТл) по данным OMNI2 за 2000-2024.

Алгоритм:
  * загружаются все годовые файлы omni2_YYYY.dat;
  * кандидаты — часы с Dst < -50 нТл;
  * кандидаты, отстоящие друг от друга не более чем на 24 ч, объединяются
    в одну бурю (дедупликация);
  * для каждой бури определяются:
      - start_time  : первое падение ниже -30 нТл перед минимумом;
      - min_time    : время минимума Dst;
      - end_time    : возврат выше -30 нТл после минимума.

Результат: data/processed/storm_catalog.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import load_omni_data

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storm_catalog.csv")

START_YEAR = 2000
END_YEAR = 2024

STORM_THRESHOLD = -50.0
ONSET_THRESHOLD = -30.0
MERGE_HOURS = 24.0


def load_all_years(start_year: int, end_year: int) -> pd.DataFrame:
    """Загружает и объединяет годовые OMNI2-файлы в один временной ряд."""
    frames = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(RAW_DIR, f"omni2_{year}.dat")
        if not os.path.exists(path):
            print(f"[find_storms] Пропущен отсутствующий файл: {path}")
            continue
        df = load_omni_data(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("Не найдено ни одного годового файла OMNI2.")
    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("datetime").reset_index(drop=True)
    return all_df


def find_storm_groups(dst: np.ndarray, times: np.ndarray) -> list[list[int]]:
    """Группирует часы с Dst < STORM_THRESHOLD в бури (разрыв > 24 ч = новая буря)."""
    candidates = np.where(np.isfinite(dst) & (dst < STORM_THRESHOLD))[0]
    if candidates.size == 0:
        return []

    groups: list[list[int]] = []
    current = [int(candidates[0])]
    for idx in candidates[1:]:
        gap_h = (times[idx] - times[current[-1]]) / np.timedelta64(1, "h")
        if gap_h <= MERGE_HOURS:
            current.append(int(idx))
        else:
            groups.append(current)
            current = [int(idx)]
    groups.append(current)
    return groups


def event_boundaries(dst: np.ndarray, min_idx: int) -> tuple[int, int]:
    """Возвращает (start_idx, end_idx) по порогу ONSET_THRESHOLD вокруг минимума."""
    start_idx = min_idx
    i = min_idx
    while i > 0 and np.isfinite(dst[i]) and dst[i] < ONSET_THRESHOLD:
        start_idx = i
        i -= 1

    end_idx = min_idx
    j = min_idx
    n = len(dst)
    while j < n - 1 and np.isfinite(dst[j]) and dst[j] < ONSET_THRESHOLD:
        end_idx = j
        j += 1
    return start_idx, end_idx


def build_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Строит каталог бурь."""
    dst = df["Dst"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()

    groups = find_storm_groups(dst, times)
    records = []
    for storm_id, group in enumerate(groups, start=1):
        min_idx = group[int(np.argmin(dst[group]))]
        min_dst = float(dst[min_idx])
        min_time = df["datetime"].iloc[min_idx]
        start_idx, end_idx = event_boundaries(dst, min_idx)
        start_time = df["datetime"].iloc[start_idx]
        end_time = df["datetime"].iloc[end_idx]
        duration = (end_time - start_time).total_seconds() / 3600.0
        records.append(
            {
                "storm_id": storm_id,
                "year": int(min_time.year),
                "min_time": min_time,
                "min_dst": min_dst,
                "start_time": start_time,
                "end_time": end_time,
                "duration_hours": duration,
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    print(f"[find_storms] Загрузка данных OMNI2 за {START_YEAR}-{END_YEAR}...")
    df = load_all_years(START_YEAR, END_YEAR)
    print(f"[find_storms] Всего строк: {len(df)}")

    catalog = build_catalog(df)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    catalog.to_csv(OUT_CSV, index=False)

    print(f"[find_storms] Найдено бурь (Dst < {STORM_THRESHOLD:.0f}): {len(catalog)}")
    print(f"[find_storms] Каталог сохранён: {OUT_CSV}")

    weak = ((catalog["min_dst"] >= -100.0)).sum()
    medium = ((catalog["min_dst"] < -100.0) & (catalog["min_dst"] >= -200.0)).sum()
    strong = (catalog["min_dst"] < -200.0).sum()
    print(
        f"[find_storms] Слабых (-50..-100): {weak}, "
        f"средних (-100..-200): {medium}, сильных (< -200): {strong}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
