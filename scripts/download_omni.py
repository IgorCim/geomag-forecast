"""Идемпотентное скачивание часового OMNI2-файла.

Источник: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/

Использование:
    python scripts/download_omni.py --year 2015
    python scripts/download_omni.py --url <url> --dest data/raw/omni2_2015.dat
"""

from __future__ import annotations

import argparse
import os
import sys

import requests

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
DEFAULT_YEAR = 2015
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

CHUNK_SIZE = 1 << 20


def download_file(url: str, dest: str) -> str:
    """Скачивает ``url`` в ``dest``, если файла ещё нет (идемпотентно).

    Возвращает путь к локальному файлу.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[download] Файл уже существует, скачивание пропущено: {dest}")
        return dest

    print(f"[download] Скачивание: {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        tmp_dest = dest + ".part"
        with open(tmp_dest, "wb") as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
        os.replace(tmp_dest, dest)

    print(f"[download] Готово: {dest}")
    return dest


def report_file(filepath: str) -> None:
    """Печатает размер файла и первые 5 строк."""
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024.0 * 1024.0)
    print(f"[download] Путь: {filepath}")
    print(f"[download] Размер: {size_mb:.2f} МБ ({size_bytes} байт)")
    print("[download] Первые 5 строк:")
    with open(filepath, "r") as handle:
        for i, line in enumerate(handle):
            if i >= 5:
                break
            print("   " + line.rstrip("\n"))


def download_year(year: int) -> str:
    """Скачивает файл за один год (идемпотентно). Возвращает путь."""
    url = f"{BASE_URL}omni2_{year}.dat"
    dest = os.path.join(DEFAULT_RAW_DIR, f"omni2_{year}.dat")
    return download_file(url, dest)


def download_years(start_year: int, end_year: int) -> list[str]:
    """Скачивает файлы за диапазон лет с логированием прогресса каждые 5 файлов."""
    paths: list[str] = []
    total = end_year - start_year + 1
    for i, year in enumerate(range(start_year, end_year + 1), start=1):
        existed = os.path.exists(
            os.path.join(DEFAULT_RAW_DIR, f"omni2_{year}.dat")
        )
        filepath = download_year(year)
        size_mb = os.path.getsize(filepath) / (1024.0 * 1024.0)
        status = "пропущено (уже есть)" if existed else "скачано"
        print(f"[download] {i}/{total} {year}: {status}, {size_mb:.2f} МБ")
        paths.append(filepath)
        if i % 5 == 0:
            print(f"[download] --- прогресс: {i}/{total} файлов ---")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Скачивание часовых OMNI2-файлов.")
    parser.add_argument("--year", type=int, default=None, help="Скачать один год")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--url", type=str, default=None)
    parser.add_argument("--dest", type=str, default=None)
    args = parser.parse_args(argv)

    if args.url or args.dest:
        year = args.year if args.year is not None else DEFAULT_YEAR
        url = args.url or f"{BASE_URL}omni2_{year}.dat"
        dest = args.dest or os.path.join(DEFAULT_RAW_DIR, f"omni2_{year}.dat")
        filepath = download_file(url, dest)
        report_file(filepath)
        return 0

    if args.year is not None:
        filepath = download_year(args.year)
        report_file(filepath)
        return 0

    paths = download_years(args.start_year, args.end_year)
    print(f"[download] Всего файлов: {len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
