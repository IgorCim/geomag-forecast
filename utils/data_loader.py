"""Загрузка данных OMNI для модели прогноза Dst-индекса.

Модуль читает ASCII-файлы **часового** формата OMNI2
(``omni2_YYYY.dat`` / ``OMNI2_YYYY.DAT``) и преобразует их в DataFrame
с унифицированными колонками::

    ['datetime', 'V', 'Bz', 'P_dyn', 'Dst']

Индексы колонок (0-based) соответствуют официальному описанию формата
OMNI2_YYYY.DAT (55 слов). Ключевые поля:

    word 17 (idx 16)  Bz GSM, нТл
    word 25 (idx 24)  Plasma (Flow) speed, км/с
    word 29 (idx 28)  Flow Pressure, нПа
    word 41 (idx 40)  Dst index, нТл

Проверка (первая строка ``omni2_2015.dat``): word 36 (E-field) = -0.62
совпадает с ``-V*Bz_GSM*1e-3 = -568*1.1*1e-3``, что подтверждает индексы
Bz GSM (idx 16) и V (idx 24).

Индексы можно переопределить через параметр ``columns``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

#: Имена 55 колонок часового OMNI2 (для inspect_omni_columns).
OMNI2_COLUMN_NAMES = [
    "Year", "Decimal Day", "Hour", "Bartels rotation", "ID IMF SC",
    "ID SW SC", "#pts IMF", "#pts plasma", "|B|", "Mag avg field vector",
    "Lat angle avg field", "Long angle avg field", "Bx GSE/GSM", "By GSE",
    "Bz GSE", "By GSM", "Bz GSM", "sigma|B|", "sigma B", "sigma Bx",
    "sigma By", "sigma Bz", "Proton temperature", "Proton density",
    "Plasma speed", "Flow long angle", "Flow lat angle", "Na/Np",
    "Flow pressure", "sigma T", "sigma N", "sigma V", "sigma phi V",
    "sigma theta V", "sigma Na/Np", "Electric field", "Plasma beta",
    "Alfven mach", "Kp", "R (sunspot)", "Dst", "AE", "pflux >1MeV",
    "pflux >2MeV", "pflux >4MeV", "pflux >10MeV", "pflux >30MeV",
    "pflux >60MeV", "Flag", "ap-index", "f10.7", "PC(N)", "AL", "AU",
    "Magnetosonic mach",
]

#: Индексы колонок (0-based) в часовом файле OMNI2_YYYY.DAT.
DEFAULT_COLUMNS: Dict[str, int] = {
    "year": 0,    # word 1
    "doy": 1,     # word 2
    "hour": 2,    # word 3
    "Bz": 16,     # word 17 - Bz GSM, нТл
    "V": 24,      # word 25 - Plasma (Flow) speed, км/с
    "P_dyn": 28,  # word 29 - Flow Pressure, нПа
    "Dst": 40,    # word 41 - Dst index, нТл
}

#: Значения-заполнители OMNI (разные для разных полей формата).
MISSING_VALUES = [
    999.9, 9999.9, 99999.9, 999999.9, 9999999.0,
    999.99, 9999.99, 99999.99, 999999.99,
    999.0, 9999.0, 99999.0, 999999.0, 9999999.0,
    99.9, 99.99, 9.999,
]

#: Заполнители для конкретных интересующих колонок.
DEFAULT_FILL_VALUES: Dict[str, float] = {
    "V": 9999.0,      # F6.0
    "Bz": 999.9,      # F6.1
    "P_dyn": 99.99,   # F6.2
    "Dst": 99999.0,   # I6
}


def inspect_omni_columns(filepath: str, n_rows: int = 3) -> None:
    """Печатает первые ``n_rows`` строк и все колонки с индексами и значениями.

    Помогает по факту определить индексы колонок в конкретном файле.
    """
    print(f"[inspect] Файл: {filepath}")
    with open(filepath, "r") as handle:
        for i in range(n_rows):
            line = handle.readline()
            if not line:
                break
            tokens = line.split()
            print(f"[inspect] Строка {i} ({len(tokens)} колонок):")
            for idx, tok in enumerate(tokens):
                name = OMNI2_COLUMN_NAMES[idx] if idx < len(OMNI2_COLUMN_NAMES) else "?"
                print(f"    [{idx:2d}] {name:<24} = {tok}")


def load_omni_data(
    filepath: str,
    columns: Optional[Dict[str, int]] = None,
    fill_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Загружает часовой OMNI2-файл и возвращает DataFrame с нужными колонками.

    Параметры
    ---------
    filepath : str
        Путь к ASCII-файлу OMNI2 (часовой формат).
    columns : dict, optional
        Переопределение индексов колонок: ``year``, ``doy``, ``hour``,
        ``V``, ``Bz``, ``P_dyn``, ``Dst``.
    fill_values : dict, optional
        Переопределение значений-заполнителей для колонок V, Bz, P_dyn, Dst.

    Возвращает
    ----------
    pandas.DataFrame
        Колонки: ``datetime``, ``V``, ``Bz``, ``P_dyn``, ``Dst``.
        Пропуски заменены на ``NaN``.
    """
    col = dict(DEFAULT_COLUMNS)
    if columns:
        col.update(columns)
    fills = dict(DEFAULT_FILL_VALUES)
    if fill_values:
        fills.update(fill_values)

    raw = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        comment="#",
        engine="python",
    )

    raw = raw.replace(MISSING_VALUES, np.nan)

    year = pd.to_numeric(raw.iloc[:, col["year"]], errors="coerce")
    doy = pd.to_numeric(raw.iloc[:, col["doy"]], errors="coerce")
    hour = pd.to_numeric(raw.iloc[:, col["hour"]], errors="coerce")

    if "minute" in col:
        minute = pd.to_numeric(raw.iloc[:, col["minute"]], errors="coerce")
    else:
        minute = pd.Series(0.0, index=raw.index)

    year_int = year.astype("Int64")
    base = pd.to_datetime(year_int.astype(str), format="%Y", errors="coerce")
    datetime = (
        base
        + pd.to_timedelta(doy - 1.0, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
    )

    data = {
        "datetime": datetime,
        "V": pd.to_numeric(raw.iloc[:, col["V"]], errors="coerce"),
        "Bz": pd.to_numeric(raw.iloc[:, col["Bz"]], errors="coerce"),
        "P_dyn": pd.to_numeric(raw.iloc[:, col["P_dyn"]], errors="coerce"),
        "Dst": pd.to_numeric(raw.iloc[:, col["Dst"]], errors="coerce"),
    }
    result = pd.DataFrame(data)

    for name, fill in fills.items():
        result[name] = result[name].mask(result[name] >= fill, np.nan)

    _log_summary(filepath, result)
    return result


def _log_summary(filepath: str, df: pd.DataFrame) -> None:
    """Печатает краткую сводку о загруженных данных в stdout."""
    n_rows = len(df)
    nan_counts = {c: int(df[c].isna().sum()) for c in ("V", "Bz", "P_dyn", "Dst")}
    valid_dt = df["datetime"].dropna()
    if len(valid_dt) > 0:
        dt_range = f"{valid_dt.min()} .. {valid_dt.max()}"
    else:
        dt_range = "нет валидных дат"

    print(f"[data_loader] Файл: {filepath}")
    print(f"[data_loader] Строк: {n_rows}")
    print(f"[data_loader] Пропуски NaN: {nan_counts}")
    print(f"[data_loader] Диапазон дат: {dt_range}")
