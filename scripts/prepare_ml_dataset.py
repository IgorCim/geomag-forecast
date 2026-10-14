"""Подготовка ML-датасета для коррекции residual BMR (Этап 3.1).

Пайплайн:
  1. Загрузка OMNI2 2000-2024, интерполяция пропусков (max gap = 3 ч);
  2. прогон BMR по каждому непрерывному сегменту;
  3. residual = Dst_observed - Dst_predicted;
  4. скользящие окна (12 ч истории -> residual[t+1]);
  5. нормализация по train;
  6. временной split (train 2000-2018 / val 2019-2021 / test 2022-2024);
  7. сохранение npz + norm_params.json + omni_with_bmr.parquet.

Запуск:
    python scripts/prepare_ml_dataset.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from find_storms import load_all_years
from models.empirical_bmr import EmpiricalBMR

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
PARQUET = os.path.join(PROCESSED_DIR, "omni_with_bmr.parquet")
NORM_JSON = os.path.join(PROCESSED_DIR, "norm_params.json")

START_YEAR = 2000
END_YEAR = 2024
MAX_GAP = 3
HISTORY = 12
HORIZON = 1
MIN_DROP_HOURS = 24
STRIDE = 1

FEATURES = ["V", "Bz", "P_dyn", "Dst_predicted", "Ey", "Q", "tau"]
SPLIT_RANGES = {"train": (2000, 2018), "val": (2019, 2021), "test": (2022, 2024)}


def load_and_clean() -> tuple[pd.DataFrame, dict]:
    """Загрузка и очистка OMNI, возвращает непрерывные сегменты и статистику."""
    raw = load_all_years(START_YEAR, END_YEAR)
    raw = raw.dropna(subset=["datetime"]).drop_duplicates("datetime").sort_values("datetime")
    raw = raw.set_index("datetime")

    full = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    present = raw.index.nunique()
    reindexed = raw.reindex(full)

    missing_mask = reindexed[["V", "Bz", "P_dyn", "Dst"]].isna().any(axis=1)
    missing = int(missing_mask.sum())

    for col in ("V", "Bz", "P_dyn", "Dst"):
        reindexed[col] = reindexed[col].interpolate(limit=MAX_GAP, limit_direction="both")
    cleaned = reindexed.dropna(subset=["V", "Bz", "P_dyn", "Dst"])

    stats = {
        "hours": int(len(full)),
        "present": int(present),
        "missing": missing,
    }
    return cleaned, stats


def split_runs(index: pd.DatetimeIndex) -> list[tuple[int, int]]:
    """Возвращает список [start, end) непрерывных часовых сегментов."""
    if len(index) == 0:
        return []
    diffs = np.diff(index.to_numpy()).astype("timedelta64[h]").astype(int)
    breaks = np.where(diffs != 1)[0]
    bounds = [0] + (breaks + 1).tolist() + [len(index)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def run_bmr_on_segments(cleaned: pd.DataFrame) -> pd.DataFrame:
    """BMR по каждому сегменту; возвращает объединённый датафрейм с прогнозом."""
    model = EmpiricalBMR()
    frames = []
    for start, end in split_runs(cleaned.index):
        segment = cleaned.iloc[start:end].copy()
        if len(segment) < 2:
            continue
        segment = segment.reset_index().rename(columns={"index": "datetime"})
        pred = model.predict(segment)
        pred["V"] = segment["V"].to_numpy()
        pred["Bz"] = segment["Bz"].to_numpy()
        pred["P_dyn"] = segment["P_dyn"].to_numpy()
        frames.append(pred)
    if not frames:
        raise RuntimeError("BMR не дал результатов.")
    out = pd.concat(frames, ignore_index=True)
    return out


def build_windows(df: pd.DataFrame, model: EmpiricalBMR) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Скользящие окна по непрерывным сегментам. Возвращает (X, y, dates, n_dropped)."""
    bz = df["Bz"].to_numpy(dtype=float)
    v = df["V"].to_numpy(dtype=float)
    bs = model.calculate_bs(bz)
    ey = model.calculate_ey(v, bs)
    df = df.assign(Ey=ey)
    df["residual"] = df["Dst_observed"] - df["Dst_predicted"]

    feat = df[FEATURES].to_numpy(dtype=float)
    residual = df["residual"].to_numpy(dtype=float)
    times = df["datetime"].to_numpy()
    idx = pd.DatetimeIndex(df["datetime"])

    xs, ys, ds = [], [], []
    n_dropped = 0
    for start, end in split_runs(idx):
        seg_start = start + MIN_DROP_HOURS
        for i in range(seg_start, end):
            if i - HISTORY < start:
                continue
            window = feat[i - HISTORY : i]
            target = residual[i + 0] if HORIZON == 1 else residual[i + HORIZON - 1]
            if window.shape[0] != HISTORY:
                continue
            xs.append(window)
            ys.append(target)
            ds.append(times[i + HORIZON - 1])
        n_dropped += min(MIN_DROP_HOURS, end - start)

    X = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    dates = np.asarray(ds, dtype="datetime64[ns]")
    return X, y, dates, n_dropped


def normalize(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = X_train.reshape(-1, X_train.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def _save_split(path: str, X: np.ndarray, y: np.ndarray, dates: np.ndarray) -> None:
    np.savez_compressed(
        path,
        X=X.astype(np.float32),
        y=y.astype(np.float32),
        dates=dates.astype("datetime64[ns]"),
    )


def main() -> int:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print(f"[ml] Загрузка OMNI2 {START_YEAR}-{END_YEAR}...")
    cleaned, stats = load_and_clean()
    runs = split_runs(cleaned.index)
    print(f"[ml] Часов: {stats['hours']}, пропусков: {stats['missing']}, разрывов: {len(runs) - 1}")

    print("[ml] Прогон BMR по сегментам...")
    bmr_df = run_bmr_on_segments(cleaned)
    bmr_df = bmr_df.sort_values("datetime").reset_index(drop=True)

    out_cols = ["datetime", "V", "Bz", "P_dyn", "Dst_observed", "Dst_predicted", "Q", "tau"]
    bmr_df[out_cols].to_parquet(PARQUET, index=False)
    print(f"[ml] Сохранено: {PARQUET}")

    model = EmpiricalBMR()
    X, y, dates, n_dropped = build_windows(bmr_df, model)
    if X.size == 0:
        print("[ml] Не удалось построить окна.")
        return 1

    years = pd.DatetimeIndex(dates).year.to_numpy()
    masks = {name: (years >= lo) & (years <= hi) for name, (lo, hi) in SPLIT_RANGES.items()}

    mean, std = normalize(X[masks["train"]])
    norm = {
        "features": FEATURES,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "y_mean": float(y[masks["train"]].mean()),
        "y_std": float(y[masks["train"]].std()),
        "train_windows": int(masks["train"].sum()),
    }
    with open(NORM_JSON, "w", encoding="utf-8") as handle:
        json.dump(norm, handle, indent=2)
    print(f"[ml] Сохранено: {NORM_JSON}")

    for name in ("train", "val", "test"):
        m = masks[name]
        Xn = apply_norm(X[m], mean, std)
        _save_split(os.path.join(PROCESSED_DIR, f"ml_{name}.npz"), Xn, y[m], dates[m])
        print(f"[ml] {name}: {int(m.sum())} окон -> ml_{name}.npz")

    train_mean = apply_norm(X[masks["train"]], mean, std).reshape(-1, X.shape[-1]).mean(axis=0)
    print(f"[ml] Нормализация train: features={FEATURES}")
    print(f"[ml] mean(std) train после нормировки: {np.abs(train_mean).max():.3e}")
    print(f"[ml] Всего окон: {len(y)} (train {int(masks['train'].sum())}, val {int(masks['val'].sum())}, test {int(masks['test'].sum())})")
    print(f"[ml] residual: mean={y.mean():.2f}, std={y.std():.2f}, min={y.min():.2f}, max={y.max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
