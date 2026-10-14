"""Валидация модели BMR на всех найденных бурях.

Для каждой бури из ``storm_catalog.csv``:
  * окно валидации: [min_time - 12ч, min_time + 24ч];
  * начальное условие: Dst*(t_start) = Dst_obs(t_start) - b*sqrt(P_dyn) + c;
  * прогон EmpiricalBMR.predict();
  * метрики: RMSE, MAE, R, peak_error, lag, skill score vs persistence.

Результат: data/processed/storm_metrics.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from find_storms import END_YEAR, START_YEAR, load_all_years
from models.empirical_bmr import EmpiricalBMR
from utils.metrics import evaluate_all

CATALOG_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storm_catalog.csv")
OUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storm_metrics.csv")

PRE_HOURS = 12
POST_HOURS = 24


def persistence_baseline(observed: np.ndarray) -> np.ndarray:
    """Dst(t) = Dst(t-1)."""
    observed = np.asarray(observed, dtype=float)
    baseline = np.empty_like(observed)
    baseline[0] = observed[0]
    baseline[1:] = observed[:-1]
    return baseline


def hours_between(t_later: pd.Timestamp, t_earlier: pd.Timestamp) -> float:
    """Разница во времени в часах."""
    return (t_later - t_earlier).total_seconds() / 3600.0


def main() -> int:
    catalog = pd.read_csv(CATALOG_CSV, parse_dates=["min_time", "start_time", "end_time"])
    print(f"[validate_all] Бурь в каталоге: {len(catalog)}")

    df = load_all_years(START_YEAR, END_YEAR)
    model = EmpiricalBMR()

    records = []
    for row in catalog.itertuples(index=False):
        min_time = row.min_time
        w_start = min_time - pd.Timedelta(hours=PRE_HOURS)
        w_end = min_time + pd.Timedelta(hours=POST_HOURS)

        window = df[(df["datetime"] >= w_start) & (df["datetime"] <= w_end)].reset_index(drop=True)
        if len(window) < 5:
            print(f"[validate_all] Бурь {row.storm_id}: слишком мало данных ({len(window)}), пропуск")
            continue

        result = model.predict(window)
        observed = result["Dst_observed"].to_numpy(dtype=float)
        predicted = result["Dst_predicted"].to_numpy(dtype=float)
        times = pd.to_datetime(result["datetime"])

        metrics = evaluate_all(observed, predicted)
        base = persistence_baseline(observed)
        base_metrics = evaluate_all(observed, base)

        i_obs = int(np.nanargmin(observed))
        i_pred = int(np.nanargmin(predicted))
        peak_error = float(predicted[i_pred] - observed[i_obs])
        lag = hours_between(times.iloc[i_pred], times.iloc[i_obs])

        rmse_model = metrics["rmse"]
        rmse_base = base_metrics["rmse"]
        skill = 1.0 - rmse_model / rmse_base if rmse_base > 0 else float("nan")

        records.append(
            {
                "storm_id": row.storm_id,
                "year": row.year,
                "min_dst": row.min_dst,
                "rmse": rmse_model,
                "mae": metrics["mae"],
                "r": metrics["correlation"],
                "peak_error": peak_error,
                "lag_hours": lag,
                "skill_score": skill,
            }
        )

    out = pd.DataFrame(records)
    out.to_csv(OUT_CSV, index=False)
    print(f"[validate_all] Обработано бурь: {len(out)}")
    print(f"[validate_all] Метрики сохранены: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
