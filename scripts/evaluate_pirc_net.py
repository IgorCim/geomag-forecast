"""Оценка обученного PiRCNet на тесте 2022-2024 (Этап 3.2).

Запуск:
    python scripts/evaluate_pirc_net.py

Считает метрики для BMR и для BMR+NN, peak error по бурям из storm_catalog.csv,
skill score. Сохраняет evaluation_report.json и test_predictions.npz.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ml_dataset import load_dst_observed, load_npz_split
from models.pirc_net import PiRCNet

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PARQUET = os.path.join(PROCESSED_DIR, "omni_with_bmr.parquet")
CATALOG = os.path.join(PROCESSED_DIR, "storm_catalog.csv")
BEST_PATH = os.path.join(MODELS_DIR, "pirc_net_best.pt")
REPORT_JSON = os.path.join(PROCESSED_DIR, "evaluation_report.json")
PREDS_NPZ = os.path.join(PROCESSED_DIR, "test_predictions.npz")

STORM_DST = -50.0
QUIET_DST = -30.0
TEST_YEARS = (2022, 2024)


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict:
    err = predicted - observed
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "r": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def main() -> int:
    if not os.path.exists(BEST_PATH):
        print(f"[eval] Нет обученной модели: {BEST_PATH}")
        return 1

    X, y, dates = load_npz_split(os.path.join(PROCESSED_DIR, "ml_test.npz"))
    dst_obs = load_dst_observed(dates, PARQUET).astype(np.float64)

    frame = pd.read_parquet(PARQUET, columns=["datetime", "Dst_predicted"]).set_index("datetime")
    dst_bmr = frame.reindex(pd.DatetimeIndex(dates))["Dst_predicted"].to_numpy(dtype=np.float64)

    model = PiRCNet()
    model.load_state_dict(torch.load(BEST_PATH, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        correction = model(torch.from_numpy(X.astype(np.float32))).numpy().astype(np.float64)
    dst_nn = dst_bmr + correction

    mask_storm = dst_obs < STORM_DST
    mask_quiet = dst_obs > QUIET_DST

    rows = {}
    for name, pred in (("bmr", dst_bmr), ("nn", dst_nn)):
        rows[name] = {
            "overall": metrics(dst_obs, pred),
            "storm": metrics(dst_obs[mask_storm], pred[mask_storm]),
            "quiet": metrics(dst_obs[mask_quiet], pred[mask_quiet]),
        }

    skill = 1.0 - rows["nn"]["overall"]["rmse"] / rows["bmr"]["overall"]["rmse"]

    series = pd.DataFrame({"datetime": pd.DatetimeIndex(dates), "obs": dst_obs, "bmr": dst_bmr, "nn": dst_nn})
    series = series.set_index("datetime")

    catalog = pd.read_csv(CATALOG)
    catalog["start_time"] = pd.to_datetime(catalog["start_time"])
    catalog["end_time"] = pd.to_datetime(catalog["end_time"])
    catalog = catalog[(catalog["year"] >= TEST_YEARS[0]) & (catalog["year"] <= TEST_YEARS[1])]

    storms = []
    for _, storm in catalog.iterrows():
        window = series[(series.index >= storm["start_time"]) & (series.index <= storm["end_time"])]
        if window.empty:
            continue
        obs_min = float(window["obs"].min())
        bmr_min = float(window["bmr"].min())
        nn_min = float(window["nn"].min())
        storms.append(
            {
                "date": storm["start_time"].strftime("%Y-%m-%d"),
                "dst_obs_min": obs_min,
                "bmr_min": bmr_min,
                "nn_min": nn_min,
                "peak_err_bmr": abs(bmr_min - obs_min),
                "peak_err_nn": abs(nn_min - obs_min),
            }
        )

    peak_bmr = float(np.mean([s["peak_err_bmr"] for s in storms])) if storms else float("nan")
    peak_nn = float(np.mean([s["peak_err_nn"] for s in storms])) if storms else float("nan")
    top10 = sorted(storms, key=lambda s: s["dst_obs_min"])[:10]

    report = {
        "device": "cpu",
        "n_test": int(len(dst_obs)),
        "n_storms_test": len(storms),
        "skill_score": float(skill),
        "peak_error_bmr": peak_bmr,
        "peak_error_nn": peak_nn,
        "metrics": rows,
        "storms_top10": top10,
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    np.savez_compressed(
        PREDS_NPZ,
        dates=pd.DatetimeIndex(dates).to_numpy().astype("datetime64[ns]"),
        dst_observed=dst_obs.astype(np.float32),
        dst_bmr=dst_bmr.astype(np.float32),
        dst_nn=dst_nn.astype(np.float32),
    )

    print(f"[eval] Тест: {len(dst_obs)} окон, бурь: {len(storms)}")
    for split in ("overall", "storm", "quiet"):
        b, n = rows["bmr"][split], rows["nn"][split]
        print(f"[eval] {split:7s}: RMSE {b['rmse']:.2f} -> {n['rmse']:.2f} | MAE {b['mae']:.2f} -> {n['mae']:.2f} | R {b['r']:.3f} -> {n['r']:.3f}")
    print(f"[eval] Peak error: BMR={peak_bmr:.2f} -> NN={peak_nn:.2f}")
    print(f"[eval] Skill score = {skill:.4f}")
    print(f"[eval] Сохранено: {REPORT_JSON}, {PREDS_NPZ}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
