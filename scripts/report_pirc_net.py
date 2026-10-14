"""Визуализация результатов PiRCNet (Этап 3.2).

Строит 5 графиков:
  * training_curves.png
  * pred_vs_obs_test.png
  * timeseries_examples.png
  * residual_histogram_compare.png
  * peak_error_compare.png

Запуск:
    python scripts/report_pirc_net.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
HISTORY = os.path.join(PROCESSED_DIR, "training_history.json")
REPORT_JSON = os.path.join(PROCESSED_DIR, "evaluation_report.json")
PREDS_NPZ = os.path.join(PROCESSED_DIR, "test_predictions.npz")

EXAMPLES = ["2024-05", "2023-04", "2023-11"]


def plot_training_curves() -> None:
    with open(HISTORY, encoding="utf-8") as handle:
        hist = json.load(handle)["history"]
    epochs = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, [h["train_loss"] for h in hist], label="train_loss")
    ax.plot(epochs, [h["val_loss"] for h in hist], label="val_loss")
    best = min(hist, key=lambda h: h["val_loss"])
    ax.axvline(best["epoch"], color="gray", ls="--", label=f"best epoch={best['epoch']}")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Weighted MSE")
    ax.set_title("Кривые обучения PiRCNet")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "training_curves.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_pred_vs_obs(data: dict) -> None:
    obs, bmr, nn = data["dst_observed"], data["dst_bmr"], data["dst_nn"]
    fig, ax = plt.subplots(figsize=(7, 7))
    lo = min(obs.min(), bmr.min(), nn.min()) - 5
    hi = max(obs.max(), bmr.max(), nn.max()) + 5
    ax.scatter(obs, bmr, s=4, alpha=0.2, color="tab:blue", label="BMR", rasterized=True)
    ax.scatter(obs, nn, s=4, alpha=0.2, color="tab:red", label="BMR+NN", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Dst observed, нТл")
    ax.set_ylabel("Dst predicted, нТл")
    ax.set_title("Predicted vs Observed (тест 2022-2024)")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "pred_vs_obs_test.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_timeseries_examples(data: dict) -> None:
    frame = pd.DataFrame(data).set_index("datetime")
    fig, axes = plt.subplots(len(EXAMPLES), 1, figsize=(12, 10))
    for ax, tag in zip(axes, EXAMPLES):
        window = frame[frame.index.strftime("%Y-%m") == tag]
        if window.empty:
            continue
        ax.plot(window.index, window["dst_observed"], "k", label="observed")
        ax.plot(window.index, window["dst_bmr"], color="tab:blue", label="BMR")
        ax.plot(window.index, window["dst_nn"], color="tab:red", label="BMR+NN")
        ax.set_title(f"{tag}: min obs={window['dst_observed'].min():.0f} нТл")
        ax.set_ylabel("Dst, нТл")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "timeseries_examples.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_residual_hist(data: dict) -> None:
    res_bmr = data["dst_observed"] - data["dst_bmr"]
    res_nn = data["dst_observed"] - data["dst_nn"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(-120, 80, 120)
    ax.hist(res_bmr, bins=bins, alpha=0.5, color="tab:blue", label=f"BMR (std={res_bmr.std():.2f})")
    ax.hist(res_nn, bins=bins, alpha=0.5, color="tab:red", label=f"BMR+NN (std={res_nn.std():.2f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("residual, нТл")
    ax.set_ylabel("частота")
    ax.set_title("Residual: до и после коррекции")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "residual_histogram_compare.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_peak_error() -> None:
    with open(REPORT_JSON, encoding="utf-8") as handle:
        storms = json.load(handle)["storms_top10"]
    labels = [s["date"] for s in storms]
    bmr = [s["peak_err_bmr"] for s in storms]
    nn = [s["peak_err_nn"] for s in storms]
    x = np.arange(len(storms))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 0.2, bmr, width=0.4, color="tab:blue", label="BMR")
    ax.bar(x + 0.2, nn, width=0.4, color="tab:red", label="BMR+NN")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("peak error |pred_min - obs_min|, нТл")
    ax.set_title("Peak error по 10 сильнейшим бурям теста")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "peak_error_compare.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def main() -> int:
    with np.load(PREDS_NPZ) as npz:
        data = {
            "datetime": pd.DatetimeIndex(npz["dates"]),
            "dst_observed": npz["dst_observed"].astype(float),
            "dst_bmr": npz["dst_bmr"].astype(float),
            "dst_nn": npz["dst_nn"].astype(float),
        }
    plot_training_curves()
    plot_pred_vs_obs(data)
    plot_timeseries_examples(data)
    plot_residual_hist(data)
    plot_peak_error()
    return 0


if __name__ == "__main__":
    sys.exit(main())
