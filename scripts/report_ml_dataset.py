"""Отчёт по ML-датасету residual BMR (Этап 3.1).

Строит 4 графика и печатает статистику residual:
  * residual_timeseries.png
  * residual_histogram.png
  * residual_vs_dst.png
  * residual_example_storms.png

Запуск:
    python scripts/report_ml_dataset.py
"""

from __future__ import annotations

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
PARQUET = os.path.join(PROCESSED_DIR, "omni_with_bmr.parquet")
NORM_JSON = os.path.join(PROCESSED_DIR, "norm_params.json")

STORM_DST = -50.0
QUIET_DST = -30.0


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    df["residual"] = df["Dst_observed"] - df["Dst_predicted"]
    bs = np.where(df["Bz"].to_numpy() < 0.0, -df["Bz"].to_numpy(), 0.0)
    df["Ey"] = df["V"].to_numpy() * bs / 1000.0
    return df


def describe(values: np.ndarray) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p5": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "n": int(values.size),
    }


def plot_timeseries(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df["datetime"], df["residual"], lw=0.4, color="tab:red")
    for year, lo, hi in (("train", 2000, 2018), ("val", 2019, 2021), ("test", 2022, 2024)):
        ax.axvspan(pd.Timestamp(f"{lo}-01-01"), pd.Timestamp(f"{hi}-12-31"), alpha=0.06, label=year)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Время")
    ax.set_ylabel("residual = Dst_obs - Dst_BMR, нТл")
    ax.set_title("Residual BMR по времени (2000-2024)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "residual_timeseries.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_histogram(df: pd.DataFrame) -> None:
    residual = df["residual"].to_numpy()
    stats = describe(residual)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residual, bins=120, color="tab:blue", alpha=0.8)
    ax.axvline(stats["mean"], color="tab:red", ls="--", label=f"mean={stats['mean']:.2f}")
    ax.axvline(stats["p5"], color="gray", ls=":", label=f"p5={stats['p5']:.1f}")
    ax.axvline(stats["p95"], color="gray", ls=":", label=f"p95={stats['p95']:.1f}")
    ax.set_xlabel("residual, нТл")
    ax.set_ylabel("частота")
    ax.set_title(f"Гистограмма residual (std={stats['std']:.2f}, n={stats['n']})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "residual_histogram.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_residual_vs_dst(df: pd.DataFrame) -> None:
    x = df["Dst_predicted"].to_numpy()
    y = df["residual"].to_numpy()
    corr = float(np.corrcoef(x, y)[0, 1])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=3, alpha=0.15, color="tab:purple", rasterized=True)
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 40)
    idx = np.digitize(x, bins)
    centers, medians = [], []
    for b in range(1, len(bins)):
        sel = idx == b
        if sel.sum() > 20:
            centers.append(0.5 * (bins[b - 1] + bins[b]))
            medians.append(np.median(y[sel]))
    ax.plot(centers, medians, color="tab:red", lw=2, label="медиана по бинам")
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Dst_predicted (BMR), нТл")
    ax.set_ylabel("residual, нТл")
    ax.set_title(f"Residual vs Dst (corr={corr:.3f})")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "residual_vs_dst.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def plot_example_storms(df: pd.DataFrame) -> None:
    storms = (
        df.assign(dst=df["Dst_observed"])
        .nsmallest(4000, "dst")["datetime"]
        .sort_values()
    )
    picked = []
    for ts in storms:
        if all(abs((ts - p).total_seconds()) > 5 * 86400 for p in picked):
            picked.append(ts)
        if len(picked) == 3:
            break

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, ts in zip(axes, picked):
        w = df[(df["datetime"] >= ts - pd.Timedelta(hours=24)) & (df["datetime"] <= ts + pd.Timedelta(hours=48))]
        ax.plot(w["datetime"], w["Dst_observed"], label="Dst observed", color="k")
        ax.plot(w["datetime"], w["Dst_predicted"], label="Dst BMR", color="tab:blue")
        ax.fill_between(w["datetime"], w["Dst_observed"], w["Dst_predicted"], color="tab:red", alpha=0.25, label="residual")
        ax.set_title(f"Буря {pd.Timestamp(ts):%Y-%m-%d} (min Dst={w['Dst_observed'].min():.0f} нТл)")
        ax.set_ylabel("Dst, нТл")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Время")
    fig.tight_layout()
    path = os.path.join(PROCESSED_DIR, "residual_example_storms.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"[report] Сохранено: {path}")


def main() -> int:
    df = load_data()
    residual = df["residual"].to_numpy()
    overall = describe(residual)
    storms = describe(residual[df["Dst_observed"].to_numpy() < STORM_DST])
    quiet = describe(residual[df["Dst_observed"].to_numpy() > QUIET_DST])
    corr_pred = float(np.corrcoef(df["Dst_predicted"], residual)[0, 1])
    corr_obs = float(np.corrcoef(df["Dst_observed"], residual)[0, 1])

    print(f"[report] Режим: n={overall['n']}")
    print(f"[report] RESIDUAL overall: mean={overall['mean']:.2f}, std={overall['std']:.2f}, "
          f"min={overall['min']:.2f}, max={overall['max']:.2f}, p5={overall['p5']:.2f}, p95={overall['p95']:.2f}")
    if storms:
        print(f"[report] RESIDUAL storms (Dst<-50): n={storms['n']}, mean={storms['mean']:.2f}, "
              f"std={storms['std']:.2f}, min={storms['min']:.2f}, max={storms['max']:.2f}")
    if quiet:
        print(f"[report] RESIDUAL quiet (Dst>-30): n={quiet['n']}, mean={quiet['mean']:.2f}, "
              f"std={quiet['std']:.2f}")
    print(f"[report] corr(residual, Dst_predicted)={corr_pred:.3f}; corr(residual, Dst_observed)={corr_obs:.3f}")

    plot_timeseries(df)
    plot_histogram(df)
    plot_residual_vs_dst(df)
    plot_example_storms(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
