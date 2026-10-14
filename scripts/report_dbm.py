"""Отчёт по валидации DBM на Earth-directed CME (Этап 2.7, onset Dst).

Вход:  data/processed/dbm_metrics.csv, data/processed/dbm_summary.csv
Выход: data/processed/dbm_trajectory_example.png
       data/processed/dbm_arrival_error_hist.png
       data/processed/dbm_onset_lag_hist.png
       data/processed/dbm_speed_error.png
       data/processed/dbm_hit_rate.png
       data/processed/dbm_validation_table.txt
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

from models.drag_based_model import DragBasedModel

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
METRICS_CSV = os.path.join(PROCESSED_DIR, "dbm_metrics.csv")
SUMMARY_CSV = os.path.join(PROCESSED_DIR, "dbm_summary.csv")


def plot_trajectory(model: DragBasedModel) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    for v0, color in ((400.0, "tab:blue"), (800.0, "tab:green"), (1500.0, "tab:red")):
        t_hit = model.time_to_l1(v0)
        traj = model.trajectory(v0, t_max=t_hit * 1.4)
        ax.plot(traj["time_hours"], traj["distance_au"], color=color, label=f"V0 = {v0:.0f} км/с")
        ax.plot([t_hit], [model.r_l1_au], "o", color=color)
    ax.axhline(model.r_l1_au, color="black", ls="--", lw=1, label="L1 (1 а.е.)")
    ax.set_xlabel("Время после старта, ч")
    ax.set_ylabel("Расстояние от Солнца, а.е.")
    ax.set_title("Траектория CME по модели DBM")
    ax.set_ylim(0, model.r_l1_au * 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(PROCESSED_DIR, "dbm_trajectory_example.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_arrival_error(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["dt_onset"], bins=25, color="tab:purple", alpha=0.8)
    for thr, color in ((6, "tab:green"), (12, "tab:orange"), (18, "tab:red")):
        ax.axvline(-thr, color=color, ls="--", lw=1)
        ax.axvline(thr, color=color, ls="--", lw=1)
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Δt = t_прогноз - t_onset, ч")
    ax.set_ylabel("Число событий")
    ax.set_title("Ошибка времени прихода DBM относительно onset Dst")
    ax.grid(alpha=0.3)
    out = os.path.join(PROCESSED_DIR, "dbm_arrival_error_hist.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_onset_lag(df: pd.DataFrame) -> str:
    lag = df["dt_onset"].to_numpy(dtype=float)
    mean = float(np.mean(lag))
    std = float(np.std(lag, ddof=1))
    median = float(np.median(lag))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(lag, bins=25, color="tab:cyan", alpha=0.8, edgecolor="black")
    ax.axvline(mean, color="tab:red", lw=2, label=f"mean = {mean:+.2f} ч")
    ax.axvline(median, color="tab:blue", lw=2, ls="--", label=f"median = {median:+.2f} ч")
    ax.axvline(0.0, color="black", lw=1)
    ax.set_xlabel("Лаг onset vs predicted, ч")
    ax.set_ylabel("Число событий")
    ax.set_title(f"Лаг onset Dst - прогноз DBM (σ = {std:.2f} ч)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(PROCESSED_DIR, "dbm_onset_lag_hist.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_speed_error(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["v0"], df["v_impact"], s=18, alpha=0.6, label="DBM (V на L1)")
    lim = [float(df["v0"].min()) * 0.9, float(df["v0"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="1:1 (без торможения)")
    ax.set_xlabel("Начальная скорость V0, км/с")
    ax.set_ylabel("Скорость DBM на L1, км/с")
    ax.set_title("Торможение CME по DBM (V0 -> V на L1)")
    ax.set_xlim(lim)
    ax.set_ylim(0, float(df["v0"].max()) * 1.1)
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(PROCESSED_DIR, "dbm_speed_error.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_hit_rate(summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    labels = ["±6 ч", "±12 ч", "±18 ч", "Dst hit"]
    values = [row["pct_6"], row["pct_12"], row["pct_18"], row["hit_rate_all"]]
    colors = ["tab:green", "tab:orange", "tab:red", "tab:blue"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}%", ha="center")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Доля событий, %")
    ax.set_title("DBM (onset Dst): попадания по времени и по Dst")
    ax.grid(alpha=0.3, axis="y")
    out = os.path.join(PROCESSED_DIR, "dbm_hit_rate.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def _fmt_row(r: pd.Series) -> str:
    return (
        f"{str(r['cme_id']):<28}{r['v0']:>7.0f}{r['v_impact']:>9.0f}"
        f"{r['dt_onset']:>9.1f}{r['dst_min_window']:>10.1f}{int(r['hit']):>5}"
    )


def write_table(df: pd.DataFrame, summary: pd.DataFrame) -> str:
    row = summary.iloc[0]
    lines = []
    lines.append("ОТЧЁТ ПО ВАЛИДАЦИИ DRAG-BASED MODEL (onset Dst, Этап 2.7)")
    lines.append("=" * 68)
    lines.append(
        f"Всего CME: {int(row['n_total'])}, Earth-directed: {int(row['n_earth_directed'])}, side: {int(row['n_side'])}"
    )
    lines.append(f"V_sw из OMNI: {int(row['n_vsw_omni'])} / {int(row['n_earth_directed'])}; mean V_sw = {row['vsw_mean']:.1f} км/с")
    lines.append(f"Onset найден (сопоставлено): {int(row['n_matched'])}")
    lines.append(f"Δt (onset): mean = {row['dt_mean']:+.2f} ч, std = {row['dt_std']:.2f} ч, median = {row['dt_median']:+.2f} ч")
    lines.append(f"Доля |Δt| <= 6 ч: {row['pct_6']:.1f}%")
    lines.append(f"Доля |Δt| <= 12 ч: {row['pct_12']:.1f}%")
    lines.append(f"Доля |Δt| <= 18 ч: {row['pct_18']:.1f}%")
    lines.append(f"Hit rate по Dst (все Earth-directed): {row['hit_rate_all']:.1f}%")
    lines.append(f"Hit rate по Dst (сопоставленные): {row['hit_rate_matched']:.1f}%")

    header = f"{'cme_id':<28}{'V0':>7}{'V_pred':>9}{'Δt,ч':>9}{'Dst_min':>10}{'hit':>5}"
    ranked = df.assign(abs_dt=df["dt_onset"].abs())
    best = ranked.sort_values("abs_dt", ascending=True).head(10)
    worst = ranked.sort_values("abs_dt", ascending=False).head(10)

    lines.append("")
    lines.append("Топ-10 ЛУЧШИХ по |Δt|:")
    lines.append(header)
    for _, r in best.iterrows():
        lines.append(_fmt_row(r))

    lines.append("")
    lines.append("Топ-10 ХУДШИХ по |Δt|:")
    lines.append(header)
    for _, r in worst.iterrows():
        lines.append(_fmt_row(r))

    text = "\n".join(lines) + "\n"
    out = os.path.join(PROCESSED_DIR, "dbm_validation_table.txt")
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    return out


def main() -> int:
    if not os.path.exists(METRICS_CSV) or not os.path.exists(SUMMARY_CSV):
        print("[report] Нет файлов метрик. Сначала запустите validate_dbm.py.")
        return 1

    df = pd.read_csv(METRICS_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    model = DragBasedModel()

    outs = [
        plot_trajectory(model),
        plot_arrival_error(df),
        plot_onset_lag(df),
        plot_speed_error(df),
        plot_hit_rate(summary),
        write_table(df, summary),
    ]
    for path in outs:
        print(f"[report] Сохранено: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
