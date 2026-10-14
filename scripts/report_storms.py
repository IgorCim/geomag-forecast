"""Агрегация метрик по бурям, визуализация и физические выводы.

Читает data/processed/storm_metrics.csv, строит сводку и 3 графика:
  * scatter_peak_error.png          - observed_min vs predicted_min;
  * histogram_peak_error.png        - распределение peak_error;
  * skill_score_vs_strength.png     - skill score vs min_dst.
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

METRICS_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "storm_metrics.csv")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def strength_class(min_dst: float) -> str:
    if min_dst >= -100.0:
        return "weak"
    if min_dst >= -200.0:
        return "medium"
    return "strong"


def main() -> int:
    df = pd.read_csv(METRICS_CSV)
    n = len(df)
    df["strength"] = df["min_dst"].apply(strength_class)

    weak = df[df["strength"] == "weak"]
    medium = df[df["strength"] == "medium"]
    strong = df[df["strength"] == "strong"]

    def ms(series: pd.Series) -> tuple[float, float]:
        return float(series.mean()), float(series.std())

    rmse_m, rmse_s = ms(df["rmse"])
    mae_m, mae_s = ms(df["mae"])
    r_m, r_s = ms(df["r"])
    pe_m, pe_s = ms(df["peak_error"])
    lag_m, lag_s = ms(df["lag_hours"])
    sk_m, sk_s = ms(df["skill_score"])

    predicted_min = df["min_dst"] + df["peak_error"]
    corr_strength = float(np.corrcoef(df["min_dst"], df["peak_error"])[0, 1])
    better = int((df["skill_score"] > 0).sum())

    _plot_scatter(df, predicted_min)
    _plot_hist(df)
    _plot_skill(df)

    print()
    print("=== ЭТАП 1.6: ВАЛИДАЦИЯ НА МНОГИХ БУРЯХ ===")
    print()
    print("1. ДАННЫЕ:")
    print(f"   - Бурь обработано: {n} (обработаны окна по каждой буре)")
    print()
    print("2. КАТАЛОГ БУРЬ:")
    print(f"   - Найдено бурь (Dst < -50): {n}")
    print(f"   - Слабых (-50..-100): {len(weak)}")
    print(f"   - Средних (-100..-200): {len(medium)}")
    print(f"   - Сильных (<-200): {len(strong)}")
    print()
    print("3. МЕТРИКИ (агрегат):")
    print(f"   - RMSE: {rmse_m:.2f} ± {rmse_s:.2f} нТл")
    print(f"   - MAE: {mae_m:.2f} ± {mae_s:.2f} нТл")
    print(f"   - R: {r_m:.3f} ± {r_s:.3f}")
    print(f"   - Peak error: {pe_m:.2f} ± {pe_s:.2f} нТл")
    print(f"   - Lag: {lag_m:.2f} ± {lag_s:.2f} ч")
    print(f"   - Skill score: {sk_m:.3f} ± {sk_s:.3f}")
    print()
    print("4. МЕТРИКИ ПО СИЛЕ:")
    for label, sub in (("Слабые (-50..-100)", weak), ("Средние (-100..-200)", medium), ("Сильные (<-200)", strong)):
        if len(sub) == 0:
            print(f"   - {label}: нет бурь")
            continue
        print(
            f"   - {label}: n={len(sub)}, RMSE={sub['rmse'].mean():.2f}, "
            f"peak_error={sub['peak_error'].mean():.2f}, skill={sub['skill_score'].mean():.3f}"
        )
    print()
    print("5. ФИЗИЧЕСКИЕ ВЫВОДЫ:")
    systematic = "да" if pe_m < -10.0 else "нет"
    has_lag = "да" if lag_m > 0.5 else "нет"
    print(f"   - Систематическое занижение: {systematic} (средний peak_error={pe_m:.2f} нТл)")
    print(f"   - Временной лаг: {has_lag} (средний lag={lag_m:.2f} ч)")
    print(f"   - Зависимость от силы: корреляция(peak_error, min_dst)={corr_strength:.3f}")
    print(f"   - Где модель лучше persistence: {better} из {n} бурь (skill_score > 0)")
    print()
    print("6. ГРАФИКИ:")
    print("   - scatter_peak_error.png: сохранён")
    print("   - histogram_peak_error.png: сохранён")
    print("   - skill_score_vs_strength.png: сохранён")
    print()
    print("7. БЛОКЕРЫ:")
    print("   нет")
    return 0, {
        "n": n,
        "weak": len(weak),
        "medium": len(medium),
        "strong": len(strong),
        "rmse_m": rmse_m,
        "rmse_s": rmse_s,
        "mae_m": mae_m,
        "mae_s": mae_s,
        "r_m": r_m,
        "r_s": r_s,
        "pe_m": pe_m,
        "pe_s": pe_s,
        "lag_m": lag_m,
        "lag_s": lag_s,
        "sk_m": sk_m,
        "sk_s": sk_s,
        "corr": corr_strength,
        "better": better,
        "weak_rmse": float(weak["rmse"].mean()) if len(weak) else float("nan"),
        "weak_pe": float(weak["peak_error"].mean()) if len(weak) else float("nan"),
        "weak_sk": float(weak["skill_score"].mean()) if len(weak) else float("nan"),
        "med_rmse": float(medium["rmse"].mean()) if len(medium) else float("nan"),
        "med_pe": float(medium["peak_error"].mean()) if len(medium) else float("nan"),
        "med_sk": float(medium["skill_score"].mean()) if len(medium) else float("nan"),
        "str_rmse": float(strong["rmse"].mean()) if len(strong) else float("nan"),
        "str_pe": float(strong["peak_error"].mean()) if len(strong) else float("nan"),
        "str_sk": float(strong["skill_score"].mean()) if len(strong) else float("nan"),
    }


def _plot_scatter(df: pd.DataFrame, predicted_min: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(df["min_dst"], predicted_min, c=df["year"], cmap="viridis", s=28, alpha=0.8)
    lims = [df["min_dst"].min() - 20, max(df["min_dst"].max(), predicted_min.max()) + 20]
    lims[0] = min(lims[0], -100)
    ax.plot(lims, lims, "k--", linewidth=1.2, label="ideal (y=x)")
    ax.axhline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.axvline(0.0, color="gray", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Observed Dst minimum (nT)")
    ax.set_ylabel("Predicted Dst minimum (nT)")
    ax.set_title("BMR predicted vs observed storm minima")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Year")
    fig.tight_layout()
    fig.savefig(os.path.join(PROCESSED_DIR, "scatter_peak_error.png"), dpi=130)
    plt.close(fig)


def _plot_hist(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(df["peak_error"], bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    ax.axvline(0.0, color="red", linewidth=1.6, label="ideal (0)")
    ax.axvline(df["peak_error"].mean(), color="orange", linewidth=1.6, linestyle="--",
               label=f"mean = {df['peak_error'].mean():.1f} nT")
    ax.set_xlabel("Peak error = predicted_min - observed_min (nT)")
    ax.set_ylabel("Number of storms")
    ax.set_title("Distribution of peak error across storms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PROCESSED_DIR, "histogram_peak_error.png"), dpi=130)
    plt.close(fig)


def _plot_skill(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(df["min_dst"], df["skill_score"], c=df["year"], cmap="viridis", s=28, alpha=0.8)
    ax.axhline(0.0, color="red", linewidth=1.6, label="persistence (SS=0)")
    ax.set_xlabel("Storm strength: observed Dst minimum (nT)")
    ax.set_ylabel("Skill score vs persistence")
    ax.set_title("Skill score vs storm strength")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.colorbar(sc, ax=ax, label="Year")
    fig.tight_layout()
    fig.savefig(os.path.join(PROCESSED_DIR, "skill_score_vs_strength.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main()[0])
