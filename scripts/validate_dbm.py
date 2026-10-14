"""Валидация DBM на Earth-directed CME по onset SYM-H/Dst (Этап 2.7).

Методика:
  * V_sw -- median V в OMNI за 24 ч до старта CME (fallback 400);
  * DBM (gamma = 0.5e-7) даёт predicted arrival на L1;
  * реальный onset бури -- первый час падения Dst > 20 нТл за 3 ч;
  * onset ищется в окне [predicted - 24 ч, predicted + 24 ч];
  * dt = predicted - onset; hit -- Dst < -30 в течение +/-12 ч от onset.

SYM-H в часовом OMNI2-файле отсутствует (индекс 41 -- это AE), поэтому
используется часовой Dst (fallback, как разрешено условием).

Выход:
  data/processed/dbm_metrics.csv  -- сопоставленные события (без NaN);
  data/processed/dbm_summary.csv  -- агрегированные метрики (одна строка).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, SCRIPTS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from download_cme_catalog import filter_earth_directed
from find_storms import load_all_years
from models.drag_based_model import DEFAULT_VSW, DragBasedModel

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
CME_CSV = os.path.join(RAW_DIR, "cme_catalog.csv")
OUT_CSV = os.path.join(PROCESSED_DIR, "dbm_metrics.csv")
SUMMARY_CSV = os.path.join(PROCESSED_DIR, "dbm_summary.csv")

START_YEAR = 2010
END_YEAR = 2024
HORIZONS = (6, 12, 18)
ONSET_THRESHOLD = -20.0
ONSET_WINDOW_H = 3
SEARCH_WINDOW_H = 24.0
DST_STORM_THRESHOLD = -30.0
DST_HIT_WINDOW_H = 12.0


def load_cme_catalog() -> pd.DataFrame:
    df = pd.read_csv(CME_CSV)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    return df


def vsw_window_has_data(start_time, omni_times: np.ndarray, window_hours: float = 24.0) -> bool:
    start = pd.Timestamp(start_time)
    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    start_np = start.to_datetime64()
    window = np.timedelta64(int(window_hours), "h")
    lo_idx = int(np.searchsorted(omni_times, start_np - window, side="left"))
    hi_idx = int(np.searchsorted(omni_times, start_np, side="right"))
    return hi_idx > lo_idx


def _onset_mask(values: np.ndarray, threshold: float = ONSET_THRESHOLD, window_hours: int = ONSET_WINDOW_H) -> np.ndarray:
    """Маска часов, с которых начинается падение Dst > |threshold| за window_hours."""
    values = np.asarray(values, dtype=float)
    n = values.size
    w = int(window_hours)
    onset = np.zeros(n, dtype=bool)
    if n <= w:
        return onset
    drop = values[:-w] - values[w:]
    cond = np.isfinite(drop) & (drop > -float(threshold))
    onset[w:] = cond & ~np.concatenate(([False], cond[:-1]))
    return onset


def detect_onsets(dst_values, times, threshold: float = ONSET_THRESHOLD, window_hours: int = ONSET_WINDOW_H):
    """Возвращает (onset_times: datetime64[], onset_idx: int[]) для всего ряда."""
    values = np.asarray(dst_values, dtype=float)
    times = np.asarray(times)
    mask = _onset_mask(values, threshold, window_hours)
    idx = np.where(mask)[0]
    return times[idx], idx


def detect_storm_onset(dst_series, threshold: float = ONSET_THRESHOLD, window_hours: int = ONSET_WINDOW_H):
    """Первый onset в ряде. dst_series -- pandas.Series с DatetimeIndex. Иначе None."""
    if isinstance(dst_series, pd.Series):
        values = dst_series.to_numpy(dtype=float)
        times = dst_series.index.to_numpy()
    else:
        values = np.asarray(dst_series, dtype=float)
        times = np.arange(values.size)

    onset_times, _ = detect_onsets(values, times, threshold, window_hours)
    if onset_times.size == 0:
        return None
    return pd.Timestamp(onset_times[0])


def first_onset_in_window(onset_times: np.ndarray, center_np, half_hours: float = SEARCH_WINDOW_H):
    """Первый onset в окне +/- half_hours вокруг center. Иначе None."""
    if onset_times.size == 0:
        return None
    center_np = np.datetime64(center_np)
    half = np.timedelta64(int(half_hours), "h")
    k = int(np.searchsorted(onset_times, center_np - half, side="left"))
    if k >= onset_times.size or onset_times[k] > center_np + half:
        return None
    return pd.Timestamp(onset_times[k])


def find_peak(times: np.ndarray, v: np.ndarray, center_np, half_hours: float = 24.0) -> tuple:
    """Пик V в окне +/- half_hours (сохранён для обратной совместимости Этапа 2.6)."""
    half = np.timedelta64(int(half_hours), "h")
    lo_idx = int(np.searchsorted(times, center_np - half, side="left"))
    hi_idx = int(np.searchsorted(times, center_np + half, side="right"))
    if hi_idx <= lo_idx:
        return None
    seg = v[lo_idx:hi_idx]
    seg_t = times[lo_idx:hi_idx]
    finite = np.isfinite(seg)
    if not finite.any():
        return None
    valid = np.where(finite)[0]
    k = int(valid[np.argmax(seg[valid])])
    v_peak = float(seg[k])
    t_peak = seg_t[k]
    base_lo = int(np.searchsorted(times, center_np - np.timedelta64(int(half_hours * 2), "h"), side="left"))
    base_hi = int(np.searchsorted(times, center_np - half, side="right"))
    base = v[base_lo:base_hi]
    base = base[np.isfinite(base)]
    baseline = float(np.median(base)) if base.size else float(np.nanmin(seg[finite]))
    if v_peak - baseline < 50.0:
        return None
    return t_peak, v_peak


def dst_in_window(times: np.ndarray, dst: np.ndarray, center_np, half_hours: float) -> float:
    window = np.timedelta64(int(half_hours), "h")
    lo_idx = int(np.searchsorted(times, center_np - window, side="left"))
    hi_idx = int(np.searchsorted(times, center_np + window, side="right"))
    if hi_idx <= lo_idx:
        return float("nan")
    segment = dst[lo_idx:hi_idx]
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return float("nan")
    return float(finite.min())


def run(catalog: pd.DataFrame, omni: pd.DataFrame, model: DragBasedModel) -> tuple[pd.DataFrame, dict]:
    times = omni["datetime"].to_numpy(dtype="datetime64[ns]")
    dst_series = omni["Dst"].to_numpy(dtype=float)
    hour = np.timedelta64(1, "h")

    onset_times, _ = detect_onsets(dst_series, times)

    records = []
    n_vsw_omni = 0
    n_matched = 0
    n_hit = 0

    for _, row in catalog.iterrows():
        cme_row = {"cme_id": row["cme_id"], "start_time": row["start_time"], "v_linear": row["v_linear"]}
        res = model.predict_arrival_with_vsw(cme_row, omni)
        v_sw = float(res["v_sw"])
        if vsw_window_has_data(row["start_time"], times):
            n_vsw_omni += 1

        t_pred = pd.Timestamp(res["arrival_time_l1"]).tz_convert("UTC").tz_localize(None)
        t_pred_np = t_pred.to_datetime64()

        onset = first_onset_in_window(onset_times, t_pred_np, SEARCH_WINDOW_H)
        if onset is None:
            continue

        onset_np = onset.to_datetime64()
        dt_onset = float((t_pred_np - onset_np) / hour)
        dst_min = dst_in_window(times, dst_series, onset_np, DST_HIT_WINDOW_H)
        hit = bool(np.isfinite(dst_min) and dst_min < DST_STORM_THRESHOLD)

        n_matched += 1
        if hit:
            n_hit += 1

        records.append(
            {
                "cme_id": row["cme_id"],
                "start_time": row["start_time"],
                "v0": float(row["v_linear"]),
                "v_sw": v_sw,
                "t_pred": t_pred,
                "t_onset": onset,
                "dt_onset": dt_onset,
                "v_impact": float(res["v_impact"]),
                "dst_min_window": dst_min,
                "hit": int(hit),
            }
        )

    matched = pd.DataFrame(records)
    stats = {"n_vsw_omni": n_vsw_omni, "n_matched": n_matched, "n_hit": n_hit, "n_onsets": int(onset_times.size)}
    return matched, stats


def main() -> int:
    print(f"[dbm] Загрузка каталога CME: {CME_CSV}")
    catalog_all = load_cme_catalog()
    catalog = filter_earth_directed(catalog_all)
    print(f"[dbm] Всего CME: {len(catalog_all)}; Earth-directed: {len(catalog)}")

    print(f"[dbm] Загрузка OMNI2 {START_YEAR}-{END_YEAR}...")
    omni = load_all_years(START_YEAR, END_YEAR)
    print(f"[dbm] Строк OMNI: {len(omni)}")

    model = DragBasedModel()
    matched, stats = run(catalog, omni, model)
    if matched.empty:
        print("[dbm] Не удалось сопоставить ни одного события.")
        return 1

    matched = matched.sort_values("start_time").reset_index(drop=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    matched.to_csv(OUT_CSV, index=False)

    n_earth = len(catalog)
    abs_dt = matched["dt_onset"].abs()
    dt_mean = float(matched["dt_onset"].mean())
    dt_std = float(matched["dt_onset"].std())
    dt_median = float(matched["dt_onset"].median())
    vsw_mean = float(matched["v_sw"].mean())
    pct = {h: float((abs_dt <= h).mean()) * 100.0 for h in HORIZONS}
    hit_rate_all = float(stats["n_hit"]) / n_earth * 100.0
    hit_rate_matched = float(stats["n_hit"]) / len(matched) * 100.0

    summary = pd.DataFrame(
        [
            {
                "n_total": len(catalog_all),
                "n_earth_directed": n_earth,
                "n_side": len(catalog_all) - n_earth,
                "n_halo": int((catalog["angular_width"] >= 300).sum()),
                "n_vsw_omni": int(stats["n_vsw_omni"]),
                "n_onsets_total": int(stats["n_onsets"]),
                "n_matched": len(matched),
                "vsw_mean": vsw_mean,
                "dt_mean": dt_mean,
                "dt_std": dt_std,
                "dt_median": dt_median,
                "pct_6": pct[6],
                "pct_12": pct[12],
                "pct_18": pct[18],
                "hit_rate_all": hit_rate_all,
                "hit_rate_matched": hit_rate_matched,
                "onset_lag_mean": dt_mean,
            }
        ]
    )
    summary.to_csv(SUMMARY_CSV, index=False)

    print(f"[dbm] V_sw из OMNI: {stats['n_vsw_omni']} / {n_earth}; mean V_sw={vsw_mean:.1f} км/с")
    print(f"[dbm] Earth-directed: {n_earth}; onset найден: {len(matched)}")
    print(f"[dbm] dt (onset): mean={dt_mean:+.2f} ч, std={dt_std:.2f} ч, median={dt_median:+.2f} ч")
    for h in HORIZONS:
        print(f"[dbm] |dt| <= {h} ч: {pct[h]:.1f}%")
    print(f"[dbm] Hit rate по Dst (все): {hit_rate_all:.1f}%; (сопоставленные): {hit_rate_matched:.1f}%")
    print(f"[dbm] Сохранено: {OUT_CSV}")
    print(f"[dbm] Сохранено: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
