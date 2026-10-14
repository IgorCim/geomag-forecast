"""Drag-Based Model (DBM) для прогноза времени прихода CME на L1.

Физика (изменять запрещено):
    V(t) = V_sw + (V0 - V_sw) / (1 + gamma * |V0 - V_sw| * t)
где t -- время в часах, V в км/с. Расстояние интегрируется численно,
время прихода на L1 ищется методом `scipy.optimize.brentq`.

Этап 2.6: V_sw берётся из реального OMNI перед CME, строится ансамбль по gamma.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

AU_KM = 1.495978707e8

DEFAULT_GAMMAS = (0.2e-7, 0.5e-7, 1.0e-7)
DEFAULT_VSW = 400.0
VSW_WINDOW_H = 24.0


class DragBasedModel:
    """Физическая модель торможения CME в солнечном ветре (DBM)."""

    def __init__(
        self,
        gamma: float = 0.5e-7,
        v_sw: float = 400.0,
        r_start_au: float = 0.1,
        r_l1_au: float = 1.0,
    ) -> None:
        self.gamma = float(gamma)
        self.v_sw = float(v_sw)
        self.r_start_au = float(r_start_au)
        self.r_l1_au = float(r_l1_au)

    def velocity(self, t, V0, gamma: float | None = None, v_sw: float | None = None):
        """Скорость CME (км/с) на момент t часов после старта."""
        gamma = self.gamma if gamma is None else float(gamma)
        v_sw = self.v_sw if v_sw is None else float(v_sw)
        t = np.asarray(t, dtype=float)
        return v_sw + (V0 - v_sw) / (1.0 + gamma * abs(V0 - v_sw) * t)

    def distance(self, t: float, V0: float, gamma: float | None = None, v_sw: float | None = None) -> float:
        """Расстояние от Солнца (а.е.) на момент t часов (численное интегрирование)."""
        integral, _ = quad(
            lambda x: float(self.velocity(x, V0, gamma, v_sw)),
            0.0,
            float(t),
            epsabs=1e-3,
            epsrel=1e-6,
            limit=200,
        )
        return self.r_start_au + integral * 3600.0 / AU_KM

    def time_to_l1(self, V0: float, gamma: float | None = None, v_sw: float | None = None) -> float:
        """Время прихода на L1 (часы) от r_start_au до r_l1_au."""
        def residual(t: float) -> float:
            return self.distance(t, V0, gamma, v_sw) - self.r_l1_au

        return float(brentq(residual, 0.0, 1.0e4, xtol=1e-6, rtol=1e-8, maxiter=200))

    def impact_speed(self, V0: float, gamma: float | None = None, v_sw: float | None = None) -> float:
        """Скорость CME на L1 (км/с)."""
        return float(self.velocity(self.time_to_l1(V0, gamma, v_sw), V0, gamma, v_sw))

    def predict(self, cme_row: dict) -> dict:
        """Прогноз прихода CME на L1 по строке каталога CME."""
        v0 = float(cme_row["v_linear"])
        travel_time = self.time_to_l1(v0)
        v_impact = self.impact_speed(v0)

        start_time = pd.Timestamp(cme_row["start_time"])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")
        arrival = start_time + pd.Timedelta(hours=travel_time)

        return {
            "cme_id": str(cme_row.get("cme_id", "")),
            "start_time": start_time.isoformat(),
            "v0": v0,
            "v_impact": v_impact,
            "v_sw": self.v_sw,
            "travel_time_hours": travel_time,
            "arrival_time_l1": arrival.isoformat(),
        }

    def predict_ensemble(
        self,
        v0: float,
        v_sw: float,
        gamma_values=None,
        start_time=None,
    ) -> dict:
        """Ансамбль прогнозов по набору gamma при данном V_sw.

        Возвращает dict: {gamma: {"transit_hours": ..., "v_impact": ...,
        "arrival_time": ISO-строка или None}}.
        """
        gammas = DEFAULT_GAMMAS if gamma_values is None else tuple(float(g) for g in gamma_values)
        start = None
        if start_time is not None:
            start = pd.Timestamp(start_time)
            if start.tzinfo is None:
                start = start.tz_localize("UTC")

        ensemble = {}
        for gamma in gammas:
            transit = self.time_to_l1(v0, gamma=gamma, v_sw=v_sw)
            v_impact = self.impact_speed(v0, gamma=gamma, v_sw=v_sw)
            arrival = (start + pd.Timedelta(hours=transit)) if start is not None else None
            ensemble[gamma] = {
                "transit_hours": float(transit),
                "v_impact": float(v_impact),
                "arrival_time": arrival.isoformat() if arrival is not None else None,
            }
        return ensemble

    def estimate_v_sw(
        self,
        start_time,
        omni_df: pd.DataFrame,
        window_hours: float = VSW_WINDOW_H,
    ) -> float:
        """Median V в OMNI за window_hours до start_time. Fallback 400 км/с."""
        start = pd.Timestamp(start_time)
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)

        times = pd.to_datetime(omni_df["datetime"])
        if getattr(times.dt, "tz", None) is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)

        lo = start - pd.Timedelta(hours=window_hours)
        mask = (times >= lo) & (times <= start)
        values = omni_df.loc[mask, "V"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float(DEFAULT_VSW)
        return float(np.median(values))

    def predict_arrival_with_vsw(self, cme_row: dict, omni_df: pd.DataFrame) -> dict:
        """Прогноз прихода CME на L1 с V_sw из OMNI (median за 24 ч до старта)."""
        v0 = float(cme_row["v_linear"])
        start_time = pd.Timestamp(cme_row["start_time"])
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize("UTC")

        v_sw = self.estimate_v_sw(start_time, omni_df)
        travel_time = self.time_to_l1(v0, gamma=self.gamma, v_sw=v_sw)
        v_impact = self.impact_speed(v0, gamma=self.gamma, v_sw=v_sw)
        arrival = start_time + pd.Timedelta(hours=travel_time)

        return {
            "cme_id": str(cme_row.get("cme_id", "")),
            "start_time": start_time.isoformat(),
            "v0": v0,
            "v_sw": v_sw,
            "v_impact": v_impact,
            "travel_time_hours": travel_time,
            "arrival_time_l1": arrival.isoformat(),
            "ensemble": self.predict_ensemble(v0, v_sw, start_time=start_time),
        }

    def trajectory(self, V0: float, t_max: float, n: int = 500) -> pd.DataFrame:
        """Профиль расстояния и скорости для построения графика."""
        times = np.linspace(0.0, float(t_max), int(n))
        distances = [self.distance(float(t), V0) for t in times]
        return pd.DataFrame({"time_hours": times, "distance_au": distances})
