"""Эмпирическая модель прогноза Dst (Burton-McPherron-Russell / O'Brien-McPherron).

Реализация модели O'Brien & McPherron (2000), являющейся модификацией
модели Burton, McPherron & Russell (1975).

Основное уравнение (Dst* — скорректированный Dst):

    dDst*/dt = Q(t) - Dst*/tau

Компоненты
----------
Dst* : Dst - b * sqrt(P_dyn) + c
Q    : a * (Ey - Ec) при Ey > Ec, иначе 0
Ey   : V * Bs / 1000  (мВ/м), V в км/с, Bs в нТл
Bs   : -Bz, если Bz < 0, иначе 0
tau  : 2.40 * exp(9.74 / (4.69 + Ey))  (часы)

Примечание по единицам: произведение V[км/с] * B[нТл] даёт 1e-3 мВ/м,
поэтому Ey = V * Bs / 1000 (мВ/м). При V=400 км/с и Bs=5 нТл Ey=2.0 мВ/м.

Численное интегрирование выполняется методом RK45 (``scipy.integrate.solve_ivp``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from .base import IndexModel


def _fill_nan(values: np.ndarray) -> np.ndarray:
    """Линейно интерполирует пропуски; остаток заполняет крайними значениями."""
    series = pd.Series(np.asarray(values, dtype=float))
    series = series.interpolate(limit_direction="both")
    return series.to_numpy(dtype=float)


class EmpiricalBMR(IndexModel):
    """Модель BMR/OBM для прогноза Dst-индекса.

    Параметры
    ---------
    a : float
        Коэффициент Q (нТл / (мВ/м · ч)), по умолчанию -4.4.
    Ec : float
        Пороговое значение Ey (мВ/м), по умолчанию 0.49.
    b : float
        Коэффициент коррекции Dst* (нТл / нПа**0.5), по умолчанию 7.26.
    c : float
        Константа коррекции Dst* (нТл), по умолчанию 11.0.
    """

    def __init__(self, a: float = -4.4, Ec: float = 0.49, b: float = 7.26, c: float = 11.0) -> None:
        self.a = a
        self.Ec = Ec
        self.b = b
        self.c = c

    def calculate_bs(self, Bz: np.ndarray) -> np.ndarray:
        """Вычисляет Bs = -Bz при Bz < 0, иначе 0 (векторизованно)."""
        Bz = np.asarray(Bz, dtype=float)
        return np.where(Bz < 0.0, -Bz, 0.0)

    def calculate_ey(self, V: np.ndarray, Bs: np.ndarray) -> np.ndarray:
        """Вычисляет электрическое поле Ey = V * Bs / 1000 (мВ/м)."""
        V = np.asarray(V, dtype=float)
        Bs = np.asarray(Bs, dtype=float)
        return V * Bs / 1000.0

    def calculate_Q(self, Ey: np.ndarray) -> np.ndarray:
        """Вычисляет инжекцию Q = a * (Ey - Ec) при Ey > Ec, иначе 0."""
        Ey = np.asarray(Ey, dtype=float)
        return np.where(Ey > self.Ec, self.a * (Ey - self.Ec), 0.0)

    def calculate_tau(self, Ey: np.ndarray) -> np.ndarray:
        """Вычисляет время релаксации tau = 2.40 * exp(9.74 / (4.69 + Ey)) часов."""
        Ey = np.asarray(Ey, dtype=float)
        return 2.40 * np.exp(9.74 / (4.69 + Ey))

    def correct_dst(self, Dst: np.ndarray, P_dyn: np.ndarray) -> np.ndarray:
        """Вычисляет Dst* = Dst - b * sqrt(P_dyn) + c."""
        Dst = np.asarray(Dst, dtype=float)
        P_dyn = np.asarray(P_dyn, dtype=float)
        return Dst - self.b * np.sqrt(P_dyn) + self.c

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Прогнозирует Dst по временному ряду параметров солнечного ветра.

        Параметры
        ---------
        df : pandas.DataFrame
            Колонки ``datetime`` (опционально), ``V``, ``Bz``, ``P_dyn``, ``Dst``.

        Возвращает
        ----------
        pandas.DataFrame
            Колонки: ``datetime``, ``Dst_observed``, ``Dst_predicted``,
            ``Dst_star``, ``Q``, ``tau``.
        """
        df = df.reset_index(drop=True)
        n = len(df)
        if n == 0:
            raise ValueError("Пустой DataFrame: нечего прогнозировать.")

        V = _fill_nan(df["V"].to_numpy(dtype=float))
        Bz = _fill_nan(df["Bz"].to_numpy(dtype=float))
        P_dyn = _fill_nan(df["P_dyn"].to_numpy(dtype=float))
        Dst = df["Dst"].to_numpy(dtype=float)

        Bs = self.calculate_bs(Bz)
        Ey = self.calculate_ey(V, Bs)
        Q = self.calculate_Q(Ey)
        tau = self.calculate_tau(Ey)

        dst_star_obs = self.correct_dst(Dst, P_dyn)

        t = self._time_hours(df)
        y0 = float(dst_star_obs[0]) if np.isfinite(dst_star_obs[0]) else 0.0

        if n == 1:
            star_pred = np.array([y0], dtype=float)
        else:
            star_pred = self._integrate(t, Q, tau, y0)

        dst_predicted = star_pred + self.b * np.sqrt(P_dyn) - self.c

        result = pd.DataFrame(
            {
                "datetime": df["datetime"] if "datetime" in df.columns else pd.RangeIndex(n),
                "Dst_observed": Dst,
                "Dst_predicted": dst_predicted,
                "Dst_star": dst_star_obs,
                "Q": Q,
                "tau": tau,
            }
        )
        return result

    @staticmethod
    def _time_hours(df: pd.DataFrame) -> np.ndarray:
        """Возвращает ось времени в часах от первой точки."""
        if "datetime" in df.columns:
            dt = pd.to_datetime(df["datetime"])
            t = (dt - dt.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 3600.0
            if np.all(np.isfinite(t)) and np.all(np.diff(t) > 0.0):
                return t
        return np.arange(len(df), dtype=float)

    @staticmethod
    def _integrate(t: np.ndarray, Q: np.ndarray, tau: np.ndarray, y0: float) -> np.ndarray:
        """Интегрирует dDst*/dt = Q - Dst*/tau методом RK45 на сетке t."""
        t = np.asarray(t, dtype=float)
        Q = np.asarray(Q, dtype=float)
        tau = np.asarray(tau, dtype=float)

        def rhs(tt: float, y: np.ndarray) -> np.ndarray:
            q = np.interp(tt, t, Q)
            ta = np.interp(tt, t, tau)
            ta = np.where(ta <= 0.0, 1e-6, ta)
            return q - y / ta

        solution = solve_ivp(
            rhs,
            (float(t[0]), float(t[-1])),
            [y0],
            t_eval=t,
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
        )
        if not solution.success or solution.y.shape[1] != len(t):
            raise RuntimeError(f"Интегрирование не удалось: {solution.message}")
        return solution.y[0]
