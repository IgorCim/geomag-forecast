"""Тесты эмпирической модели BMR/OBM и метрик."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.empirical_bmr import EmpiricalBMR
from utils.metrics import evaluate_all, mae, rmse


@pytest.fixture()
def model() -> EmpiricalBMR:
    return EmpiricalBMR()


def _make_df(V, Bz, P_dyn, Dst, start: str = "2024-01-01") -> pd.DataFrame:
    n = len(V)
    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=n, freq="h"),
            "V": np.asarray(V, dtype=float),
            "Bz": np.asarray(Bz, dtype=float),
            "P_dyn": np.asarray(P_dyn, dtype=float),
            "Dst": np.asarray(Dst, dtype=float),
        }
    )


def test_bs_calculation(model: EmpiricalBMR) -> None:
    """Bs = -Bz для Bz<0, иначе 0."""
    result = model.calculate_bs(np.array([-5.0, 0.0, 3.0]))
    np.testing.assert_allclose(result, [5.0, 0.0, 0.0])


def test_ey_calculation(model: EmpiricalBMR) -> None:
    """V=400 км/с, Bs=5 нТл -> Ey = 2.0 мВ/м."""
    ey = model.calculate_ey(np.array([400.0]), np.array([5.0]))
    np.testing.assert_allclose(ey, [2.0], rtol=1e-12)


def test_Q_threshold(model: EmpiricalBMR) -> None:
    """Ниже порога Ec инжекции нет, выше — Q отрицательно."""
    q_low = model.calculate_Q(np.array([0.3]))
    q_high = model.calculate_Q(np.array([1.0]))
    np.testing.assert_allclose(q_low, [0.0])
    assert q_high[0] < 0.0


def test_tau_max(model: EmpiricalBMR) -> None:
    """При Ey=0 время релаксации максимально (~19.2 ч)."""
    tau = model.calculate_tau(np.array([0.0]))
    assert abs(tau[0] - 19.2) < 0.2


def test_predict_runs(model: EmpiricalBMR) -> None:
    """predict() не падает и возвращает ожидаемые колонки на 100 строках."""
    rng = np.random.default_rng(0)
    n = 100
    V = 400.0 + 50.0 * rng.random(n)
    Bz = rng.normal(0.0, 5.0, n)
    P_dyn = 2.0 + 0.5 * rng.random(n)
    Dst = rng.normal(0.0, 10.0, n)

    result = model.predict(_make_df(V, Bz, P_dyn, Dst))

    assert len(result) == n
    for col in ("datetime", "Dst_observed", "Dst_predicted", "Dst_star", "Q", "tau"):
        assert col in result.columns
    assert np.all(np.isfinite(result["Dst_predicted"].to_numpy()))


def test_predict_negative_during_storm(model: EmpiricalBMR) -> None:
    """Сильная южная Bz (большой Ey) даёт отрицательный Dst."""
    p = 2.3
    V = [400.0] * 24 + [600.0] * 6 + [400.0] * 24
    Bz = [0.0] * 24 + [-10.0] * 6 + [0.0] * 24
    P_dyn = [p] * len(V)
    Dst = [0.0] * len(V)

    result = model.predict(_make_df(V, Bz, P_dyn, Dst))
    assert np.nanmin(result["Dst_predicted"].to_numpy()) < -20.0


def test_predict_recovery(model: EmpiricalBMR) -> None:
    """После бури Dst релаксирует обратно к нулю."""
    p = 2.3
    V = [400.0] * 24 + [600.0] * 6 + [400.0] * 200
    Bz = [0.0] * 24 + [-10.0] * 6 + [0.0] * 200
    P_dyn = [p] * len(V)
    Dst = [0.0] * len(V)

    result = model.predict(_make_df(V, Bz, P_dyn, Dst))
    predicted = result["Dst_predicted"].to_numpy()

    storm_min = np.nanmin(predicted)
    assert storm_min < -20.0
    assert abs(predicted[-1]) < 5.0
    assert abs(predicted[-1]) < abs(storm_min) / 5.0


def test_metrics() -> None:
    """RMSE для идентичных массивов равен нулю."""
    y = np.array([1.0, -2.0, 3.5, 0.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0

    y_pred = y + 1.0
    np.testing.assert_allclose(rmse(y, y_pred), 1.0)

    stats = evaluate_all(y, y)
    assert stats["rmse"] == 0.0
    assert stats["n"] == 4
