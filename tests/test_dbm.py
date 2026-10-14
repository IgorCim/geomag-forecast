"""Тесты базовой архитектуры (IndexModel) и модели DBM (Stage 2)."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from download_cme_catalog import filter_earth_directed, is_earth_directed, is_halo
from models.base import IndexModel
from models.drag_based_model import DEFAULT_GAMMAS, DragBasedModel
from models.empirical_bmr import EmpiricalBMR
from validate_dbm import (
    HORIZONS,
    detect_storm_onset,
    first_onset_in_window,
    find_peak,
)


@pytest.fixture()
def dbm() -> DragBasedModel:
    return DragBasedModel()


def test_base_model() -> None:
    """IndexModel абстрактен, а EmpiricalBMR — его наследник."""
    with pytest.raises(TypeError):
        IndexModel()  # type: ignore[abstract]
    assert issubclass(EmpiricalBMR, IndexModel)


def test_bmr_still_works() -> None:
    """Модель Этапа 1 продолжает работать после рефакторинга."""
    n = 60
    V = np.array([400.0] * 20 + [600.0] * 10 + [400.0] * 30)
    Bz = np.array([0.0] * 20 + [-10.0] * 10 + [0.0] * 30)
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "V": V,
            "Bz": Bz,
            "P_dyn": np.full(n, 2.3),
            "Dst": np.zeros(n),
        }
    )
    result = EmpiricalBMR().predict(df)
    assert {"datetime", "Dst_observed", "Dst_predicted"}.issubset(result.columns)
    assert np.nanmin(result["Dst_predicted"].to_numpy()) < -20.0


def test_dbm_slow_cme(dbm: DragBasedModel) -> None:
    """Медленный CME (V0 < V_sw) ускоряется к скорости ветра."""
    v0 = 300.0
    early = float(dbm.velocity(0.0, v0))
    later = float(dbm.velocity(24.0, v0))
    assert early == pytest.approx(v0, rel=1e-9)
    assert later > v0
    assert later <= dbm.v_sw
    assert dbm.impact_speed(v0) > v0


def test_dbm_fast_cme(dbm: DragBasedModel) -> None:
    """Быстрый CME (V0 > V_sw) тормозит о ветер."""
    v0 = 1500.0
    later = float(dbm.velocity(24.0, v0))
    assert later < v0
    assert later >= dbm.v_sw
    assert dbm.impact_speed(v0) < v0


def test_dbm_no_change(dbm: DragBasedModel) -> None:
    """Если V0 == V_sw, скорость постоянна, а время = путь / V_sw."""
    assert dbm.velocity(0.0, 400.0) == pytest.approx(400.0, rel=1e-12)
    assert dbm.velocity(50.0, 400.0) == pytest.approx(400.0, rel=1e-12)
    assert dbm.impact_speed(400.0) == pytest.approx(400.0, rel=1e-12)
    expected_h = (dbm.r_l1_au - dbm.r_start_au) * 1.495978707e8 / 400.0 / 3600.0
    assert dbm.time_to_l1(400.0) == pytest.approx(expected_h, rel=1e-3)


def test_time_to_l1(dbm: DragBasedModel) -> None:
    """V0=1000 км/с даёт время прихода в диапазоне 30-80 часов."""
    t = dbm.time_to_l1(1000.0)
    assert 30.0 < t < 80.0
    assert dbm.distance(t, 1000.0) == pytest.approx(dbm.r_l1_au, abs=1e-3)


def test_impact_speed(dbm: DragBasedModel) -> None:
    """Скорость на L1 ниже стартовой для быстрого CME."""
    v_impact = dbm.impact_speed(1500.0)
    assert dbm.v_sw <= v_impact < 1500.0


def test_predict_returns_dict(dbm: DragBasedModel) -> None:
    """predict() возвращает словарь с ожидаемыми ключами."""
    row = {
        "cme_id": "2015-03-01T00:12:05-CME",
        "start_time": "2015-03-01T00:12:05Z",
        "v_linear": 1000.0,
    }
    out = dbm.predict(row)
    expected = {
        "cme_id",
        "start_time",
        "v0",
        "v_impact",
        "v_sw",
        "travel_time_hours",
        "arrival_time_l1",
    }
    assert expected.issubset(out.keys())
    assert out["cme_id"] == "2015-03-01T00:12:05-CME"
    assert out["v0"] == 1000.0
    assert out["travel_time_hours"] > 0.0
    arrival = pd.Timestamp(out["arrival_time_l1"])
    assert arrival > pd.Timestamp(row["start_time"])


def test_earth_directed_filter() -> None:
    """Фильтр выделяет halo-событие и отбрасывает боковое CME."""
    df = pd.DataFrame(
        {
            "cme_id": ["halo-1", "side-1"],
            "angular_width": [360.0, 90.0],
            "source_lon": [np.nan, 75.0],
        }
    )
    filtered = filter_earth_directed(df)
    assert list(filtered["cme_id"]) == ["halo-1"]


def test_halo_is_earth_directed() -> None:
    """Полный halo (width=360) является Earth-directed."""
    assert is_halo(360.0, "") is True
    assert is_earth_directed(360.0) is True


def test_side_cme_not_earth_directed() -> None:
    """Боковое CME (width=60, lon=90) не Earth-directed."""
    assert is_earth_directed(60.0, 90.0) is False
    assert is_earth_directed(100.0, -85.0) is False


def test_validation_only_earth_directed() -> None:
    """Валидация не должна включать боковые CME после фильтрации."""
    df = pd.DataFrame(
        {
            "cme_id": ["front-halo", "edge-front", "side-cme"],
            "angular_width": [360.0, 70.0, 60.0],
            "source_lon": [np.nan, 20.0, 90.0],
        }
    )
    earth = filter_earth_directed(df)
    ids = set(earth["cme_id"])
    assert "front-halo" in ids
    assert "edge-front" in ids
    assert "side-cme" not in ids
    assert bool(earth["is_earth_directed"].all())


def test_ensemble_returns_3(dbm: DragBasedModel) -> None:
    """predict_ensemble возвращает ровно 3 результата (по gamma)."""
    ensemble = dbm.predict_ensemble(v0=1000.0, v_sw=400.0)
    assert len(ensemble) == 3
    assert set(ensemble.keys()) == set(DEFAULT_GAMMAS)
    for payload in ensemble.values():
        assert payload["transit_hours"] > 0.0
        assert payload["v_impact"] > 0.0


def test_vsw_from_omni(dbm: DragBasedModel) -> None:
    """predict_arrival_with_vsw берёт median V из окна 24 ч до старта."""
    times = pd.date_range("2015-03-16 00:00", periods=48, freq="h")
    v = np.array([350.0] * 24 + [500.0] * 24)
    omni = pd.DataFrame({"datetime": times, "V": v})
    start = "2015-03-17 00:00"

    mask = (times >= pd.Timestamp(start) - pd.Timedelta(hours=24)) & (times <= pd.Timestamp(start))
    expected = float(np.median(v[mask]))
    out = dbm.predict_arrival_with_vsw(
        {"cme_id": "x", "start_time": start, "v_linear": 900.0}, omni
    )
    assert out["v_sw"] == pytest.approx(expected)
    assert out["v_sw"] != 400.0


def test_ensemble_ci_ordered(dbm: DragBasedModel) -> None:
    """Для быстрого CME приход упорядочен: γ min <= median <= γ max."""
    ensemble = dbm.predict_ensemble(v0=1500.0, v_sw=400.0)
    gammas = sorted(ensemble.keys())
    transits = [ensemble[g]["transit_hours"] for g in gammas]
    assert transits[0] <= transits[1] <= transits[2]


def test_v_obs_is_peak() -> None:
    """V_obs -- максимум V в окне поиска."""
    times = np.arange(np.datetime64("2015-03-18T00"), np.datetime64("2015-03-18T12"), np.timedelta64(1, "h"))
    v = np.array([400.0, 410.0, 420.0, 700.0, 650.0, 500.0, 480.0, 470.0, 460.0, 450.0, 440.0, 430.0])
    center = np.datetime64("2015-03-18T06")
    peak = find_peak(times, v, center, half_hours=6)
    assert peak is not None
    t_peak, v_peak = peak
    assert v_peak == pytest.approx(700.0)
    assert pd.Timestamp(t_peak) == pd.Timestamp("2015-03-18T03")


def test_detect_storm_onset() -> None:
    """Детектор находит первый час падения Dst > 20 нТл за 3 ч."""
    values = [-5.0] * 10 + [-10.0, -20.0, -35.0, -45.0, -45.0]
    times = pd.date_range("2015-03-17 00:00", periods=len(values), freq="h")
    series = pd.Series(values, index=times)

    onset = detect_storm_onset(series)
    assert onset is not None
    assert onset == pd.Timestamp("2015-03-17 12:00")


def test_onset_in_window() -> None:
    """first_onset_in_window возвращает первый onset в окне, иначе None."""
    onset_times = pd.to_datetime(
        ["2015-03-17T00:00", "2015-03-17T10:00", "2015-03-18T00:00"]
    ).to_numpy()

    center = np.datetime64("2015-03-17T12:00")
    first = first_onset_in_window(onset_times, center, half_hours=6.0)
    assert first == pd.Timestamp("2015-03-17 10:00")

    outside = first_onset_in_window(onset_times, np.datetime64("2015-03-20T00:00"), half_hours=6.0)
    assert outside is None


def test_metrics_no_24h() -> None:
    """Метрика ±24 ч удалена из горизонтов."""
    assert 24 not in HORIZONS
    assert set(HORIZONS) == {6, 12, 18}


def test_v_sw_still_used(dbm: DragBasedModel) -> None:
    """V_sw из OMNI по-прежнему используется вместо константы 400."""
    times = pd.date_range("2015-03-16 00:00", periods=30, freq="h")
    omni = pd.DataFrame({"datetime": times, "V": np.full(30, 520.0)})
    out = dbm.predict_arrival_with_vsw(
        {"cme_id": "y", "start_time": "2015-03-17 00:00", "v_linear": 800.0}, omni
    )
    assert out["v_sw"] == pytest.approx(520.0)
    assert out["v_sw"] != 400.0
