"""Метрики качества прогноза: RMSE, MAE, корреляция.

Все метрики игнорируют пары значений, содержащие NaN или inf.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _paired_valid(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Возвращает пары значений, конечные в обоих массивах."""
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Среднеквадратичная ошибка (Root Mean Squared Error)."""
    a, b = _paired_valid(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Средняя абсолютная ошибка (Mean Absolute Error)."""
    a, b = _paired_valid(y_true, y_pred)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a - b)))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Коэффициент корреляции Пирсона."""
    a, b = _paired_valid(y_true, y_pred)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Возвращает словарь со всеми метриками: RMSE, MAE, корреляция и число точек."""
    a, _ = _paired_valid(y_true, y_pred)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "correlation": correlation(y_true, y_pred),
        "n": int(a.size),
    }
