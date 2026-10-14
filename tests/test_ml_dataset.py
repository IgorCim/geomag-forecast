"""Тесты ML-датасета residual BMR (Этап 3.1)."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ml_dataset import DstResidualDataset

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES = ["V", "Bz", "P_dyn", "Dst_predicted", "Ey", "Q", "tau"]
SPLIT_RANGES = {"train": (2000, 2018), "val": (2019, 2021), "test": (2022, 2024)}


def _load(split: str) -> dict:
    path = os.path.join(PROCESSED_DIR, f"ml_{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def test_shapes() -> None:
    for split in SPLIT_RANGES:
        data = _load(split)
        X, y = data["X"], data["y"]
        assert X.ndim == 3
        assert X.shape[1:] == (12, 7)
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0] > 0
        assert X.dtype == np.float32
        assert y.dtype == np.float32


def test_no_leakage() -> None:
    years = {}
    for split, (lo, hi) in SPLIT_RANGES.items():
        data = _load(split)
        split_years = pd.DatetimeIndex(data["dates"]).year.to_numpy()
        assert split_years.min() >= lo and split_years.max() <= hi, split
        years[split] = set(split_years.tolist())
    assert not (years["train"] & years["val"])
    assert not (years["train"] & years["test"])
    assert not (years["val"] & years["test"])
    assert max(years["train"]) <= 2018 < min(years["val"]) <= 2021 < min(years["test"])


def test_norm_params_saved() -> None:
    path = os.path.join(PROCESSED_DIR, "norm_params.json")
    assert os.path.exists(path)
    with open(path, encoding="utf-8") as handle:
        norm = json.load(handle)
    assert norm["features"] == FEATURES
    assert len(norm["mean"]) == 7
    assert len(norm["std"]) == 7
    assert all(s > 0 for s in norm["std"])


def test_no_nan() -> None:
    for split in SPLIT_RANGES:
        data = _load(split)
        assert np.isfinite(data["X"]).all(), split
        assert np.isfinite(data["y"]).all(), split


def test_residual_magnitude() -> None:
    y = np.concatenate([_load(s)["y"] for s in SPLIT_RANGES])
    std = float(np.std(y))
    assert 0.0 < std < 200.0
    assert abs(float(np.mean(y))) < 50.0


def test_torch_dataset() -> None:
    ds = DstResidualDataset.from_npz(os.path.join(PROCESSED_DIR, "ml_train.npz"))
    assert len(ds) > 0
    x_item, y_item = ds[0]
    assert tuple(x_item.shape) == (12, 7)
    assert y_item.ndim == 0
    batch_x, batch_y = ds[0]
    assert batch_x.dtype.is_floating_point
