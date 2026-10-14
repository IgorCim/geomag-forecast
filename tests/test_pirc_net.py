"""Тесты PiRCNet (Этап 3.2)."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ml_dataset import load_npz_split
from models.pirc_net import PiRCNet, WeightedMSELoss
from train_pirc_net import EarlyStopping

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def test_model_forward() -> None:
    model = PiRCNet()
    out = model(torch.randn(2, 12, 7))
    assert tuple(out.shape) == (2,)


def test_model_save_load() -> None:
    torch.manual_seed(0)
    model = PiRCNet()
    model.eval()
    x = torch.randn(4, 12, 7)
    with torch.no_grad():
        ref = model(x)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "m.pt")
        torch.save(model.state_dict(), path)
        clone = PiRCNet()
        clone.load_state_dict(torch.load(path, map_location="cpu"))
        clone.eval()
        with torch.no_grad():
            out = clone(x)
    assert torch.allclose(ref, out, atol=1e-6)


def test_weighted_loss() -> None:
    criterion = WeightedMSELoss(storm_weight=5.0, threshold=-50.0)
    pred = torch.zeros(2)
    target = torch.ones(2)
    quiet = criterion(pred, target, torch.tensor([0.0, 0.0]))
    storm = criterion(pred, target, torch.tensor([-60.0, -60.0]))
    assert float(quiet) == 1.0
    assert float(storm) == 5.0
    assert storm > quiet


def test_training_reduces_loss() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(256, 12, 7)
    y = 0.3 * X[:, -1, 0] + 2.0 * X[:, -1, 2]
    dst = torch.full((256,), -20.0)
    model = PiRCNet()
    criterion = WeightedMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    first = None
    last = None
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(X), y, dst)
        loss.backward()
        optimizer.step()
        if first is None:
            first = float(loss.detach())
        last = float(loss.detach())
    assert last < first


def test_predictions_shape() -> None:
    X, _, _ = load_npz_split(os.path.join(PROCESSED_DIR, "ml_test.npz"))
    n = 64
    model = PiRCNet()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[:n].astype(np.float32)))
    assert tuple(pred.shape) == (n,)
    assert torch.isfinite(pred).all()


def test_early_stopping() -> None:
    stopper = EarlyStopping(patience=3)
    assert stopper.step(10.0, 1) is True
    assert stopper.step(9.0, 2) is True
    assert stopper.should_stop is False
    stopper.step(9.5, 3)
    stopper.step(9.5, 4)
    stopper.step(9.5, 5)
    assert stopper.should_stop is True
    assert stopper.best_epoch == 2
