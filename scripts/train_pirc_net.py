"""Обучение PiRCNet — LSTM-корректора ошибки BMR (Этап 3.2 / 3.3).

Запуск:
    python scripts/train_pirc_net.py
    python scripts/train_pirc_net.py --lr 3e-4 --dropout 0.3 --patience 20 \
        --weight_decay 1e-4 --storm_weight 3.0

Этап 3.3: гиперпараметры вынесены в CLI-аргументы. Дефолты смягчены
(lr 1e-3 -> 3e-4, dropout 0.2 -> 0.3, patience 15 -> 20,
weight_decay 1e-5 -> 1e-4, storm_weight 5.0 -> 3.0), чтобы модель меньше
перестраховывалась на средних бурях и не портила корректные прогнозы BMR.

Воспроизводимость: seed=42 (torch + numpy). Split строго по времени из Этапа 3.1.
Val используется для scheduler/early stopping; test не загружается.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.ml_dataset import load_dst_observed, load_npz_split
from models.pirc_net import PiRCNet, WeightedMSELoss

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PARQUET = os.path.join(PROCESSED_DIR, "omni_with_bmr.parquet")
BEST_PATH = os.path.join(MODELS_DIR, "pirc_net_best.pt")
HISTORY_PATH = os.path.join(PROCESSED_DIR, "training_history.json")

SEED = 42
BATCH_SIZE = 256
LR = 3e-4
MAX_EPOCHS = int(os.environ.get("PIRC_MAX_EPOCHS", "100"))
PATIENCE = 20
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
STORM_WEIGHT = 3.0
SCHED_PATIENCE = 5
SCHED_FACTOR = 0.5

SMOKE_BATCH_SIZE = 1024
SMOKE_EPOCHS = 2
SMOKE_TRAIN = 5000
SMOKE_VAL = 1000
SMOKE_LOG_EVERY = 50


class EarlyStopping:
    """Ранняя остановка по val_loss с сохранением лучшей эпохи."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float, epoch: int) -> bool:
        """Возвращает True, если val_loss улучшился."""
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_loader(
    split: str,
    shuffle: bool,
    generator: torch.Generator | None = None,
    batch_size: int = BATCH_SIZE,
    limit: int | None = None,
) -> DataLoader:
    X, y, dates = load_npz_split(os.path.join(PROCESSED_DIR, f"ml_{split}.npz"))
    if limit is not None:
        X, y, dates = X[:limit], y[:limit], dates[:limit]
    dst = load_dst_observed(dates, PARQUET)
    if not np.isfinite(dst).all():
        raise RuntimeError(f"NaN в Dst_observed для split={split}")
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
        torch.from_numpy(dst.astype(np.float32)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def run_epoch(model, loader, criterion, optimizer, device, log_every: int = 0) -> float:
    model.train()
    total, n = 0.0, 0
    for step, (xb, yb, dstb) in enumerate(loader, start=1):
        xb, yb, dstb = xb.to(device), yb.to(device), dstb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb, dstb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total += loss.item() * xb.size(0)
        n += xb.size(0)
        if log_every and step % log_every == 0:
            print(f"[train]   batch {step:5d}: loss={loss.item():.4f}")
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb, dstb in loader:
        xb, yb, dstb = xb.to(device), yb.to(device), dstb.to(device)
        loss = criterion(model(xb), yb, dstb)
        total += loss.item() * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def main(
    smoke_test: bool = False,
    lr: float = LR,
    dropout: float = DROPOUT,
    patience: int = PATIENCE,
    weight_decay: float = WEIGHT_DECAY,
    storm_weight: float = STORM_WEIGHT,
) -> int:
    set_seed(SEED)
    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("[train] ВНИМАНИЕ: CUDA недоступна, обучение на CPU.")

    if smoke_test:
        batch_size = SMOKE_BATCH_SIZE
        max_epochs = SMOKE_EPOCHS
        train_limit, val_limit = SMOKE_TRAIN, SMOKE_VAL
        log_every = SMOKE_LOG_EVERY
        print(f"[train] SMOKE TEST: train={train_limit}, val={val_limit}, "
              f"batch={batch_size}, epochs={max_epochs}")
    else:
        batch_size = BATCH_SIZE
        max_epochs = MAX_EPOCHS
        train_limit = val_limit = None
        log_every = 0

    generator = torch.Generator().manual_seed(SEED)
    train_loader = build_loader("train", shuffle=True, generator=generator,
                                batch_size=batch_size, limit=train_limit)
    val_loader = build_loader("val", shuffle=False, batch_size=batch_size, limit=val_limit)
    print(f"[train] Размеры: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}")

    model = PiRCNet(dropout=dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Device: {device}; параметров: {n_params}")
    print(f"[train] Гиперпараметры: lr={lr}, dropout={dropout}, patience={patience}, "
          f"weight_decay={weight_decay}, storm_weight={storm_weight}")

    criterion = WeightedMSELoss(storm_weight=storm_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=SCHED_PATIENCE, factor=SCHED_FACTOR
    )
    stopper = EarlyStopping(patience=patience)

    history = []
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, log_every=log_every)
        val_loss = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr_now})

        improved = stopper.step(val_loss, epoch)
        scheduler.step(val_loss)
        if improved and not smoke_test:
            torch.save(model.state_dict(), BEST_PATH)

        print(f"[train] epoch {epoch:3d}: train={train_loss:.4f} val={val_loss:.4f} "
              f"lr={lr_now:.2e}{' *' if improved else ''}")

        if stopper.should_stop:
            print(f"[train] Early stopping на эпохе {epoch} (best epoch={stopper.best_epoch})")
            break

    elapsed = time.time() - started
    if smoke_test:
        decreasing = history[-1]["train_loss"] < history[0]["train_loss"]
        print(f"[train] Smoke: время={elapsed:.1f} с; train_loss {history[0]['train_loss']:.4f} -> "
              f"{history[-1]['train_loss']:.4f}; уменьшился={'да' if decreasing else 'нет'}")
        print("[train] Smoke: модель/история НЕ сохраняются (checkpoint не перезаписывается).")
        return 0

    meta = {
        "n_params": n_params,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "epochs_run": len(history),
        "max_epochs": max_epochs,
        "best_epoch": stopper.best_epoch,
        "best_val_loss": stopper.best,
        "stopped_early": stopper.should_stop,
        "hyperparams": {
            "lr": lr,
            "dropout": dropout,
            "patience": patience,
            "weight_decay": weight_decay,
            "storm_weight": storm_weight,
        },
        "seed": SEED,
        "history": history,
    }
    with open(HISTORY_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
    print(f"[train] Время: {elapsed / 60.0:.1f} мин; best epoch={stopper.best_epoch}, "
          f"best val={stopper.best:.4f}; сохранено: {BEST_PATH}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение PiRCNet")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Быстрый прогон: subsample + 2 эпохи, без сохранения модели")
    parser.add_argument("--lr", type=float, default=LR,
                        help="Скорость обучения Adam (default: 3e-4)")
    parser.add_argument("--dropout", type=float, default=DROPOUT,
                        help="Dropout в LSTM и FC (default: 0.3)")
    parser.add_argument("--patience", type=int, default=PATIENCE,
                        help="Patience early stopping (default: 20)")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                        help="L2-регуляризация Adam (default: 1e-4)")
    parser.add_argument("--storm_weight", type=float, default=STORM_WEIGHT,
                        help="Вес часов бури в WeightedMSELoss (default: 3.0)")
    args = parser.parse_args()
    sys.exit(main(
        smoke_test=args.smoke_test,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        weight_decay=args.weight_decay,
        storm_weight=args.storm_weight,
    ))
