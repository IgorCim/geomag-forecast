"""Torch Dataset для residual-корректора BMR (Этап 3.1)."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class DstResidualDataset(Dataset):
    """Датасет скользящих окон: X (12, 7) -> y (скаляр residual).

    Параметры
    ---------
    X : np.ndarray
        Признаки формы ``(N, 12, 7)`` (float).
    y : np.ndarray
        Целевой residual формы ``(N,)``.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[index], self.y[index]

    @classmethod
    def from_npz(cls, path: str) -> "DstResidualDataset":
        """Загружает датасет из ``.npz``-файла (X, y[, dates])."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл датасета не найден: {path}")
        with np.load(path) as data:
            return cls(data["X"], data["y"])


def load_npz_split(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Возвращает (X, y, dates) из npz-сплита."""
    with np.load(path) as data:
        return data["X"], data["y"], data["dates"]


def load_dst_observed(dates: np.ndarray, parquet_path: str) -> np.ndarray:
    """Сопоставляет целевым датам окон наблюдённый Dst из parquet-ряда."""
    import pandas as pd

    frame = pd.read_parquet(parquet_path, columns=["datetime", "Dst_observed"])
    frame = frame.set_index("datetime")["Dst_observed"]
    aligned = frame.reindex(pd.DatetimeIndex(dates))
    return aligned.to_numpy(dtype=np.float32)
