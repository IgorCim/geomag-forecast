"""PiRCNet — Physics-Informed Residual-Correction Network (Этап 3.2).

Уни-направленный (causal) LSTM-корректор систематической ошибки BMR.
Вход: окно 12 ч × 7 признаков. Выход: скалярная поправка к Dst_BMR.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PiRCNet(nn.Module):
    """Physics-Informed Residual-Correction Network.

    LSTM(7->64, 2 layers) -> FC(64->32) -> FC(32->1).
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last).squeeze(-1)


class WeightedMSELoss(nn.Module):
    """Взвешенная MSE: буря (Dst_observed < threshold) весит больше.

    Параметры
    ---------
    storm_weight : float
        Вес для часов бури, по умолчанию 5.0.
    threshold : float
        Порог Dst_observed (нТл), ниже которого час считается бурей, -50.
    """

    def __init__(self, storm_weight: float = 5.0, threshold: float = -50.0) -> None:
        super().__init__()
        self.storm_weight = storm_weight
        self.threshold = threshold

    def forward(self, pred: torch.Tensor, target: torch.Tensor, dst_observed: torch.Tensor) -> torch.Tensor:
        weight = torch.where(
            dst_observed < self.threshold,
            torch.full_like(dst_observed, self.storm_weight),
            torch.ones_like(dst_observed),
        )
        return (weight * (pred - target) ** 2).mean()
