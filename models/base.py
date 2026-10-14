"""Базовые абстракции для всех моделей прогноза геомагнитных индексов."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class IndexModel(ABC):
    """Базовый класс для всех моделей прогноза геомагнитных индексов."""

    @abstractmethod
    def predict(self, solar_wind_df: pd.DataFrame) -> pd.DataFrame:
        """Возвращает DataFrame с колонками: datetime, <index>_observed, <index>_predicted."""
        raise NotImplementedError
