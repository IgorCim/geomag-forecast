# geomag-forecast — прогноз Dst-индекса (Этап 1) и прихода CME на L1 (Этап 2)

Эмпирическая модель прогноза Dst-индекса по методу **O'Brien & McPherron (2000)** —
модификация модели **Burton-McPherron-Russell (1975)**.

## Физическая модель

```
dDst*/dt = Q(t) - Dst*/tau

Dst* = Dst - b * sqrt(P_dyn) + c
Q    = a * (Ey - Ec),  если Ey > Ec, иначе 0
Ey   = V * Bs / 1000            (мВ/м)
Bs   = -Bz,  если Bz < 0, иначе 0
tau  = 2.40 * exp(9.74 / (4.69 + Ey))   (часы)
```

Параметры по умолчанию: `a = -4.4`, `Ec = 0.49`, `b = 7.26`, `c = 11.0`.

Единицы: `V` — км/с, `Bz`/`Bs` — нТл (GSM), `P_dyn` — нПа, `Dst`/`Dst*` — нТл,
`Ey` — мВ/м. Поскольку `V[км/с] * B[нТл]` даёт `1e-3` мВ/м, множитель `1/1000`
обязателен (`V=400`, `Bs=5` → `Ey=2.0` мВ/м).

Интегрирование выполняется методом **RK45** (`scipy.integrate.solve_ivp`).
Начальное условие: `Dst*(t=0) = Dst(t=0) - b*sqrt(P_dyn(t=0)) + c`.

## Структура проекта

```
geomag-forecast/
├── data/
│   ├── raw/                    # исходные OMNI-файлы
│   └── processed/              # обработанные CSV
├── models/
│   ├── __init__.py
│   ├── base.py                 # IndexModel (ABC) - базовый интерфейс моделей
│   ├── empirical_bmr.py        # эмпирическая модель EmpiricalBMR (Dst)
│   └── drag_based_model.py     # физическая модель DragBasedModel (приход CME)
├── utils/
│   ├── __init__.py
│   ├── data_loader.py          # загрузка OMNI
│   └── metrics.py              # RMSE, MAE, R
├── scripts/
│   ├── download_omni.py        # скачивание OMNI2
│   ├── download_cme_catalog.py # скачивание каталога CME (DONKI/CDAW)
│   ├── find_storms.py          # каталог бурь
│   ├── validate_all_storms.py  # валидация BMR по бурям
│   ├── validate_model.py       # валидация на одной буре
│   ├── validate_dbm.py         # валидация DBM по OMNI2
│   ├── report_storms.py        # сводка и графики BMR
│   └── report_dbm.py           # графики и таблица DBM
├── tests/
│   ├── __init__.py
│   ├── test_empirical_bmr.py
│   └── test_dbm.py
├── configs/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Архитектура моделей

Все модели прогноза геомагнитных индексов наследуют абстрактный
`models.base.IndexModel` и реализуют:

```python
def predict(self, solar_wind_df: pd.DataFrame) -> pd.DataFrame:
    # -> datetime, <index>_observed, <index>_predicted
```

`EmpiricalBMR` (Этап 1) — эмпирическая модель Dst.

`DragBasedModel` (Этап 2) — физическая модель DBM для прогноза прихода
CME на L1. Скорость CME описывается аналитически
`V(t) = V_sw + (V0 - V_sw)/(1 + gamma*|V0 - V_sw|*t)` (t в часах); расстояние
интегрируется численно, время прихода на L1 ищется методом
`scipy.optimize.brentq`. Модель прогнозирует время прихода и скорость ICME
на L1 и потому не наследует `IndexModel` (это модель распространения,
а не временного ряда индекса).

```python
from models import DragBasedModel

dbm = DragBasedModel(gamma=0.5e-7, v_sw=400.0, r_start_au=0.1, r_l1_au=1.0)
out = dbm.predict({
    "cme_id": "2015-03-17T03:00:00-CME",
    "start_time": "2015-03-17T03:00:00Z",
    "v_linear": 1000.0,
})
print(out["arrival_time_l1"], out["v_impact"], out["travel_time_hours"])
```

## Этап 2: валидация DBM

```bash
python scripts/download_cme_catalog.py --start-year 2010 --end-year 2024
python scripts/validate_dbm.py
python scripts/report_dbm.py
```

`download_cme_catalog.py` пишет `data/raw/cme_catalog.csv`; `validate_dbm.py`
сопоставляет прогноз с приходами в OMNI2 (скачок `V > 100` км/с за 3 ч в окне
`±48` ч) и пишет `data/processed/dbm_metrics.csv`; `report_dbm.py` строит
графики (`dbm_trajectory_example.png`, `dbm_arrival_error_hist.png`,
`dbm_speed_error.png`) и таблицу `dbm_validation_table.txt`.

Если NASA DONKI недоступен (отдаёт HTML вместо JSON), скрипт автоматически
использует единый текстовый каталог CDAW/LASCO.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск тестов

```bash
python -m pytest tests/ -v
```

## Пример использования

```python
from utils.data_loader import load_omni_data
from models.empirical_bmr import EmpiricalBMR
from utils.metrics import evaluate_all

# Загрузка данных OMNI (часовой формат)
df = load_omni_data("data/raw/omni_min_20240101.txt")

# Модель и прогноз
model = EmpiricalBMR()  # параметры по умолчанию
result = model.predict(df)

# Оценка качества
metrics = evaluate_all(result["Dst_observed"], result["Dst_predicted"])
print(metrics)
```

`result` содержит колонки: `datetime`, `Dst_observed`, `Dst_predicted`,
`Dst_star`, `Q`, `tau`.

## Конфигурация

Файл `configs/config.yaml` содержит параметры модели и индексы колонок OMNI.
Значения из конфигурации передаются в `EmpiricalBMR(a=..., Ec=..., b=..., c=...)`.
Индексы колонок OMNI можно передать в `load_omni_data(filepath, columns={...})`.

## Примечания

- `load_omni_data` по умолчанию использует индексы колонок стандартного
  часового формата OMNI2. Для нестандартных файлов переопределите `columns`.
- Пропуски OMNI (`999.9`, `9999.9`, `99999.9`) заменяются на `NaN`;
  при прогнозе входные ряды интерполируются.
