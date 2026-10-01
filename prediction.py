"""Загрузка сохранённых конвейеров и проверка входов; без HTTP и обучения."""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.pipeline import Pipeline

ANGLE = "Угол нашивки, град"
RATIO = "Соотношение матрица-наполнитель"
STRICTLY_POSITIVE = {RATIO, "Плотность, кг/м3", "модуль упругости, ГПа"}
NON_NEGATIVE = {
    "Количество отвердителя, м.%", "Содержание эпоксидных групп,%_2",
    "Поверхностная плотность, г/м2", "Потребление смолы, г/м2",
    "Шаг нашивки", "Плотность нашивки",
}
MODES = ("properties", "ratio")


class InputError(ValueError):
    """Ошибка пользовательского ввода, которую можно показать в интерфейсе."""


class ModelLoadError(RuntimeError):
    """Модели или пример не удалось подготовить к работе."""


@dataclass(frozen=True)
class Task:
    title: str
    features: tuple[str, ...]
    models: Mapping[str, Pipeline]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    example: Mapping[str, float]


@dataclass(frozen=True)
class Prediction:
    values: Mapping[str, float]
    predictions: Mapping[str, float]
    warnings: tuple[str, ...]


def parse_number(raw: str, feature: str) -> float:
    """Принимает точку, запятую и научную запись; запрещает NaN/Inf."""
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(f"Заполните поле «{feature}».")
    try:
        number = float(raw.strip().replace(",", "."))
    except (ValueError, OverflowError):
        raise InputError(f"Поле «{feature}» должно содержать число.") from None
    if not math.isfinite(number):
        raise InputError(f"Поле «{feature}» должно содержать конечное число.")
    return number


def validate_domain(feature: str, value: float) -> None:
    """Только явно заданные ограничения интерфейса, не полная физическая модель."""
    if feature == ANGLE and value not in (0.0, 90.0):
        raise InputError("Для угла нашивки поддерживаются только 0 и 90 градусов.")
    if feature in STRICTLY_POSITIVE and value <= 0:
        raise InputError(f"Поле «{feature}» должно быть больше нуля.")
    if feature in NON_NEGATIVE and value < 0:
        raise InputError(f"Поле «{feature}» не может быть отрицательным.")


class PredictionService:
    """Сервис использует порядок признаков и диапазоны из сохранённых моделей."""

    def __init__(self, tasks: Mapping[str, Task]) -> None:
        self.tasks = dict(tasks)

    @classmethod
    def load(cls, model_path: Path, example_path: Path) -> PredictionService:
        if not model_path.is_file():
            raise ModelLoadError(
                "Не найден models.joblib. Восстановите файл из комплекта "
                "или выполните composite.ipynb до последней ячейки."
            )
        if not example_path.is_file():
            raise ModelLoadError("Не найден examples.json. Восстановите его из комплекта.")
        try:
            # joblib загружает только локальный доверенный файл из комплекта.
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                bundle = joblib.load(model_path)
            examples = json.loads(example_path.read_text(encoding="utf-8"))
            if set(bundle) != set(MODES):
                raise ValueError("Неверный набор задач")
            tasks = {}
            for mode in MODES:
                item = bundle[mode]
                features = tuple(item["features"])
                models = item["models"]
                if len(features) != (11 if mode == "properties" else 10):
                    raise ValueError("Неверное число признаков")
                if len(set(features)) != len(features) or not models:
                    raise ValueError("Неверная схема признаков")
                first_scaler = next(iter(models.values())).named_steps["scaler"]
                for model in models.values():
                    if not isinstance(model, Pipeline):
                        raise ValueError("Ожидался Pipeline")
                    scaler = model.named_steps["scaler"]
                    if tuple(scaler.feature_names_in_) != features:
                        raise ValueError("Порядок входов не совпадает")
                    if not np.array_equal(scaler.data_min_, first_scaler.data_min_) or not np.array_equal(scaler.data_max_, first_scaler.data_max_):
                        raise ValueError("Диапазоны моделей одной задачи не совпадают")
                example = examples[mode]
                if set(example) != set(features):
                    raise ValueError("Неверная схема примера")
                tasks[mode] = Task(
                    title=item["title"], features=features, models=models,
                    minimum=tuple(map(float, first_scaler.data_min_)),
                    maximum=tuple(map(float, first_scaler.data_max_)),
                    example=example,
                )
            service = cls(tasks)
            for mode in MODES:
                service.predict(mode, {key: str(value) for key, value in tasks[mode].example.items()})
            return service
        except Exception as exc:
            raise ModelLoadError(
                "Не удалось загрузить совместимые модели или пример. "
                "Используйте файлы из одного комплекта и установите зависимости "
                "командой: python -m pip install -r requirements-app.txt"
            ) from exc

    def task(self, mode: str) -> Task:
        try:
            return self.tasks[mode]
        except KeyError:
            raise InputError("Неизвестный режим расчёта.") from None

    def predict(self, mode: str, raw_values: Mapping[str, str]) -> Prediction:
        task = self.task(mode)
        values, notes = {}, []
        for feature, lower, upper in zip(task.features, task.minimum, task.maximum):
            value = parse_number(raw_values.get(feature, ""), feature)
            validate_domain(feature, value)
            values[feature] = value
            if value < lower or value > upper:
                notes.append(
                    f"«{feature}»: {value:g} вне обучающего диапазона "
                    f"[{lower:.6g}; {upper:.6g}]. Значение не изменено."
                )
        row = pd.DataFrame([[values[key] for key in task.features]], columns=task.features)
        try:
            with np.errstate(over="raise", invalid="raise"):
                predictions = {name: float(model.predict(row)[0]) for name, model in task.models.items()}
        except (ValueError, FloatingPointError, OverflowError):
            raise InputError("Значения слишком велики для численно устойчивого расчёта.") from None
        if not all(math.isfinite(value) for value in predictions.values()):
            raise InputError("Не удалось получить конечный прогноз. Проверьте масштаб входов.")
        return Prediction(values, predictions, tuple(notes))
