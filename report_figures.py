"""Воспроизводимые исследовательские рисунки пояснительной записки.

Каждая диаграмма строится в отдельной Figure. Составные иллюстрации
объединяются из растров без изменения исходных числовых данных.
Схемы и снимки интерфейса не являются результатом этого модуля.
"""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from PIL import Image
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

DPI = 180
BINS = 20


def _raster(figure: Figure) -> Image.Image:
    """Получить изображение фиксированного размера и закрыть Figure."""
    stream = BytesIO()
    figure.savefig(stream, format="png", dpi=DPI, facecolor=figure.get_facecolor())
    plt.close(figure)
    stream.seek(0)
    return Image.open(stream).convert("RGB")


def _join(images: list[Image.Image], columns: int) -> Image.Image:
    if not images or columns < 1:
        raise ValueError("Нужны изображения и положительное число столбцов.")
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * width, rows * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % columns) * width, (index // columns) * height))
    return canvas


def _axes(width: float, height: float, title: str) -> tuple[Figure, Axes]:
    figure = plt.figure(figsize=(width, height), dpi=DPI)
    axes = figure.add_subplot(111)
    axes.set_title(title, fontsize=12)
    axes.tick_params(labelsize=10)
    axes.grid(alpha=0.2)
    return figure, axes


def export_report_figures(
    *,
    data: pd.DataFrame,
    descriptive_clean: pd.DataFrame,
    training_properties: pd.DataFrame,
    normalized_properties: np.ndarray,
    training_ratio: pd.DataFrame,
    normalized_ratio: np.ndarray,
    test_features: pd.DataFrame,
    test_targets: pd.DataFrame,
    pipelines: Mapping[str, Pipeline],
    network: MLPRegressor,
    output_dir: str | Path = "figures",
) -> dict[str, str]:
    """Построить рисунки 1–4, 6–17 и 19 из результата текущего запуска."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    codes = {name: f"P{index}" for index, name in enumerate(data.columns, 1)}
    manifest: dict[str, dict[str, str]] = {}

    def save(number: int, image: Image.Image, description: str) -> None:
        path = destination / f"figure_{number:02d}.png"
        image.save(path)
        manifest[str(number)] = {
            "file": path.name,
            "description": description,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    histograms, boxes = [], []
    for name in data.columns:
        figure, axes = _axes(3.4, 2.36, codes[name])
        axes.hist(data[name], bins=BINS)
        axes.set_ylabel("Частота", fontsize=10)
        figure.tight_layout(pad=0.9)
        histograms.append(_raster(figure))
        figure, axes = _axes(3.4, 2.55, codes[name])
        axes.boxplot(data[name], tick_labels=[codes[name]])
        axes.set_ylabel("Значение", fontsize=10)
        figure.tight_layout(pad=0.9)
        boxes.append(_raster(figure))
    save(1, _join(histograms, 3), f"Гистограммы {data.shape[1]} исходных показателей; {len(data)} строки; {BINS} интервалов.")
    save(2, _join(boxes, 3), f"Квартильные диаграммы {data.shape[1]} исходных показателей; правило 1,5 IQR.")

    correlation = descriptive_clean.corr()
    figure, axes = _axes(8.6, 7.1, f"Корреляции Пирсона: {len(descriptive_clean)} наблюдений")
    image = axes.imshow(correlation.to_numpy(), vmin=-1, vmax=1)
    figure.colorbar(image, ax=axes, fraction=0.045, pad=0.03)
    axes.grid(False)
    axes.set_xticks(range(len(codes)), labels=codes.values(), rotation=45)
    axes.set_yticks(range(len(codes)), labels=codes.values())
    for row in range(len(codes)):
        for column in range(len(codes)):
            axes.text(column, row, f"{correlation.iloc[row, column]:.2f}",
                      ha="center", va="center", fontsize=7)
    figure.tight_layout(pad=1.1)
    save(3, _raster(figure), "Корреляционная матрица описательной выборки df_clean.")

    pairs = [
        (data.columns[4], data.columns[7]),
        (data.columns[1], data.columns[8]),
        (data.columns[9], data.columns[0]),
    ]
    scatterplots = []
    for feature, target in pairs:
        figure, axes = _axes(3.4, 3.0, f"{codes[feature]} и {codes[target]}")
        axes.scatter(descriptive_clean[feature], descriptive_clean[target], s=8, alpha=0.5)
        axes.set_xlabel(codes[feature])
        axes.set_ylabel(codes[target])
        figure.tight_layout(pad=1.0)
        scatterplots.append(_raster(figure))
    save(4, _join(scatterplots, 3), f"Пары P5/P8, P2/P9 и P10/P1; {len(descriptive_clean)} строк df_clean.")

    normalized = [
        pd.DataFrame(normalized_properties, columns=training_properties.columns,
                     index=training_properties.index),
        pd.DataFrame(normalized_ratio, columns=training_ratio.columns,
                     index=training_ratio.index),
    ]
    frames = [training_properties, training_ratio]
    for number, feature in enumerate(training_properties.columns, 6):
        panels = []
        for stage, stage_frames in [("До нормализации", frames), ("После нормализации", normalized)]:
            figure, axes = _axes(4.3, 2.40, f"{codes[feature]}: {stage.lower()}")
            values = np.concatenate([frame[feature].to_numpy() for frame in stage_frames
                                     if feature in frame.columns])
            edges = np.histogram_bin_edges(values, bins=BINS)
            for frame, label, linestyle in zip(stage_frames,
                    [f"Свойства, n={len(training_properties)}", f"Соотношение, n={len(training_ratio)}"], ["-", "--"]):
                if feature in frame.columns:
                    counts, _ = np.histogram(frame[feature], bins=edges)
                    axes.stairs(counts, edges, label=label, linestyle=linestyle, linewidth=1.4)
            axes.set_xlabel("Исходное значение" if stage == "До нормализации" else "Нормализованное значение", fontsize=10)
            axes.set_ylabel("Частота", fontsize=10)
            axes.legend(fontsize=8, frameon=False)
            figure.tight_layout(pad=0.8)
            panels.append(_raster(figure))
        save(number, _join(panels, 2), f"{codes[feature]}: {feature}; распределения до и после MinMaxScaler.")

    prediction_plots = []
    for target, key, label in [
        (data.columns[7], "Модуль, ГПа", "Модуль, ГПа"),
        (data.columns[8], "Прочность, МПа", "Прочность, МПа"),
    ]:
        observed = test_targets[target].to_numpy()
        predicted = pipelines[key].predict(test_features)
        lower = float(min(observed.min(), predicted.min()))
        upper = float(max(observed.max(), predicted.max()))
        figure, axes = _axes(4.3, 3.7, label)
        axes.scatter(observed, predicted, s=10, alpha=0.6)
        axes.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.1)
        axes.set_xlim(lower, upper)
        axes.set_ylim(lower, upper)
        axes.set_xlabel("Наблюдаемое значение")
        axes.set_ylabel("Прогноз Ridge")
        figure.tight_layout(pad=1.0)
        prediction_plots.append(_raster(figure))
    save(17, _join(prediction_plots, 2), f"Наблюдения и прогнозы выбранных Ridge; {len(test_features)} неизменённых тестовых строк.")

    figure, axes = _axes(8.4, 3.8, "Функция потерь выбранной нейронной сети")
    axes.plot(range(1, len(network.loss_curve_) + 1), network.loss_curve_)
    axes.set_xlabel("Итерация")
    axes.set_ylabel("Функция потерь обучения")
    figure.tight_layout(pad=1.1)
    save(19, _raster(figure), "Функция потерь обучения выбранного MLPRegressor; не тестовая MAE.")

    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {number: str(destination / item["file"]) for number, item in manifest.items()}
