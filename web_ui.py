"""HTML-представление: не обучает модели и не выполняет прогнозирование."""
from __future__ import annotations

from html import escape
from pathlib import Path
from string import Template
from typing import Mapping

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from prediction import ANGLE, Prediction, PredictionService, Task

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "page.html"
TAB_LABELS = {
    "properties": "Модуль и прочность",
    "ratio": "Соотношение матрица-наполнитель",
}


def render_field(task: Task, index: int, value: str | float) -> str:
    name = task.features[index]
    field_id = f"field-{index}"
    attributes = (
        f'id="{field_id}" name="{escape(name, quote=True)}" '
        f'data-example="{task.example[name]}" '
        f'aria-describedby="hint-{index}" required'
    )
    if name == ANGLE:
        options = ['<option value="">Выберите угол</option>']
        for angle in (0, 90):
            selected = " selected" if str(value) in (str(angle), str(float(angle))) else ""
            options.append(f'<option value="{angle}"{selected}>{angle}°</option>')
        control = f'<select {attributes}>{"".join(options)}</select>'
        hint = "Поддерживаемые значения: 0° и 90°."
    else:
        control = (
            f'<input {attributes} value="{escape(str(value), quote=True)}" '
            'inputmode="decimal" autocomplete="off" placeholder="Введите число">'
        )
        hint = f"Диапазон обучения: {task.minimum[index]:.6g} — {task.maximum[index]:.6g}"
    return (
        f'<label for="{field_id}"><span>{escape(name)}</span>{control}'
        f'<small id="hint-{index}">{hint}</small></label>'
    )


def describe_model(pipeline: Pipeline) -> str:
    """Описание фактически загруженной модели без повторения её настроек в UI."""
    estimator = pipeline.named_steps["model"]
    if isinstance(estimator, Ridge):
        return f"Ridge, alpha = {estimator.alpha:g}"
    if isinstance(estimator, MLPRegressor):
        # Размеры обученных матриц, а не заранее заданная строка архитектуры.
        sizes = (estimator.coefs_[0].shape[0],
                 *(weights.shape[1] for weights in estimator.coefs_))
        return "MLP, " + " → ".join(map(str, sizes))
    return type(estimator).__name__


def describe_task_models(task: Task) -> str:
    descriptions = {target: describe_model(model)
                    for target, model in task.models.items()}
    unique = tuple(dict.fromkeys(descriptions.values()))
    if len(unique) == 1:
        return unique[0]
    return "; ".join(f"{target}: {description}"
                     for target, description in descriptions.items())


def render_result(result: Prediction | None, task: Task) -> str:
    if result is None:
        return ""
    items = "".join(
        f'<div class="result-row"><span>{escape(name)}</span>'
        f'<strong data-raw-value="{value!r}">{value:.4f}</strong></div>'
        for name, value in result.predictions.items()
    )
    warning = ""
    if result.warnings:
        notes = "".join(f"<li>{escape(note)}</li>" for note in result.warnings)
        warning = (
            '<aside class="warning"><b>Осторожно: выход за диапазон обучения</b>'
            f'<ul>{notes}</ul><p>Надёжность такого прогноза не подтверждена.</p></aside>'
        )
    model_name = escape(describe_task_models(task))
    return (
        f'<section id="prediction-result" aria-live="polite">{warning}'
        f'<div class="result"><h2>Результат расчёта</h2>{items}'
        f'<p class="result-note">Модели: {model_name}. '
        'Числа округлены только для отображения.</p></div></section>'
    )


def render_tabs(mode: str) -> str:
    tabs = []
    for key, label in TAB_LABELS.items():
        active = 'class="active" aria-current="page"' if key == mode else 'class=""'
        tabs.append(f'<a {active} href="/?mode={key}">{label}</a>')
    return "".join(tabs)


def render_page(
    service: PredictionService,
    mode: str = "properties",
    values: Mapping[str, str | float] | None = None,
    result: Prediction | None = None,
    error: str = "",
) -> str:
    task = service.task(mode)
    values = values or {}
    fields = "".join(
        render_field(task, index, values.get(name, ""))
        for index, name in enumerate(task.features)
    )
    error_html = f'<div class="error" role="alert">{escape(error)}</div>' if error else ""
    template = Template(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return template.substitute(
        tabs=render_tabs(mode), title=escape(task.title),
        feature_count=len(task.features), error=error_html,
        mode=escape(mode, quote=True), fields=fields,
        result=render_result(result, task),
    )
