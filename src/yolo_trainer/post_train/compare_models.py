import json
from pathlib import Path

import plotly.graph_objects as go
import typer

from yolo_trainer.post_train.utils import RESULT_JSON, RESULT_SPLITTED_DATASETS


def inner_main(
    models_dir: Path = typer.Argument(..., dir_okay=True, file_okay=False, help="Path to models to compare."),
    output_dir: Path = typer.Option(..., dir_okay=True, file_okay=True, help="Path to output dir to save charts"),
) -> None:
    result_paths = list(models_dir.rglob(RESULT_JSON))
    result_splitted_datasets_paths = list(models_dir.rglob(RESULT_SPLITTED_DATASETS))

    result_paths.sort()
    result_splitted_datasets_paths.sort()
    result_splitted_datasets = {}
    results = {}

    output_dir.mkdir(parents=True, exist_ok=True)

    for result_file in result_splitted_datasets_paths:
        model_name = result_file.parent.parent.parent.name
        with open(result_file, "r") as file:
            result_splitted_datasets[model_name] = json.load(file)

    for result_file in result_paths:
        model_name = result_file.parent.parent.name
        with open(result_file, "r") as file:
            results[model_name] = json.load(file)

    metrics = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)", "fitness"]
    figures = []

    for metric in metrics:
        fig = go.Figure()
        datasets_set = set()
        traces: dict[str, list[tuple[str, float]]] = {}

        for model_data in result_splitted_datasets.values():
            datasets_set.update(model_data.keys())
        datasets = sorted(datasets_set)

        for model_name, model_data in result_splitted_datasets.items():
            for dataset in datasets:
                value = model_data.get(dataset, {}).get(metric, None)
                if value is not None:
                    if model_name not in traces:
                        traces[model_name] = []
                    traces[model_name].append((dataset, value))

        for model_name, data_points in traces.items():
            fig.add_trace(
                go.Bar(
                    x=[f"{dataset}" for dataset, _ in data_points],
                    y=[value for _, value in data_points],
                    name=model_name,
                    marker=dict(opacity=0.7),
                )
            )

        fig.update_layout(
            title=f"Comparison of {metric}",
            xaxis=dict(title="Dataset Types", categoryorder="array", categoryarray=datasets),
            yaxis=dict(title="Metric Value"),
            barmode="group",
            legend_title="Models",
        )
        figures.append(fig)

    html_file = output_dir / "metrics_comparison_splitted_datasets.html"
    with open(html_file, "w") as f:
        for fig in figures:
            f.write(fig.to_html(full_html=False, include_plotlyjs="cdn"))

    print(f"Charts saved to: {html_file}")

    fig = go.Figure()

    for model_name, model_metrics in results.items():
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[model_metrics[metric] for metric in metrics],
                name=model_name,
            )
        )

    fig.update_layout(
        title="Comparison of Metrics Across Models",
        xaxis_title="Metrics",
        yaxis_title="Value",
        barmode="group",
        legend_title="Models",
    )

    output_html = output_dir / "metrics_comparison_full.html"
    fig.write_html(output_html)

    print(f"Chart saved to: {output_html}")


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
