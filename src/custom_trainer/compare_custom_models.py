import json
from pathlib import Path

import plotly.graph_objects as go
import typer

from custom_trainer.evaluate_custom_model import evaluate_custom_model

METRICS = ["precision", "recall", "f1"]


def inner_main(
    models_dir: Path = typer.Argument(
        Path("runs/detect/custom_model"), help="Path to directory with trained custom models"
    ),
    eval: bool = typer.Option(False, help="Flag to evaluate models"),
    output_dir: Path = typer.Option(None, help="Path to output directory"),
    dataset_path: Path = typer.Option(None, help="Path to dataset"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval:
        for model_path in models_dir.rglob("*.pt"):
            evaluate_custom_model(dataset_path, model_path.parent)

    result_paths = list(models_dir.rglob("test_results.json"))
    result_paths.sort()
    results = {}

    for result_file in result_paths:
        model_name = result_file.parent.name
        with open(result_file, "r") as file:
            results[model_name] = json.load(file)

    fig = go.Figure()

    for model_name, model_metrics in results.items():
        fig.add_trace(
            go.Bar(
                x=METRICS,
                y=[model_metrics[metric] for metric in METRICS],
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
