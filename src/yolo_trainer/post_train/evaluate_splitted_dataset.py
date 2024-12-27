import csv
import json
from pathlib import Path

import numpy as np
import typer
from munch import Munch
from ultralytics import YOLO

from yolo_trainer.post_train.utils import RESULT_ONLY_COMBINED, RESULT_SPLITTED_DATASETS


def inner_main(
    dataset_path: Path = typer.Argument(None, dir_okay=True, file_okay=True, help="Path to yolov8 settings file."),
    model_path: Path = typer.Option(
        "yolo11n.pt",
        dir_okay=False,
        file_okay=True,
        help="Path to pretrained YOLO model.",
    ),
    image_width: int = typer.Option(640, help="Image width."),
    split: str = typer.Option("test", help=""),
) -> None:
    datasets_result = {}

    dataset_yamls_paths = list(dataset_path.rglob(f"*{split}*.yaml"))
    dataset_yamls_paths.sort()

    for id, settings_path in enumerate(dataset_yamls_paths):
        dataset_name = settings_path.parent.name

        # Load the model.
        model = YOLO(model_path)
        settings = Munch(
            save=False,
            imgsz=image_width,
            name=f"{model_path.parent.parent.name}/evaluation_results/all/{dataset_name}",
            split=split,
        )

        result = model.val(data=settings_path, **settings)
        result_dict: dict[str, np.float64 | float] = result.results_dict
        for key in result_dict.keys():
            result_dict[key] = float(result_dict[key])
        datasets_result[dataset_name] = result_dict

        if id == 0:
            datasets_result["combined"] = result_dict.copy()  # Użycie kopii
        else:
            for key in result_dict.keys():
                datasets_result["combined"][key] = (datasets_result["combined"][key] * id + result_dict[key]) / (id + 1)

    print(datasets_result)

    with open(
        f"{model_path.parent.parent}/evaluation_results/all/{RESULT_SPLITTED_DATASETS}",
        mode="w",
    ) as file:
        json.dump(datasets_result, file, indent=4)
    with open(
        f"{model_path.parent.parent}/evaluation_results/all/{RESULT_ONLY_COMBINED}",
        mode="w",
    ) as file:
        json.dump(datasets_result["combined"], file, indent=4)

    with open(
        f"{model_path.parent.parent}/evaluation_results/all/result.csv",
        mode="w",
        newline="",
    ) as file:
        writer = csv.writer(file)
        headers = ["Category"] + list(next(iter(datasets_result.values())).keys())
        writer.writerow(headers)

        if "combined" in datasets_result:
            combined_row = ["combined"] + list(datasets_result["combined"].values())
            writer.writerow(combined_row)

        for category, metrics in datasets_result.items():
            if category == "combined":
                continue
            row = [category] + list(metrics.values())
            writer.writerow(row)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
