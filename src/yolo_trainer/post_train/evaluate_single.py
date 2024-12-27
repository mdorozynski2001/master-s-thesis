import json
from pathlib import Path

import numpy as np
import typer
from munch import Munch
from ultralytics import YOLO

from yolo_trainer.post_train.utils import RESULT_JSON


def inner_main(
    settings_path: Path = typer.Argument(None, dir_okay=False, file_okay=True, help="Path to yolov8 settings file."),
    model_path: Path = typer.Option(
        "yolo11n.pt",
        dir_okay=False,
        file_okay=True,
        help="Path to pretrained YOLO model.",
    ),
    image_width: int = typer.Option(640, help="Image width."),
    split: str = typer.Option("test", help=""),
) -> None:

    # Load the model.
    model = YOLO(model_path)
    settings = Munch(
        save=True,
        imgsz=image_width,
        name=f"{model_path.parent.parent.name}/evaluation_results",
        split=split,
    )

    result = model.val(data=settings_path, **settings)
    result_dict: dict[str, np.float64] = result.results_dict
    for key in result_dict.keys():
        result_dict[key] = float(result_dict[key])

    with open(f"{model_path.parent.parent}/evaluation_results/{RESULT_JSON}", mode="w") as file:
        json.dump(result_dict, file, indent=4)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
