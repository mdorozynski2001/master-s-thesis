from pathlib import Path

import typer
from munch import Munch
from ultralytics import YOLO


def inner_main(
    settings_path: Path = typer.Argument(
        None, dir_okay=False, file_okay=True, help="Path to yolov8 settings file."
    ),
    model_path: Path = typer.Option(
        "yolov8n.pt",
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

    _ = model.val(data=settings_path, **settings)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
