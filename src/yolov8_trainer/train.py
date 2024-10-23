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
    batch_size: int = typer.Option(8, help="Requested batch size."),
    epochs: int = typer.Option(10, help="Expected number of epochs."),
    image_width: int = typer.Option(640, help="Image width."),
    workers_count: int = typer.Option(
        4, help="Number of workers for data loading and preparation."
    ),
):

    # Load the model.
    model = YOLO(model_path)

    settings = Munch(
        epochs=epochs,
        batch=batch_size,
        imgsz=image_width,
        workers=workers_count,
        device="cuda",
        name="train",
    )

    model.train(data=settings_path, **settings)


def main():
    typer.run(inner_main)


if __name__ == "__main__":
    main()
