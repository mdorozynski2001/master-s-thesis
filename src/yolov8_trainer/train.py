from pathlib import Path

import typer
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
    epochs: int = typer.Option(20, help="Expected number of epochs."),
    image_size: int = typer.Option(640, help="Image width."),
):

    # Load the model.
    model = YOLO(model_path)

    # Training.
    _ = model.train(
        data=settings_path,
        imgsz=image_size,
        epochs=epochs,
        batch=batch_size,
        device="cuda",
        name="yolov8n_custom",
    )


def main():
    typer.run(inner_main)


if __name__ == "__main__":
    main()
