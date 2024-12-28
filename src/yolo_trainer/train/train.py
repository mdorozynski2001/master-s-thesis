from pathlib import Path
from typing import Optional

import typer
from munch import Munch  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]


def inner_main(
    settings_path: Path = typer.Argument(None, dir_okay=False, file_okay=True, help="Path to yolo settings file."),
    model_path: Path = typer.Option(
        "yolo11n.pt",
        dir_okay=False,
        file_okay=True,
        help="Path to pretrained YOLO model.",
    ),
    batch_size: int = typer.Option(64, help="Requested batch size."),
    epochs: int = typer.Option(200, help="Expected number of epochs."),
    image_width: int = typer.Option(640, help="Image width."),
    workers_count: int = typer.Option(4, help="Number of workers for data loading and preparation."),
    cfg_path: Optional[Path] = typer.Option(
        None,
        help="Optional path to config file for training overrides.",
        file_okay=True,
        dir_okay=False,
    ),
    deterministic: bool = typer.Option(True, help="Sets an identical seed for `pytorch`, `numpy` and `python.random`."),
    seed: Optional[int] = typer.Option(42, help="Specify a specific seed for deterministic training."),
    resume_training: bool = typer.Option(
        False, help="Inidicates if training should be resumed from the last checkpoint."
    ),
) -> None:

    # Load the model.
    model = YOLO(model_path)

    settings = Munch(
        epochs=epochs,
        batch=batch_size,
        imgsz=image_width,
        workers=workers_count,
        deterministic=deterministic,
        seed=seed,
        cache=False,
        resume=resume_training,
        pretrained=True,
    )

    if cfg_path and cfg_path.is_file():
        settings.update(cfg=str(cfg_path))

    model.train(data=settings_path, **settings)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
