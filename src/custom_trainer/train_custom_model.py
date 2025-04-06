from pathlib import Path

import typer

from custom_trainer.model import ObjectDetectorTrainer


def inner_main(
    dataset_path: Path = typer.Argument(..., help=""),
    image_size: int = typer.Option(600, help=""),
    epochs: int = typer.Option(200, help=""),
    batch_size: int = typer.Option(64, help=""),
    num_classes: int = typer.Option(3, help=""),
    num_boxes: int = typer.Option(30, help=""),
) -> None:
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
        epochs=epochs,
    )
    trainer.train_and_validate()


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
