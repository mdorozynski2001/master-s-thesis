from pathlib import Path

import typer

from custom_trainer.model import ObjectDetectorTrainer


def inner_main(
    dataset_path: Path = typer.Argument(..., help="path to dataset"),
    image_size: int = typer.Option(600, help="image size"),
    epochs: int = typer.Option(200, help="epochs"),
    batch_size: int = typer.Option(64, help="batch size"),
    num_classes: int = typer.Option(3, help="number of classes"),
    num_boxes: int = typer.Option(30, help="number of boxes"),
    bacbkone_trainable_layers: int = typer.Option(5, help="number of backbone trainable layers"),
    lambda_box: float = typer.Option(5.0, help="lambda box"),
    lambda_cls: float = typer.Option(1.0, help="lambda cls"),
) -> None:
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
        epochs=epochs,
        trainable_layers=bacbkone_trainable_layers,
        lambda_box=lambda_box,
        lambda_cls=lambda_cls,
    )
    trainer.train_and_validate()


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
