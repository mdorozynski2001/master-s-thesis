from pathlib import Path

import torch
import typer

from custom_trainer.model import ObjectDetectorTrainer


def inner_main(
    dataset_path: Path = typer.Argument(..., help=""),
    image_size: int = typer.Option(600, help=""),
    num_classes: int = typer.Option(3, help=""),
    num_boxes: int = typer.Option(30, help=""),
    model_path: Path = typer.Option(None, help=""),
) -> None:
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
    )
    trainer.model.load_state_dict(torch.load(model_path))
    trainer.output_folder = model_path.parent
    _ = trainer.validate_model(val=False)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
