import json
from pathlib import Path

import torch
import typer

from custom_trainer.model import ObjectDetectorTrainer


def evaluate_custom_model(
    dataset_path: Path = typer.Argument(..., help=""),
    model_path: Path = typer.Option(None, help=""),
) -> None:
    with open(model_path / "summary.json", "r") as f:
        data = json.load(f)

    trainer = ObjectDetectorTrainer(
        image_size=data["image_size"],
        num_classes=data["num_classes"],
        num_boxes=data["num_boxes"],
        dataset_path=dataset_path,
        lambda_box=data["lambda_box"],
        lambda_cls=data["lambda_cls"],
    )
    trainer.model.load_state_dict(torch.load(model_path / "best.pt"))
    trainer.output_folder = model_path
    _ = trainer.validate_model(val=False)


def main() -> None:
    typer.run(evaluate_custom_model)


if __name__ == "__main__":
    main()
