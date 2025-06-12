import time
from pathlib import Path

import typer
from tqdm import tqdm
from ultralytics import YOLO


def inner_main(
    model_path: Path = typer.Argument(..., help="Path to the model file"),
    image_path: Path = typer.Argument(..., help="Path to the image file"),
    image_size: int = typer.Option(640, help="Image size"),
) -> None:
    model = YOLO(model_path)
    images = list(image_path.rglob("*.jpg")) + list(image_path.rglob("*.png"))
    times = {
        "preprocess": [],
        "inference": [],
        "postprocess": [],
    }
    for image in tqdm(images):
        result = model.predict(image, conf=0.25, iou=0.7, imgsz=image_size, verbose=False)

        times["preprocess"].append(result[0].speed["preprocess"])
        times["inference"].append(result[0].speed["inference"])
        times["postprocess"].append(result[0].speed["postprocess"])
    print(f"Average time taken: {sum(times['preprocess']) / len(times['preprocess']):.3f} miliseconds")
    print(f"Average time taken: {sum(times['inference']) / len(times['inference']):.3f} miliseconds")
    print(f"Average time taken: {sum(times['postprocess']) / len(times['postprocess']):.3f} miliseconds")
    print(f"Total average time taken: {(sum(times['preprocess']) + sum(times['inference']) + sum(times['postprocess'])) / len(images):.3f} miliseconds")


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
