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
    times = []
    for image in tqdm(images):
        start_time = time.time()
        _ = model.predict(image, conf=0.25, iou=0.7, imgsz=image_size)
        end_time = time.time()
        times.append(end_time - start_time)
    print(f"Average time taken: {sum(times) / len(times)} seconds")


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
