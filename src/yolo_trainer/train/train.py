import csv
import time
from pathlib import Path
from typing import Optional

import torch
import typer
from munch import Munch  # type: ignore[import-untyped]
from ultralytics import YOLO  # type: ignore[import-untyped]


class CustomCallback:
    def __init__(self):
        self.start_time = None
        self.file = open("runs/detect/train/training_time_GPU_memory.csv", "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["Epoch", "GPU Memory (MB)", "Epoch Time (s)"])

    def on_train_epoch_start(self, trainer):
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer):
        # Obliczanie zużycia GPU w GB
        gpu_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)  # Konwersja B -> GB
        torch.cuda.reset_peak_memory_stats()  # Resetowanie pomiaru dla kolejnej epoki

        # Obliczanie czasu trwania epoki w minutach i sekundach
        epoch_duration_sec = time.time() - self.start_time
        minutes, seconds = divmod(int(epoch_duration_sec), 60)
        epoch_duration_formatted = f"{minutes:02d}:{seconds:02d}"  # Format HH:MM

        # Zapis do CSV
        self.writer.writerow(
            [trainer.epoch, round(gpu_memory_gb, 2), epoch_duration_formatted]  # Zaokrąglenie do 2 miejsc po przecinku
        )
        self.file.flush()  # Natychmiastowy zapis do pliku


def inner_main(
    settings_path: Path = typer.Argument(None, dir_okay=False, file_okay=True, help="Path to yolo settings file."),
    model_path: Path = typer.Option(
        "yolo11n.pt",
        dir_okay=False,
        file_okay=True,
        help="Path to pretrained YOLO model.",
    ),
    batch_size: int = typer.Option(64, help="Requested batch size."),
    epochs: int = typer.Option(100, help="Expected number of epochs."),
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

    callback = CustomCallback()
    model.add_callback("on_train_epoch_start", callback.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", callback.on_train_epoch_end)

    model.train(data=settings_path, **settings)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
