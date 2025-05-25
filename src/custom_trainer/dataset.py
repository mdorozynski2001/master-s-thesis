import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms  # type: ignore[import-untyped]


class YoloDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, image_folder: Path, label_folder: Path, transform: transforms.Compose = None) -> None:
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder)]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        label_path = os.path.join(self.label_folder, Path(img_name).stem + ".txt")

        # Wczytanie obrazu
        image = Image.open(img_path).convert("RGB")

        # Wczytanie etykiet
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                if class_id == 2:
                    class_id = 1
                x_center, y_center, width, height = map(float, parts[1:])
                x_min = max(0, x_center - width / 2)
                y_min = max(0, y_center - height / 2)
                x_max = min(1, x_center + width / 2)
                y_max = min(1, y_center + height / 2)

                labels.append([class_id, x_min, y_min, x_max, y_max])
        if len(labels) == 0:
            labels.append([2, 0, 0, 0, 0])

        # Konwersja do tensora
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels, dtype=torch.float32)
