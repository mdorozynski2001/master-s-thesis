from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
import typer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from custom_trainer.dataset import YoloDataset


def calculate_linear_input(image_size: int, feature_extractor: nn.Sequential) -> int:
    # Tworzymy sztuczne wejście o rozmiarze (1, 3, image_size, image_size)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Przepuszczamy obraz przez feature extractor
    feature_map = feature_extractor(dummy_input)

    # Obliczamy liczbę neuronów wejściowych do pierwszej warstwy Linear
    flattened_size = feature_map.view(feature_map.size(0), -1).shape[1]

    return int(flattened_size)


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 3, image_size: int = 640):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(calculate_linear_input(image_size, self.feature_extractor), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.cls_head = nn.Linear(128, num_classes)  # Klasyfikacja
        self.box_head = nn.Linear(128, 4)  # Współrzędne bboxów

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.flatten(x)
        features = self.fc_layers(x)
        class_logits = self.cls_head(features)
        bbox_preds = self.box_head(features)
        return class_logits, bbox_preds


class CustomDetector:
    def __init__(self, num_classes: int = 3, image_size: int = 640, epochs: int = 10, batch_size: int = 32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(num_classes, image_size).to(self.device)
        self.epochs = epochs

        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001)

        # Transformacje obrazów
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            return torch.stack(images, dim=0), list(targets)

        self.train_dataset = YoloDataset(Path("train/images"), Path("train/labels"), transform=transform)
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.val_dataset = YoloDataset(Path("val/images"), Path("val/labels"), transform=transform)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self.test_dataset = YoloDataset(Path("test/images"), Path("test/labels"), transform=transform)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def loss_fn(self, class_preds, box_preds, class_targets, box_targets):
        cls_loss = nn.CrossEntropyLoss()(class_preds, class_targets)
        reg_loss = nn.SmoothL1Loss()(box_preds, box_targets)
        return cls_loss + reg_loss

    def train(self) -> None:
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for images, targets in tqdm(self.train_dataloader):
                images = images.to(self.device)

                # Rozdzielamy klasy i bboxy (teraz jako listy)

                class_targets = [
                    t[:, 0].long().to(self.device) if t.dim() > 1 else torch.zeros(1, dtype=torch.long).to(self.device)
                    for t in targets
                ]
                box_targets = [
                    t[:, 1].long().to(self.device) if t.dim() > 1 else torch.zeros(1, dtype=torch.long).to(self.device)
                    for t in targets
                ]
                # box_targets = [t[:, 1:].to(self.device) for t in targets]

                class_preds, box_preds = self.model(images)

                # Obliczamy straty dla każdego elementu w batchu
                losses = []
                losses = []
                for i in range(len(images)):  # Iterujemy po batchu obrazów
                    for j in range(len(class_targets[i])):  # Iterujemy po obiektach w obrazie
                        loss = self.loss_fn(
                            class_preds[i].unsqueeze(0),
                            box_preds[i].unsqueeze(0),
                            class_targets[i][j].unsqueeze(0),
                            box_targets[i][j].unsqueeze(0),
                        )
                        losses.append(loss)

                loss = torch.stack(losses).mean()  # Średnia strata dla batcha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.train_dataloader)}")
        torch.save(self.model.state_dict(), "model_weights.pt")

    def eval(self) -> None:
        self.model.eval()  # Ustawienie modelu w tryb ewaluacji
        image, _ = self.train_dataset[10]  # Pobieramy pojedynczy obraz
        image_resized = image.unsqueeze(0).to(self.device)  # Dodajemy wymiar batcha

        # Wykonanie predykcji
        with torch.no_grad():
            class_logits, bbox_preds = self.model(image_resized)

        # Predykcja klasy i współrzędnych bbox
        predicted_class = torch.argmax(class_logits, dim=1).item()
        bbox = bbox_preds[0].cpu().numpy()

        # Wyświetlanie predykcji
        print(f"Predykcja: Klasa {predicted_class}, Bbox: {bbox}")

        # Zamiana obrazu PIL na format OpenCV (BGR)
        image = image.permute(1, 2, 0).cpu().numpy()  # Zmieniamy z CxHxW na HxWxC
        image = (image * 255).astype(np.uint8)  # Skala [0,1] -> [0,255]
        image = cv2.resize(image, (1920, 1080))
        # Współrzędne bbox w formacie [x_center, y_center, width, height]
        # Zamiana na [x_min, y_min, x_max, y_max]
        height, width, _ = image.shape
        x_center, y_center, w, h = bbox

        # Przekształcamy z proporcji na współrzędne w pikselach
        x_min = int((x_center - w / 2) * width)
        y_min = int((y_center - h / 2) * height)
        x_max = int((x_center + w / 2) * width)
        y_max = int((y_center + h / 2) * height)

        # Rysowanie prostokąta na obrazie (w OpenCV)
        image_with_bbox = cv2.rectangle(image.copy(), (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        WINDOW_NAME = "aaa"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Wyświetlanie obrazu z bbox za pomocą OpenCV
        cv2.imshow(WINDOW_NAME, image_with_bbox)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def inner_main(
    image_size: int = typer.Option(640, help=""),
    epochs: int = typer.Option(10, help=""),
    batch_size: int = typer.Option(32, help=""),
) -> None:
    detector = CustomDetector(num_classes=1, image_size=image_size, epochs=epochs, batch_size=batch_size)
    detector.train()
    detector.eval()


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
