from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from custom_trainer.dataset import YoloDataset


class NeuralNetwork(nn.Module):
    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(200 * 200 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(512, num_classes)  # Klasyfikacja
        self.box_head = nn.Linear(512, 4)  # Współrzędne bboxów

    def forward(self, x) -> tuple[nn.Linear, nn.Linear]:
        x = self.flatten(x)
        features = self.linear_relu_stack(x)
        class_logits = self.cls_head(features)
        bbox_preds = self.box_head(features)
        return class_logits, bbox_preds


class CustomDetector:
    def __init__(self, num_classes: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork(num_classes).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Transformacje obrazów
        transform = transforms.Compose(
            [
                transforms.Resize((200, 200)),  # Dopasowanie do modelu
                transforms.ToTensor(),
            ]
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            return torch.stack(images, dim=0), list(targets)

        self.dataset = YoloDataset(Path("images"), Path("labels"), transform=transform)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    def loss_fn(self, class_preds, box_preds, class_targets, box_targets):
        cls_loss = nn.CrossEntropyLoss()(class_preds, class_targets)
        reg_loss = nn.SmoothL1Loss()(box_preds, box_targets)
        return cls_loss + reg_loss

    def train(self) -> None:
        for epoch in range(10):
            self.model.train()
            total_loss = 0

            for images, targets in tqdm(self.dataloader):
                images = images.to(self.device)

                # Rozdzielamy klasy i bboxy (teraz jako listy)
                class_targets = [t[:, 0].long().to(self.device) for t in targets]
                box_targets = [t[:, 1:].to(self.device) for t in targets]

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

            print(f"Epoch {epoch+1}, Loss: {total_loss / len(self.dataloader)}")
        torch.save(self.model.state_dict(), "model_weights.pt")

    def eval(self) -> None:
        # dummy_input = torch.randn(1, 3, 200, 200).to(self.device)  # Dopasuj rozmiar do swojego modelu
        # traced_model = torch.jit.trace(self.model, dummy_input)

        self.model.eval()  # Ustawienie modelu w tryb ewaluacji
        image, _ = self.dataset[10]  # Pobieramy pojedynczy obraz
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


detector = CustomDetector(3)
detector.train()
detector.eval()
