import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm


class YoloDataset(Dataset):
    def __init__(self, image_folder: Path, label_folder: Path, transform: transforms.Compose = None) -> None:
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        label_path = os.path.join(self.label_folder, img_name.replace(".jpg", ".txt"))

        # Wczytanie obrazu
        image = Image.open(img_path).convert("RGB")

        # Wczytanie etykiet
        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append([class_id, x_center, y_center, width, height])

        # Konwersja do tensora
        if self.transform:
            image = self.transform(image)

        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels


# Transformacje obrazów
transform = transforms.Compose(
    [
        transforms.Resize((200, 200)),  # Dopasowanie do modelu
        transforms.ToTensor(),
    ]
)


def collate_fn(batch):
    images, targets = zip(*batch)  # Rozpakowanie batcha na listy obrazów i etykiet
    images = torch.stack(images, dim=0)  # Stackujemy obrazy, bo mają ten sam rozmiar
    return images, list(targets)  # Targets zwracamy jako listę, bo mają różne rozmiary


# Tworzenie zbioru danych
dataset = YoloDataset(Path("images"), Path("labels"), transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


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

    def forward(self, x):
        x = self.flatten(x)
        features = self.linear_relu_stack(x)
        class_logits = self.cls_head(features)
        bbox_preds = self.box_head(features)
        return class_logits, bbox_preds


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork().to(device)
print(model)


def loss_fn(class_preds, box_preds, class_targets, box_targets):
    cls_loss = nn.CrossEntropyLoss()(class_preds, class_targets)
    reg_loss = nn.SmoothL1Loss()(box_preds, box_targets)
    return cls_loss + reg_loss


optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(200):
    model.train()
    total_loss = 0

    for images, targets in tqdm(dataloader):
        images = images.to(device)

        # Rozdzielamy klasy i bboxy (teraz jako listy)
        class_targets = [t[:, 0].long().to(device) for t in targets]
        box_targets = [t[:, 1:].to(device) for t in targets]

        class_preds, box_preds = model(images)

        # Obliczamy straty dla każdego elementu w batchu
        losses = []
        losses = []
        for i in range(len(images)):  # Iterujemy po batchu obrazów
            for j in range(len(class_targets[i])):  # Iterujemy po obiektach w obrazie
                loss = loss_fn(
                    class_preds[i].unsqueeze(0),
                    box_preds[i].unsqueeze(0),
                    class_targets[i][j].unsqueeze(0),
                    box_targets[i][j].unsqueeze(0),
                )
                losses.append(loss)

        loss = torch.stack(losses).mean()  # Średnia strata dla batcha

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

dummy_input = torch.randn(1, 3, 200, 200).to(device)  # Dopasuj rozmiar do swojego modelu
traced_model = torch.jit.trace(model, dummy_input)

torch.save(model.state_dict(), "model_weights.pt")

model.eval()  # Ustawienie modelu w tryb ewaluacji
image, _ = dataset[10]  # Pobieramy pojedynczy obraz
image_resized = image.unsqueeze(0).to(device)  # Dodajemy wymiar batcha

# Wykonanie predykcji
with torch.no_grad():
    class_logits, bbox_preds = model(image_resized)

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
