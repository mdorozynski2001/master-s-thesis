from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.ops as ops  # type: ignore[import-untyped]
import typer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from tqdm import tqdm

from custom_trainer.dataset import YoloDataset


class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super(DetectionLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred_bboxes, pred_classes, target_bboxes, target_classes) -> torch.Tensor:
        """
        pred_bboxes: (batch, num_boxes, 4)
        pred_classes: (batch, num_boxes, num_classes)
        target_bboxes: (batch, num_boxes, 4)
        target_classes: (batch, num_boxes)
        """

        # 🛑 Dopasowanie wymiarów target_bboxes do pred_bboxes
        batch_size = pred_bboxes.shape[0]
        if target_bboxes.shape[0] != batch_size:
            target_bboxes = target_bboxes[:batch_size]
            target_classes = target_classes[:batch_size]

        # 🛑 Zabezpieczenie przed pustymi tensorami
        if target_bboxes.numel() == 0 or target_classes.numel() == 0:
            return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        # Obliczamy straty
        bbox_loss = self.lambda_coord * self.bbox_loss(pred_bboxes, target_bboxes)
        class_loss = self.class_loss(pred_classes.permute(0, 2, 1), target_classes)

        return bbox_loss + class_loss


class Backbone(nn.Module):
    def __init__(self, trainable_layers: int = 3) -> None:
        super(Backbone, self).__init__()

        # Extract layers up to the final convolutional block
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

        # Optionally freeze some layers for transfer learning
        layers_to_freeze = len(list(self.feature_extractor.children())) - trainable_layers
        for i, layer in enumerate(self.feature_extractor.children()):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


def calculate_linear_input(image_size: int, feature_extractor: Backbone) -> int:
    dummy_input = torch.randn(1, 3, image_size, image_size)

    feature_map = feature_extractor(dummy_input)

    flattened_size = feature_map.view(feature_map.size(0), -1).shape[1]

    return int(flattened_size)


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_boxes: int = 10) -> None:
        super(DetectionHead, self).__init__()
        self.flatten = nn.Flatten()

        self.num_boxes = num_boxes  # Maksymalna liczba obiektów do wykrycia
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.class_head = nn.Linear(128, num_boxes * num_classes)
        self.bbox_head = nn.Linear(128, num_boxes * 4)
        self.bbox_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(x)

        x = self.fc_layers(x)

        # Przewidywanie skrzynek i klas
        bbox_predictions = self.bbox_activation(self.bbox_head(x).view(-1, self.num_boxes, 4))  # Wymusi zakres [0,1]
        class_logits = self.class_head(x)
        # print("Shape before reshape:", class_logits.shape)  # Debugowanie

        num_classes = self.class_head.out_features // self.num_boxes  # Obliczamy faktyczne num_classes
        class_logits = class_logits.view(-1, self.num_boxes, num_classes)

        return bbox_predictions, class_logits


class ObjectDetector(nn.Module):
    def __init__(self, backbone: Backbone, num_classes: int, num_boxes: int = 10, image_size: int = 300):
        super(ObjectDetector, self).__init__()
        self.backbone = backbone
        self.detection_head = DetectionHead(
            in_channels=calculate_linear_input(image_size, backbone), num_classes=num_classes, num_boxes=num_boxes
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        bbox_predictions, class_logits = self.detection_head(features)
        return bbox_predictions, class_logits


class ObjectDetectorTrainer:
    def __init__(
        self, num_classes: int = 3, image_size: int = 350, epochs: int = 10, batch_size: int = 32, num_boxes: int = 10
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ObjectDetector(Backbone(), num_classes, num_boxes, image_size).to(self.device)
        self.epochs = epochs

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = DetectionLoss().to(self.device)

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

    def train_model(self):
        self.model.train()
        total_loss = 0.0

        for images, targets in tqdm(self.train_dataloader, desc="Training"):
            images = images.to(self.device)

            # Sprawdzamy, czy batch zawiera targety
            if not isinstance(targets, list) or len(targets) == 0:
                continue

            target_bboxes_list = [t[:, 1:].to(self.device) for t in targets if t.shape[0] > 0]
            target_classes_list = [t[:, 0].long().to(self.device) for t in targets if t.shape[0] > 0]

            if len(target_bboxes_list) == 0:
                continue

            num_boxes = self.model.detection_head.num_boxes
            batch_size = images.shape[0]

            # Dopasowanie wymiarów
            target_bboxes = torch.stack(
                [
                    torch.cat([t, torch.zeros(max(0, num_boxes - t.shape[0]), 4, device=self.device)])
                    for t in target_bboxes_list
                ]
            )[:batch_size]
            target_classes = torch.stack(
                [
                    torch.cat(
                        [t, torch.full((max(0, num_boxes - t.shape[0]),), 0, dtype=torch.long, device=self.device)]
                    )
                    for t in target_classes_list
                ]
            )[:batch_size]

            self.optimizer.zero_grad()
            pred_bboxes, pred_classes = self.model(images)

            loss = self.criterion(pred_bboxes, pred_classes, target_bboxes, target_classes)
            loss = torch.Tensor(loss)

            if torch.isnan(loss):
                continue

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate_model(self, val):
        if val:
            dataloader = self.val_dataloader
        else:
            dataloader = self.train_dataloader
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for images, targets in tqdm(dataloader, desc="Validation"):
                images = images.to(self.device)

                if not isinstance(targets, list) or len(targets) == 0:
                    continue

                target_bboxes = [t[:, 1:].to(self.device) for t in targets if t.shape[0] > 0]
                target_classes = [t[:, 0].long().to(self.device) for t in targets if t.shape[0] > 0]

                if len(target_bboxes) == 0:
                    continue

                num_boxes = self.model.detection_head.num_boxes
                batch_size = images.shape[0]

                target_bboxes = torch.stack(
                    [
                        torch.cat([t, torch.zeros(max(0, num_boxes - t.shape[0]), 4, device=self.device)])
                        for t in target_bboxes
                    ]
                )[:batch_size]
                target_classes = torch.stack(
                    [
                        torch.cat(
                            [t, torch.full((max(0, num_boxes - t.shape[0]),), 0, dtype=torch.long, device=self.device)]
                        )
                        for t in target_classes
                    ]
                )[:batch_size]

                pred_bboxes, pred_classes = self.model(images)

                loss = self.criterion(pred_bboxes, pred_classes, target_bboxes, target_classes)

                if torch.isnan(loss):
                    continue

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def train_and_validate(self, val: bool = True, epochs=10, lr=0.001, patience=10):
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_model()
            val_loss = self.validate_model(val)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), "best_model.pt")
                print("🔹 New best model saved!")
            else:
                patience_counter += 1
                print(f"🔸 No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

        print("Training complete!")


def run_inference(
    model: ObjectDetector, dataloader: DataLoader, device: torch.device, conf_threshold=0.5, iou_threshold=0.3
):
    model.eval()
    detections = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Running Inference"):
            images = images.to(device)
            pred_bboxes, pred_classes = model(images)

            for i in range(len(images)):
                # Softmax dla klas i wybór najlepszej
                probs = torch.softmax(pred_classes[i], dim=-1)
                scores, labels = probs.max(dim=-1)

                # Filtrowanie bboxów na podstawie confidence threshold
                keep = scores > conf_threshold
                boxes = pred_bboxes[i][keep]
                scores = scores[keep]
                labels = labels[keep]

                # Non-Maximum Suppression (NMS)
                if len(boxes) > 0:
                    keep_idx = ops.nms(boxes, scores, iou_threshold)
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                    labels = labels[keep_idx]

                detections.append((images[i].cpu(), boxes.cpu(), labels.cpu(), scores.cpu()))

    return detections


def visualize_detections(detections, class_names, num_images=5):
    """
    detections: Lista [(image, boxes, labels, scores)]
    class_names: Lista nazw klas np. ["person", "car", "dog"]
    num_images: Ile obrazów wyświetlić
    """
    for i in range(min(num_images, len(detections))):
        image, boxes, labels, scores = detections[i]

        # 🛑 KONWERSJA: Zmieniamy tensory Torch na Numpy
        image = image.permute(1, 2, 0).cpu().numpy()  # Konwersja z CHW do HWC
        image = (image * 255).astype("uint8")  # 🛑 Skala 0-255
        image = np.ascontiguousarray(image)  # 🛑 Naprawa układu pamięci

        h, w, _ = image.shape
        print("\n\n\n")
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box.tolist()

            print(x1, y1, x2, y2)
            # 🛑 Przeskalowanie bboxów do wymiarów obrazu
            x1 = int((x1) * w)
            y1 = int((y1) * h)
            x2 = int((x2) * w)
            y2 = int((y2) * h)

            label_name = class_names[label] if label < len(class_names) else f"Class {label}"

            # 🛑 Rysowanie bboxa (OpenCV wymaga uint8 i prawidłowego układu pamięci)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label_name} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def test_model_on_images(model, dataloader, class_names, device, num_images=5):
    detections = run_inference(model, dataloader, device)
    visualize_detections(detections, class_names, num_images)


def postprocess_detections(
    bbox_predictions: torch.Tensor, class_logits: torch.Tensor, conf_threshold: float = 0.5, iou_threshold: float = 0.3
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Przetwarza wyniki detekcji, odfiltrowując niskie prawdopodobieństwa i nakładające się obiekty.

    bbox_predictions: (batch_size, num_boxes, 4)
    class_logits: (batch_size, num_boxes, num_classes)
    """
    batch_size, num_boxes, _ = bbox_predictions.shape
    final_boxes = []
    final_labels = []

    for i in range(batch_size):
        # Obliczamy prawdopodobieństwa klas
        probs = torch.softmax(class_logits[i], dim=-1)  # (num_boxes, num_classes)
        scores, labels = probs.max(dim=-1)  # Najlepsza klasa dla każdego boxa

        # Wybieramy tylko te boxy, które mają score > threshold
        keep = scores > conf_threshold
        boxes = bbox_predictions[i][keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) > 0:
            # Stosujemy Non-Maximum Suppression (NMS)
            keep_idx = ops.nms(boxes, scores, iou_threshold)
            boxes = boxes[keep_idx]
            labels = labels[keep_idx]

        final_boxes.append(boxes)
        final_labels.append(labels)

    return final_boxes, final_labels


# Nazwy klas (dopasuj do swoich)
class_names = ["Class 0", "Class 1", "Class 2"]
trainer = ObjectDetectorTrainer(image_size=500)
trainer.train_and_validate(epochs=20)

test_model_on_images(trainer.model, trainer.test_dataloader, class_names, trainer.device, num_images=5)
