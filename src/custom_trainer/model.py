import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.ops as ops  # type: ignore[import-untyped]
import typer
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from custom_trainer.dataset import YoloDataset


class DetectionLoss(nn.Module):
    def __init__(self, lambda_coord: float = 5.0, lambda_noobj: float = 0.5) -> None:
        super(DetectionLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_classes: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> torch.Tensor:
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
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
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
        )

        self.class_head = nn.Linear(256, num_boxes * num_classes)
        self.bbox_head = nn.Linear(256, num_boxes * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        x = self.fc_layers(x)
        bbox_predictions = torch.sigmoid(self.bbox_head(x)).view(-1, self.num_boxes, 4)  # 🛑 Sigmoid -> [0,1]
        class_logits = self.class_head(x).view(-1, self.num_boxes, self.class_head.out_features // self.num_boxes)
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
        self,
        dataset_path: Path,
        num_classes: int = 2,
        image_size: int = 350,
        epochs: int = 10,
        batch_size: int = 32,
        num_boxes: int = 10,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ObjectDetector(Backbone(), num_classes, num_boxes, image_size).to(self.device)
        self.epochs = epochs
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.image_size = image_size
        self.batch_size = batch_size

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            return torch.stack(images, dim=0), list(targets)

        self.train_dataset = YoloDataset(
            dataset_path / Path("train/images"), dataset_path / Path("train/labels"), transform=transform
        )
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )

        self.val_dataset = YoloDataset(
            dataset_path / Path("val/images"), dataset_path / Path("val/labels"), transform=transform
        )
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self.test_dataset = YoloDataset(
            dataset_path / Path("test/images"), dataset_path / Path("test/labels"), transform=transform
        )
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=1e-3, steps_per_epoch=len(self.train_dataloader), epochs=self.epochs
        )
        self.criterion = DetectionLoss().to(self.device)

        current_date = datetime.now()
        formatted_date = current_date.strftime("%d_%m_%Y")
        self.output_folder = Path(f"runs/detect/custom_model/train_{formatted_date}")

        self.output_folder.mkdir(parents=True, exist_ok=True)

    def save_model(self, final_save: bool = False) -> None:
        torch.save(self.model.state_dict(), self.output_folder / "best.pt")

        if final_save:
            model_data = {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_boxes": self.num_boxes,
            }
            with open(self.output_folder / "summary.json", "w") as file:
                json.dump(model_data, file, indent=4)

            torch.save(self.model.state_dict(), self.output_folder / "last.pt")

    def train_model(self) -> float:
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
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / len(self.train_dataloader)

    def validate_model(self, val: bool) -> float:
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

    def train_and_validate(self, val: bool = True, patience: int = 5) -> None:
        best_val_loss = float("inf")
        patience_counter = 0
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            train_loss = self.train_model()
            val_loss = self.validate_model(val)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
                print("🔹 New best model saved!")
            else:
                patience_counter += 1
                print(f"🔸 No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print("🛑 Early stopping triggered.")
                break

        print("Training complete!")
        self.save_model(final_save=True)


def run_inference(
    model: ObjectDetector,
    dataloader: DataLoader,
    device: torch.device,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.3,
) -> list[tuple[Any, Any, torch.Tensor, torch.Tensor]]:
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


def visualize_detections(
    detections: list[Any], class_names: list[str], num_images: int = 5, conf_threshold: float = 0.5
) -> None:
    for i in range(min(num_images, len(detections))):
        image, boxes, labels, scores = detections[i]
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * 255).astype("uint8")
        image = np.ascontiguousarray(image)

        h, w, _ = image.shape
        for box, label, score in zip(boxes, labels, scores):
            if score < conf_threshold:
                continue  # 🛑 Filtrujemy bboxy z niskim score

            x1, y1, x2, y2 = box.tolist()
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            label_name = class_names[label] if label < len(class_names) else f"Class {label}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label_name} {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.show()


def test_model_on_images(
    model: ObjectDetector, dataloader: DataLoader, class_names: list[str], device: torch.device, num_images: int = 5
) -> None:
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


def inner_main(
    dataset_path: Path = typer.Argument(..., help=""),
    image_size: int = typer.Option(500, help=""),
    epochs: int = typer.Option(200, help=""),
    batch_size: int = typer.Option(16, help=""),
    num_classes: int = typer.Option(2, help=""),
    num_boxes: int = typer.Option(30, help=""),
    train: bool = typer.Option(True, help=""),
    model_path: Path = typer.Option(None, help=""),
) -> None:
    class_names = ["Class 0", "Class 1", "Class 2"]
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
        epochs=epochs,
    )
    if train:
        trainer.train_and_validate()
    else:
        trainer.model.load_state_dict(torch.load(model_path))
        test_model_on_images(trainer.model, trainer.test_dataloader, class_names, trainer.device, num_images=100)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
