import csv
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision.ops as ops  # type: ignore[import-untyped]
import typer
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from custom_trainer.dataset import YoloDataset


class DetectionLoss(nn.Module):
    def __init__(
        self,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_conf: float = 1.0,
        lambda_iou: float = 3.0,
        lambda_metrics: float = 5.0,
    ) -> None:
        super(DetectionLoss, self).__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss(ignore_index=2)
        self.conf_loss = nn.BCELoss()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_conf = lambda_conf
        self.lambda_iou = lambda_iou
        self.lambda_metrics = lambda_metrics

    def calculate_iou(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Calculate intersection coordinates
        x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

        # Calculate intersection area
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # Calculate union area
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union = pred_area + target_area - intersection

        # Calculate IoU
        iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
        return iou

    def calculate_precision_recall_loss(
        self, pred_classes: torch.Tensor, target_classes: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Convert predictions to class indices
        pred_indices = torch.argmax(pred_classes, dim=-1)

        # Ensure both tensors have the same shape
        if pred_indices.shape != target_classes.shape:
            # Reshape to match if possible
            if pred_indices.numel() == target_classes.numel():
                pred_indices = pred_indices.view(target_classes.shape)
            else:
                # If shapes are incompatible, use a default loss
                return torch.tensor(0.0, device=pred_classes.device), torch.tensor(0.0, device=pred_classes.device)

        # Flatten tensors for sklearn metrics
        pred_flat = pred_indices.view(-1).cpu().detach().numpy()
        target_flat = target_classes.view(-1).cpu().detach().numpy()

        # Calculate precision and recall using sklearn
        # We need to handle the case where there are no predictions or targets
        if len(pred_flat) == 0 or len(target_flat) == 0:
            return torch.tensor(0.0, device=pred_classes.device), torch.tensor(0.0, device=pred_classes.device)

        # Calculate precision and recall for each class
        precision = precision_score(target_flat, pred_flat, average="macro", zero_division=0)
        recall = recall_score(target_flat, pred_flat, average="macro", zero_division=0)

        # Convert back to tensors
        precision_tensor = torch.tensor(1.0 - precision, device=pred_classes.device)
        recall_tensor = torch.tensor(1.0 - recall, device=pred_classes.device)

        return precision_tensor, recall_tensor

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_classes: torch.Tensor,
        confidences: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_classes: torch.Tensor,
    ) -> torch.Tensor:
        if target_bboxes.numel() == 0 or target_classes.numel() == 0:
            return torch.tensor(0.0, device=pred_bboxes.device, requires_grad=True)

        # Bounding box loss for objects
        bbox_mask = (target_classes != 2).unsqueeze(-1)
        bbox_loss: torch.Tensor = self.lambda_coord * self.bbox_loss(
            pred_bboxes[bbox_mask.expand_as(pred_bboxes)], target_bboxes[bbox_mask.expand_as(target_bboxes)]
        )

        # IoU loss for objects
        iou = self.calculate_iou(pred_bboxes, target_bboxes)
        iou_loss = self.lambda_iou * (1 - iou[bbox_mask.squeeze(-1)]).mean()

        # Classification loss
        pred_classes = pred_classes.permute(0, 2, 1)
        class_loss: torch.Tensor = self.class_loss(pred_classes, target_classes)

        # Confidence loss
        # For objects (target_classes != 2), confidence should be high
        # For background (target_classes == 2), confidence should be low
        obj_mask = (target_classes != 2).float()
        conf_loss = self.lambda_conf * self.conf_loss(confidences.squeeze(-1), obj_mask)

        # Background loss (no object loss)
        pred_probs = torch.softmax(pred_classes.permute(0, 2, 1), dim=-1)
        noobj_mask = target_classes == 2
        bg_probs = pred_probs[..., 0]
        noobj_loss = self.lambda_noobj * torch.mean(noobj_mask.float() * bg_probs)

        # Precision and recall loss
        precision_loss, recall_loss = self.calculate_precision_recall_loss(pred_classes, target_classes)
        metrics_loss = self.lambda_metrics * (precision_loss + recall_loss)

        return bbox_loss + class_loss + noobj_loss + conf_loss + iou_loss + metrics_loss


class Backbone(nn.Module):
    def __init__(self, trainable_layers: int = 5) -> None:
        super(Backbone, self).__init__()

        self.feature_extractor = nn.Sequential(
            # First block - input 600x600x1 -> 300x300x32
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            # Second block - 300x300x32 -> 150x150x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            # Third block - 150x150x64 -> 75x75x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            # Fourth block - 75x75x128 -> 38x38x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            # Fifth block - 38x38x256 -> 19x19x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
        )

        layers_to_freeze = len(list(self.feature_extractor.children())) - trainable_layers
        for i, layer in enumerate(self.feature_extractor.children()):
            if i < layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = self.feature_extractor(x)
        return features


def calculate_linear_input(image_size: int, feature_extractor: Backbone) -> int:
    dummy_input = torch.randn(1, 1, image_size, image_size)

    feature_map = feature_extractor(dummy_input)

    flattened_size = feature_map.view(feature_map.size(0), -1).shape[1]

    return int(flattened_size)


class DetectionHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, num_boxes: int = 10) -> None:
        super(DetectionHead, self).__init__()
        self.flatten = nn.Flatten()
        self.num_boxes = num_boxes

        # Enhanced feature extraction with residual connections
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            # Residual block 1
            ResidualBlock(1024),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            # Residual block 2
            ResidualBlock(512),
        )

        # Separate heads for classification and bounding box prediction
        self.class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_boxes * num_classes),
            nn.Sigmoid(),
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Linear(256, num_boxes * 4), nn.Sigmoid()
        )

        # Confidence head for objectness score
        self.conf_head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Linear(256, num_boxes), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        features = self.fc_layers(x)

        # Get predictions from each head
        bbox_predictions = self.bbox_head(features).view(-1, self.num_boxes, 4)
        num_classes = self.class_head[3].out_features // self.num_boxes
        class_logits = self.class_head(features).view(-1, self.num_boxes, num_classes)
        confidences = self.conf_head(features).view(-1, self.num_boxes, 1)

        # Apply confidence to class logits
        class_logits = class_logits * confidences

        return bbox_predictions, class_logits, confidences


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.1),
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += identity  # Residual connection
        out = self.relu(out)
        return out


class ObjectDetector(nn.Module):
    def __init__(self, backbone: Backbone, num_classes: int, num_boxes: int = 10, image_size: int = 300):
        super(ObjectDetector, self).__init__()
        self.backbone = backbone
        self.detection_head = DetectionHead(
            in_channels=calculate_linear_input(image_size, backbone), num_classes=num_classes, num_boxes=num_boxes
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        bbox_predictions, class_logits, confidences = self.detection_head(features)
        return bbox_predictions, class_logits, confidences


class ObjectDetectorTrainer:
    def __init__(
        self,
        dataset_path: Path,
        num_classes: int = 3,
        image_size: int = 600,
        epochs: int = 200,
        batch_size: int = 64,
        num_boxes: int = 30,
        trainable_layers: int = 5,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_conf: float = 1.0,
        lambda_iou: float = 3.0,
        lambda_metrics: float = 5.0,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ObjectDetector(Backbone(trainable_layers), num_classes, num_boxes, image_size).to(self.device)
        self.epochs = epochs
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.image_size = image_size
        self.batch_size = batch_size

        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                    ],
                    p=0.3,
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.ToTensor(),
            ]
        )

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ]
        )

        def collate_fn(batch):
            images, targets = zip(*batch)
            return torch.stack(images, dim=0), list(targets)

        self.train_dataset = YoloDataset(
            dataset_path / Path("train/images"), dataset_path / Path("train/labels"), transform=train_transform
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

        # Improved optimizer configuration
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0005,  # Lower initial learning rate
            weight_decay=2e-4,  # Increased weight decay for better regularization
            betas=(0.9, 0.999),  # Default Adam betas
            eps=1e-8,  # Default Adam epsilon
            amsgrad=True,  # Use AMSGrad variant for better convergence
        )

        # Improved scheduler configuration
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.002,  # Higher max learning rate
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.epochs,
            pct_start=0.3,  # Longer warmup phase (30% of training)
            anneal_strategy="cos",  # Cosine annealing
            three_phase=True,  # Use three-phase learning rate schedule
        )

        # Add gradient clipping
        self.grad_clip = 1.0  # Clip gradients to this value

        self.criterion = DetectionLoss(lambda_coord, lambda_noobj, lambda_conf, lambda_iou, lambda_metrics).to(
            self.device
        )

    def save_model(self, final_save: bool = False) -> None:
        torch.save(self.model.state_dict(), self.output_folder / "best.pt")

        if final_save:
            model_data = {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_boxes": self.num_boxes,
                "lambda_coord": self.criterion.lambda_coord,
                "lambda_noobj": self.criterion.lambda_noobj,
                "lambda_conf": self.criterion.lambda_conf,
                "lambda_iou": self.criterion.lambda_iou,
                "lambda_metrics": self.criterion.lambda_metrics,
            }
            with open(self.output_folder / "summary.json", "w") as file:
                json.dump(model_data, file, indent=4)

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

            target_bboxes = torch.stack(
                [
                    torch.cat([t, torch.zeros(max(0, num_boxes - t.shape[0]), 4, device=self.device)])
                    for t in target_bboxes_list
                ]
            )[:batch_size]
            target_classes = torch.stack(
                [
                    torch.cat(
                        [t, torch.full((max(0, num_boxes - t.shape[0]),), 2, dtype=torch.long, device=self.device)]
                    )
                    for t in target_classes_list
                ]
            )[:batch_size]

            self.optimizer.zero_grad()
            pred_bboxes, pred_classes, confidences = self.model(images)

            loss = self.criterion(pred_bboxes, pred_classes, confidences, target_bboxes, target_classes)
            loss = torch.Tensor(loss)

            if torch.isnan(loss):
                continue

            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

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
        all_preds = []
        all_targets = []

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
                            [t, torch.full((max(0, num_boxes - t.shape[0]),), 2, dtype=torch.long, device=self.device)]
                        )
                        for t in target_classes
                    ]
                )[:batch_size]

                pred_bboxes, class_logits, confidences = self.model(images)
                pred_classes = torch.argmax(class_logits, dim=-1)

                loss = self.criterion(pred_bboxes, class_logits, confidences, target_bboxes, target_classes)

                if torch.isnan(loss):
                    continue

                total_loss += loss.item()

                all_preds.extend(pred_classes.view(-1).cpu().tolist())
                all_targets.extend(target_classes.view(-1).cpu().tolist())

        self.precision = precision_score(all_targets, all_preds, average="macro", zero_division=0)
        self.recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        self.f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

        print(f"Precision: {self.precision:.4f} | Recall: {self.recall:.4f} | F1: {self.f1:.4f}")

        if not val:
            with open(self.output_folder / "test_results.json", mode="w") as file:
                data = {
                    "precision": self.precision,
                    "recall": self.recall,
                    "f1": self.f1,
                    "loss": total_loss / len(dataloader),
                }
                json.dump(data, file, indent=4)

        return total_loss / len(dataloader)

    def train_and_validate(self, val: bool = True) -> None:
        patience = max(5, int(self.epochs * (0.1)))
        output_folder_template = Path("runs/detect/custom_model/train")
        self.output_folder = output_folder_template
        i = 2
        while self.output_folder.exists():
            self.output_folder = output_folder_template.parent / f"{output_folder_template.name}_{i}"
            i += 1

        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.metrics_csv_file = self.output_folder / "train_metrics.csv"

        with open(self.metrics_csv_file, mode="w") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "recall", "precision", "f1", "train_loss", "val_loss"])

            best_val_loss = float("inf")
            patience_counter = 0
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                train_loss = self.train_model()
                val_loss = self.validate_model(val)
                writer.writerow(
                    [
                        epoch,
                        round(self.recall, 4),
                        round(self.precision, 4),
                        round(self.f1, 4),
                        round(train_loss, 4),
                        round(val_loss, 4),
                    ]
                )

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
) -> list[tuple[Any, Any, torch.Tensor, torch.Tensor, torch.Tensor]]:
    model.eval()
    detections = []

    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Running Inference"):
            images = images.to(device)
            pred_bboxes, pred_classes, confidences = model(images)

            for i in range(len(images)):
                # Get class probabilities
                probs = torch.softmax(pred_classes[i], dim=-1)
                scores, labels = probs.max(dim=-1)

                # Apply confidence threshold
                keep = scores > conf_threshold
                boxes = pred_bboxes[i][keep]
                scores = scores[keep]
                labels = labels[keep]
                confs = confidences[i][keep]

                if len(boxes) > 0:
                    # Apply NMS
                    keep_idx = ops.nms(boxes, scores, iou_threshold)
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                    labels = labels[keep_idx]
                    confs = confs[keep_idx]

                detections.append((images[i].cpu(), boxes.cpu(), labels.cpu(), scores.cpu(), confs.cpu()))

    return detections


def visualize_detections(
    detections: list[Any], class_names: list[str], num_images: int = 5, conf_threshold: float = 0.2
) -> None:
    for i in range(min(num_images, len(detections))):
        image, boxes, labels, scores, confidences = detections[i]
        # Handle grayscale image (1 channel)
        image = image.squeeze(0).cpu().numpy()  # Remove channel dimension
        image = (image * 255).astype("uint8")
        image = np.ascontiguousarray(image)

        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        h, w = image.shape  # Now shape is (H, W) for grayscale
        for box, label, score, conf in zip(boxes, labels, scores, confidences):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            label_name = class_names[label] if label < len(class_names) else f"Class {label}"
            cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_rgb,
                f"{label_name} {score:.2f} (conf: {conf.item():.2f})",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        plt.figure(figsize=(8, 8))
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.show()


def test_model_on_images(
    model: ObjectDetector, dataloader: DataLoader, class_names: list[str], device: torch.device, num_images: int = 5
) -> None:
    detections = run_inference(model, dataloader, device)
    visualize_detections(detections, class_names, num_images)


def inner_main(
    dataset_path: Path = typer.Argument(..., help="path to dataset"),
    image_size: int = typer.Option(600, help="image size"),
    epochs: int = typer.Option(200, help="epochs"),
    batch_size: int = typer.Option(64, help="batch size"),
    num_classes: int = typer.Option(3, help="number of classes"),
    num_boxes: int = typer.Option(30, help="number of boxes"),
    train: bool = typer.Option(True, help="train model"),
    model_path: Path = typer.Option(None, help="model path"),
    bacbkone_trainable_layers: int = typer.Option(5, help="number of backbone trainable layers"),
    lambda_coord: float = typer.Option(5.0, help="lambda coord"),
    lambda_noobj: float = typer.Option(0.5, help="lambda noobj"),
    lambda_conf: float = typer.Option(1.0, help="lambda conf"),
    lambda_iou: float = typer.Option(3.0, help="lambda iou"),
    lambda_metrics: float = typer.Option(5.0, help="lambda metrics"),
) -> None:
    class_names = ["Class 0", "Class 1", "Class 2"]
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
        epochs=epochs,
        trainable_layers=bacbkone_trainable_layers,
        lambda_coord=lambda_coord,
        lambda_noobj=lambda_noobj,
        lambda_conf=lambda_conf,
        lambda_iou=lambda_iou,
        lambda_metrics=lambda_metrics,
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
