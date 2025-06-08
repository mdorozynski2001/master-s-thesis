import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
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
        lambda_box: float = 5.0,
        lambda_cls: float = 1.0,
    ) -> None:
        super(DetectionLoss, self).__init__()
        self.box_loss = nn.MSELoss(reduction="none")
        self.cls_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=2)
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls

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

        # Create object mask (class 2 is background)
        obj_mask = (target_classes != 2).float()  # [batch_size, num_boxes]

        # Box loss (MSE)
        box_loss = self.box_loss(pred_bboxes, target_bboxes)  # [batch_size, num_boxes, 4]
        box_loss = box_loss.mean(dim=-1)  # [batch_size, num_boxes]
        box_loss = (box_loss * obj_mask).sum() / (obj_mask.sum() + 1e-16)

        # Classification loss (CrossEntropy)
        pred_classes = pred_classes.permute(0, 2, 1)  # [batch_size, num_classes, num_boxes]
        cls_loss = self.cls_loss(pred_classes, target_classes)  # [batch_size, num_boxes]
        cls_loss = (cls_loss * obj_mask).sum() / (obj_mask.sum() + 1e-16)

        # Confidence loss (MSE)
        conf_loss = self.box_loss(confidences.squeeze(-1), obj_mask).mean()

        # Combine losses
        total_loss = self.lambda_box * box_loss + self.lambda_cls * cls_loss + conf_loss

        return total_loss


class Backbone(nn.Module):
    def __init__(self, trainable_layers: int = 5) -> None:
        super(Backbone, self).__init__()

        self.feature_extractor = nn.Sequential(
            # First block - input 600x600x3 -> 300x300x32
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            # Second block - 300x300x32 -> 150x150x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.1),
            # Third block - 150x150x64 -> 75x75x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            # Fourth block - 75x75x128 -> 38x38x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
            # Fifth block - 38x38x256 -> 19x19x512
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),
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
    dummy_input = torch.randn(1, 3, image_size, image_size)

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
        lambda_box: float = 5.0,
        lambda_cls: float = 1.0,
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
            lr=0.001,  # Slightly higher initial learning rate
            weight_decay=1e-4,  # Reduced weight decay
            betas=(0.9, 0.999),
        )

        # Improved scheduler configuration
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=0.001,  # Reduced max learning rate
            steps_per_epoch=len(self.train_dataloader),
            epochs=self.epochs,
            pct_start=0.2,  # Shorter warmup
            anneal_strategy="cos",
        )

        # Add gradient clipping
        self.grad_clip = 1.0  # Clip gradients to this value

        self.criterion = DetectionLoss(lambda_box, lambda_cls)

    def save_model(self, final_save: bool = False) -> None:
        torch.save(self.model.state_dict(), self.output_folder / "best.pt")

        if final_save:
            model_data = {
                "num_classes": self.num_classes,
                "image_size": self.image_size,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "num_boxes": self.num_boxes,
                "lambda_box": self.criterion.lambda_box,
                "lambda_cls": self.criterion.lambda_cls,
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

    def validate_model(self, val: bool = True) -> float:
        if val:
            dataloader = self.val_dataloader
        else:
            dataloader = self.test_dataloader

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

                # Filter out background class (2) for metrics
                mask = target_classes != 2
                filtered_preds = pred_classes[mask].cpu().tolist()
                filtered_targets = target_classes[mask].cpu().tolist()

                all_preds.extend(filtered_preds)
                all_targets.extend(filtered_targets)

        # Calculate metrics only on non-background classes
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
        patience = max(5, int(self.epochs * (0.05)))
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
            writer.writerow(["epoch", "recall", "precision", "f1", "train_loss", "val_loss", "train_time", "val_time"])

            best_val_loss = float("inf")
            patience_counter = 0
            for epoch in range(self.epochs):
                print(f"\nEpoch {epoch + 1}/{self.epochs}")
                start_train_time = time.time()
                train_loss = self.train_model()
                end_train_time = time.time()
                train_time = int(end_train_time - start_train_time)
                train_time_string = f"{train_time // 60}:{train_time % 60}"

                start_val_time = time.time()
                val_loss = self.validate_model(val)
                end_val_time = time.time()
                val_time = int(end_val_time - start_val_time)
                val_time_string = f"{val_time // 60}:{val_time % 60}"

                writer.writerow(
                    [
                        epoch,
                        round(self.recall, 4),
                        round(self.precision, 4),
                        round(self.f1, 4),
                        round(train_loss, 4),
                        round(val_loss, 4),
                        train_time_string,
                        val_time_string,
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
    detections: list[Any], class_names: list[str], output_dir: Path, num_images: int = 5, conf_threshold: float = 0.1
) -> None:
    # Create output directories
    detections_dir = output_dir / "with_detections"
    original_dir = output_dir / "original"
    detections_dir.mkdir(parents=True, exist_ok=True)
    original_dir.mkdir(parents=True, exist_ok=True)

    current_idx = 0

    def on_trackbar_change(x):
        nonlocal current_idx
        current_idx = x

    cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Image", "Detections", 0, len(detections) - 1, on_trackbar_change)

    while True:
        if current_idx >= len(detections):
            current_idx = 0
        elif current_idx < 0:
            current_idx = len(detections) - 1

        image, boxes, labels, scores, confidences = detections[current_idx]
        # Convert tensor to numpy array and adjust for RGB visualization
        image = image.permute(1, 2, 0).cpu().numpy()  # Change from (C, H, W) to (H, W, C)
        image = (image * 255).astype("uint8")
        # Convert RGB to BGR for OpenCV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = np.ascontiguousarray(image)

        # Create a copy for detections
        image_with_detections = image.copy()

        h, w, _ = image.shape
        for box, label, score, conf in zip(boxes, labels, scores, confidences):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)

            label_name = class_names[label] if label < len(class_names) else f"Class {label}"
            cv2.rectangle(image_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image_with_detections,
                label_name,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Detections", image_with_detections)
        cv2.setTrackbarPos("Image", "Detections", current_idx)

        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("n"):
            current_idx += 1
        elif key == ord("p"):
            current_idx -= 1
        elif key == ord("s"):
            # Save both images
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(detections_dir / f"image_{current_idx}_{timestamp}.jpg"), image_with_detections)
            cv2.imwrite(str(original_dir / f"image_{current_idx}_{timestamp}.jpg"), image)
            print(f"Saved images to {output_dir}")

    cv2.destroyAllWindows()


def test_model_on_images(
    model: ObjectDetector, dataloader: DataLoader, class_names: list[str], device: torch.device, output_dir: Path, num_images: int = 5
) -> None:
    detections = run_inference(model, dataloader, device)
    visualize_detections(detections, class_names, output_dir, num_images)


def inner_main(
    dataset_path: Path = typer.Argument(..., help="path to dataset"),
    image_size: int = typer.Option(640, help="image size"),
    epochs: int = typer.Option(200, help="epochs"),
    batch_size: int = typer.Option(64, help="batch size"),
    num_classes: int = typer.Option(3, help="number of classes"),
    num_boxes: int = typer.Option(30, help="number of boxes"),
    model_path: Path = typer.Option(None, help="model path"),
    bacbkone_trainable_layers: int = typer.Option(4, help="number of backbone trainable layers"),
    lambda_box: float = typer.Option(5.0, help="lambda box"),
    lambda_cls: float = typer.Option(1.0, help="lambda cls"),
    output_dir: Path = typer.Option(Path("test_images"), help="output directory for saved images"),
) -> None:
    class_names = ["Person", "Forklift", "Class 2"]
    trainer = ObjectDetectorTrainer(
        image_size=image_size,
        batch_size=batch_size,
        num_classes=num_classes,
        num_boxes=num_boxes,
        dataset_path=dataset_path,
        epochs=epochs,
        trainable_layers=bacbkone_trainable_layers,
        lambda_box=lambda_box,
        lambda_cls=lambda_cls,
    )
    trainer.model.load_state_dict(torch.load(model_path))
    test_model_on_images(trainer.model, trainer.test_dataloader, class_names, trainer.device, output_dir, num_images=1000)


def main() -> None:
    typer.run(inner_main)


if __name__ == "__main__":
    main()
