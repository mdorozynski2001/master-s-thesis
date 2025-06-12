from pathlib import Path

import cv2
import numpy as np
import torch
import typer
from torchvision import transforms

from custom_trainer.model import Backbone, ObjectDetector


def process_frame(frame: np.ndarray, model: ObjectDetector, device: torch.device, image_size: int = 600) -> np.ndarray:
    # Prepare image for model
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ]
    )

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Transform image
    image = transform(frame_rgb).unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        pred_bboxes, pred_classes, confidences = model(image)

    # Process predictions for first (and only) image in batch
    probs = torch.softmax(pred_classes[0], dim=-1)
    scores, labels = probs.max(dim=-1)

    # Apply confidence threshold
    conf_threshold = 0.1
    keep = confidences[0].squeeze(-1) > conf_threshold
    boxes = pred_bboxes[0][keep]
    scores = scores[keep]
    labels = labels[keep]
    confs = confidences[0][keep]

    # Draw detections
    h, w = frame.shape[:2]
    for box, label, score, conf in zip(boxes, labels, scores, confs):
        x1, y1, x2, y2 = box.tolist()
        x1 = int(x1 * w)
        y1 = int(y1 * h)
        x2 = int(x2 * w)
        y2 = int(y2 * h)

        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label_text = f"Class {label.item()} {score:.2f} (conf: {conf.item():.2f})"
        cv2.putText(
            frame,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return frame


def main(
    model_path: Path = typer.Argument(..., help="Path to the model file"),
    image_size: int = typer.Option(640, help="Input image size"),
    num_classes: int = typer.Option(3, help="Number of classes"),
    num_boxes: int = typer.Option(30, help="Number of boxes"),
    camera_id: int = typer.Option(0, help="Camera ID (usually 0 for default webcam)"),
) -> None:
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = ObjectDetector(Backbone(), num_classes, num_boxes, image_size).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Press 'q' to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Process frame
            processed_frame = process_frame(frame, model, device, image_size)

            # Display FPS
            cv2.imshow("Live Detection", processed_frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
