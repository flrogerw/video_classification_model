import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from collections import Counter
from dotenv import load_dotenv
from typing import Optional

from classes.video_utils import ConfusionMatrix, ConfidenceGraph
from classes.video_classifier import VideoFrameClipDataset

class ClipClassifierTrainer:
    """
    Trainer class for CLIP-based video frame classifier.
    Handles:
        - Dataset loading & balancing
        - Train/validation split
        - Training & validation loop
        - Confusion matrix plotting
        - Confidence graph plotting
    """

    def __init__(self):
        # Load env variables once
        load_dotenv()

        # Config from env
        self.confusion_matrix_enabled = os.getenv("CONFUSION_MATRIX", "False").lower() in ("true", "1", "yes")
        self.annotations_dir: Optional[str] = os.getenv("ANNOTATIONS_DIR")
        self.black_threshold: float = float(os.getenv("BLACK_THRESHOLD", 1.0))
        self.motion_threshold: float = float(os.getenv("MOTION_THRESHOLD", 2.0))
        self.clip_model_name: Optional[str] = os.getenv("CLIP_MODEL")
        self.fps_interval: float = float(os.getenv("FPS", 0.3))
        self.content_fps: float = float(os.getenv("CONTENT_FPS", 1.0))
        self.balance_tolerance: float = float(os.getenv("BALANCE_TOLERANCE", 1.0))
        self.batch_size: int = int(os.getenv("DATA_BATCH_SIZE", 32))
        self.epochs: int = int(os.getenv("EPOCH_COUNT", 10))

    def _load_dataset(self) -> Subset:
        """Load and return the balanced dataset."""
        full_dataset = VideoFrameClipDataset(
            annotation_dir=self.annotations_dir,
            black_threshold=self.black_threshold,
            motion_threshold=self.motion_threshold,
            clip_model=self.clip_model_name,
            interval_sec=self.fps_interval,
            content_fps=self.content_fps
        )

        return VideoFrameClipDataset.get_balanced_subset(
            dataset=full_dataset,
            tolerance=self.balance_tolerance
        )

    @staticmethod
    def _compute_class_weights(dataset: VideoFrameClipDataset, device: str) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        class_counts: Counter = VideoFrameClipDataset.analyze_dataset(dataset)
        return VideoFrameClipDataset.compute_class_weights(
            counter=class_counts,
            num_classes=2
        ).to(device)

    def train(self, device: str, model: nn.Module) -> None:
        """Main training loop."""
        try:
            dataset = self._load_dataset()
            class_weights = self._compute_class_weights(dataset, device)

            # Split dataset
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            # Data loaders
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, pin_memory=False)

            # Loss & optimizer
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            val_accuracies, val_confidences = [], []

            for epoch in range(self.epochs):
                self._train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch)
                val_acc, avg_conf = self._validate(model, val_loader, loss_fn, device, epoch, val_accuracies, val_confidences)

            # Plot confidence graph
            graph = ConfidenceGraph()
            graph.set_data(val_accuracies, val_confidences)
            graph.plot()

        except Exception as e:
            raise RuntimeError(f"Training failed: {e}")

    @staticmethod
    def _train_one_epoch(model, train_loader, loss_fn, optimizer, device, epoch):
        """Run a single training epoch."""
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).long()
            preds = model(inputs)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (preds.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch + 1} - Train Loss: {total_loss:.4f} - Train Acc: {train_acc:.2f}")

    def _validate(self, model, val_loader, loss_fn, device, epoch, val_accuracies, val_confidences):
        """Run validation for one epoch."""
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        all_preds, all_labels, all_confidences = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, dim=1)

                all_confidences.extend(max_probs.cpu().tolist())
                loss = loss_fn(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                if self.confusion_matrix_enabled:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        val_acc = val_correct / val_total if val_total > 0 else 0.0
        val_accuracies.append(val_acc)
        val_confidences.append(avg_conf)
        val_loss /= val_total if val_total > 0 else 1

        print(f"           Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f} - Confidence: {avg_conf}")

        if self.confusion_matrix_enabled and all_preds and all_labels:
            plotter = ConfusionMatrix(target_classes=['Content', 'Bumpers'])
            plotter.set_labels(all_labels, all_preds)
            plotter.plot(epoch=epoch + 1)

        return val_acc, avg_conf
