"""
Training Module for CLIP Classifier

This module trains a CLIP-based classifier on video frame embeddings,
using additional metadata features. It balances datasets, splits
training/validation sets, and optionally plots confusion matrices and
confidence graphs based on environment variables.

Environment variables loaded once at the top for configuration.
"""

import os
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from collections import Counter
from classes.utils import ConfusionMatrix, ConfidenceGraph
from classes.clip_classifier import VideoFrameClipDataset
from dotenv import load_dotenv
from typing import Optional

# Load environment variables once
load_dotenv()

# Config flags and hyperparameters
CONFUSION_MATRIX: bool = os.getenv("CONFUSION_MATRIX", "False").lower() in ("true", "1", "yes")
ANNOTATIONS_DIR: Optional[str] = os.getenv("ANNOTATIONS_DIR")
BLACK_THRESHOLD: float = float(os.getenv("BLACK_THRESHOLD", 1.0))
MOTION_THRESHOLD: float = float(os.getenv("MOTION_THRESHOLD", 2.0))
CLIP_MODEL_NAME: Optional[str] = os.getenv("CLIP_MODEL")
FPS_INTERVAL: float = float(os.getenv("FPS", 0.3))
BALANCE_TOLERANCE: float = float(os.getenv("BALANCE_TOLERANCE", 1.0))
DATA_BATCH_SIZE: int = int(os.getenv("DATA_BATCH_SIZE", 32))
EPOCH_COUNT: int = int(os.getenv("EPOCH_COUNT", 10))


def train(device: str, model: nn.Module) -> None:
    """
    Train the CLIP classifier using video frame embeddings and metadata.

    Args:
        device: Device to run training on ("cuda", "mps", or "cpu").
        model: The classifier model to train.

    Raises:
        RuntimeError: If training fails due to dataset loading or runtime errors.
    """
    try:
        # Load full dataset and balance it
        full_dataset = VideoFrameClipDataset(
            annotation_dir=ANNOTATIONS_DIR,
            black_threshold=BLACK_THRESHOLD,
            motion_threshold=MOTION_THRESHOLD,
            clip_model=CLIP_MODEL_NAME,
            interval_sec=FPS_INTERVAL,
        )

        balanced_dataset = VideoFrameClipDataset.get_balanced_subset(
            dataset=full_dataset,
            tolerance=BALANCE_TOLERANCE
        )

        # Analyze class distribution
        class_counts: Counter = VideoFrameClipDataset.analyze_dataset(balanced_dataset)
        class_weights = VideoFrameClipDataset.compute_class_weights(
            counter=class_counts, num_classes=2
        ).to(device)

        # Train/validation split
        train_size = int(0.8 * len(balanced_dataset))
        val_size = len(balanced_dataset) - train_size
        train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

        # Data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=DATA_BATCH_SIZE, shuffle=True, pin_memory=False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=DATA_BATCH_SIZE, pin_memory=False
        )

        # Loss and optimizer
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        val_accuracies = []
        val_confidences = []

        for epoch in range(EPOCH_COUNT):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            # Training loop
            for inputs, labels in train_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

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

            # Validation loop
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            all_preds = []
            all_labels = []
            all_confidences = []

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

                    if CONFUSION_MATRIX:
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

            average_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
            val_acc = val_correct / val_total if val_total > 0 else 0.0
            val_accuracies.append(val_acc)
            val_confidences.append(average_confidence)
            val_loss /= val_total if val_total > 0 else 1

            print(f"           Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}")

            if CONFUSION_MATRIX and all_preds and all_labels:
                plotter = ConfusionMatrix(target_classes=['Content', 'Bumpers'])
                plotter.set_labels(all_labels, all_preds)
                plotter.plot(epoch=epoch + 1)

        # Plot confidence graph after training
        graph = ConfidenceGraph()
        graph.set_data(val_accuracies, val_confidences)
        graph.plot()

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
