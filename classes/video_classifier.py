import json
import os
import random
import clip
import cv2
import numpy as np
import torch

from collections import defaultdict
from typing import Any, Counter
from torch import nn
from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset, Subset

"""
Module: clip_frame_classifier
-----------------------------
This module defines a simple feedforward PyTorch neural network classifier
designed to classify CLIP image embeddings into a fixed number of classes.
"""


class ClipFrameClassifier(nn.Module):
    """
    Simple feedforward classifier for CLIP embeddings.

    Attributes:
        fc (nn.Sequential): Feedforward layers including linear, ReLU activations, and dropout.
    """

    def __init__(self, input_dim: int = 512, num_classes: int = 2) -> None:
        """
        Initialize the classifier layers.

        Args:
            input_dim (int): Dimensionality of the input embeddings.
            num_classes (int): Number of output classes for classification.
        """
        super().__init__()
        try:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        except Exception as e:
            print(f"[ERROR] Failed to initialize layers: {e}")
            raise

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the classifier.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output logits tensor of shape (batch_size, num_classes).
        """
        try:
            return self.fc(x)
        except Exception as e:
            print(f"[ERROR] Forward pass failed: {e}")
            raise


"""
video_frame_clip_dataset.py

Dataset class for lazily loading video frames, skipping black frames, 
and computing CLIP embeddings on-the-fly. 
Stores only metadata (video path, timestamp, label) to minimize memory usage.
"""


class VideoFrameClipDataset(Dataset):
    """
    A PyTorch Dataset for lazily loading video frames and computing CLIP embeddings.

    Frames are loaded on-demand, with optional black frame and motion threshold filtering.
    Metadata (video path, timestamp, label) is stored to minimize memory usage.
    """

    def __init__(
            self,
            annotation_dir: str,
            clip_model: str,
            black_threshold: float = 1.0,
            motion_threshold: float = 2.0,
            interval_sec: float = 1.0,
            content_fps: float = 1.0
    ) -> None:
        """
        Initialize the dataset.

        Args:
            annotation_dir: Directory containing JSON annotation files.
            clip_model: Name of the CLIP model to load.
            black_threshold: Grayscale mean threshold to consider a frame "black".
            motion_threshold: Threshold for detecting motion changes.
            interval_sec: Frame sampling interval in seconds.
        """
        self.annotation_dir = annotation_dir
        self.interval_sec = interval_sec
        self.samples: list[tuple[str, float, int]] = []
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.prev_frame = None
        self.clip_model = clip_model
        self.black_threshold = black_threshold
        self.motion_threshold = motion_threshold
        self.content_fps = content_fps

        # Load CLIP model
        try:
            self.model, self.preprocess = clip.load(self.clip_model, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

        # Get list of annotation files
        try:
            annotation_files = [f for f in os.listdir(self.annotation_dir) if f.endswith(".json")]
        except Exception as e:
            raise RuntimeError(f"Error reading annotation directory '{self.annotation_dir}': {e}")

        # Process each annotation file
        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotation_dir, annotation_file)

            # Load annotation JSON
            try:
                with open(annotation_path, "r") as f:
                    annotation = json.load(f)
            except Exception as e:
                print(f"Error reading {annotation_file}: {e}")
                continue

            video_file = annotation.get("file_path")
            if not video_file or not os.path.exists(video_file):
                print(f"Skipping missing video: {video_file}")
                continue

            duration = annotation.get("video_duration")

            # Extract annotation time ranges
            time_ranges = []
            for key in ("bumpers", "content", "commercial"):
                segments = annotation.get(key)
                if not segments:
                    continue
                if isinstance(segments[0], (int, float)):
                    segments = [segments]
                for start, end in segments:
                    start = max(0, (start or 0))
                    end = (end or duration)
                    time_ranges.append((start, end))

            if not time_ranges:
                continue

            # Collect samples at the given interval
            t = 0.0
            epsilon = 0.01  # Prevent going out of bounds
            while t < (duration - epsilon):
                if any(start <= t <= end for start, end in time_ranges):
                    try:
                        label = self.get_label(t, annotation)
                        self.samples.append((video_file, t, label))
                    except Exception as e:
                        print(f"Error assigning label at {t}s in {video_file}: {e}")
                t += 1 if label == 0 else self.interval_sec

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a frame embedding and label for the given index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (CLIP embedding tensor, label tensor).
            If a black frame is detected, returns a zero vector and label -1.
        """
        try:
            video_path, timestamp, label = self.samples[idx]
        except IndexError:
            raise RuntimeError(f"Index {idx} is out of bounds for dataset length {len(self.samples)}")

        # Open video and read frame
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame at {timestamp}s in {video_path}")

        # Skip black frames
        if self.is_black_frame(frame):
            return torch.zeros((512,), device=self.device), torch.tensor(-1).to(self.device)

        # Convert and preprocess frame
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_tensor = self.preprocess(image).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Error processing frame at {timestamp}s: {e}")

        # Compute CLIP embedding
        try:
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor.unsqueeze(0)).squeeze(0)
        except Exception as e:
            raise RuntimeError(f"Error computing CLIP embedding for {video_path} at {timestamp}s: {e}")

        return embedding, torch.tensor(label).to(self.device)

    @staticmethod
    def compute_class_weights(counter: Counter, num_classes: int = 2) -> torch.Tensor:
        """
        Compute inverse frequency-based class weights.

        Args:
            counter: Counter of class frequencies.
            num_classes: Total number of classes.

        Returns:
            torch.Tensor: Weights tensor of shape [num_classes].
        """
        total = sum(counter.values())
        weights = []
        for i in range(num_classes):
            count = counter.get(i, 0)
            if count > 0:
                weights.append(total / count)
            else:
                weights.append(0.0)
        return torch.tensor(weights, dtype=torch.float)

    @staticmethod
    def analyze_dataset(dataset: Dataset) -> Counter:
        """
        Analyze label distribution in a dataset.

        Args:
            dataset: Dataset or Subset returning (x, label) pairs.

        Returns:
            Counter mapping label → frequency.
        """
        label_counts = Counter()
        try:
            if isinstance(dataset, Subset):
                indices = dataset.indices
                base_dataset = dataset.dataset
                for idx in indices:
                    _, label = base_dataset[idx]
                    label_counts[int(label)] += 1
            else:
                for idx in range(len(dataset)):
                    _, label = dataset[idx]
                    label_counts[int(label)] += 1
        except Exception as e:
            raise RuntimeError(f"Error analyzing dataset: {e}")
        return label_counts

    @staticmethod
    def get_balanced_subset(dataset: Dataset, tolerance: float = 3.0) -> Subset:
        """
        Create a balanced subset within ±tolerance of the smallest class size.

        Args:
            dataset: Dataset returning (x, label) pairs.
            tolerance: Allowed deviation ratio from the smallest class size.

        Returns:
            Subset with balanced label distribution.
        """
        try:
            label_to_indices = defaultdict(list)
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                label = int(label)
                if label >= 0:
                    label_to_indices[label].append(idx)

            # Ensure we have at least one class
            if not label_to_indices:
                raise RuntimeError("No valid labels found in dataset.")

            print("Label distribution in full dataset:")
            for label, indices in sorted(label_to_indices.items()):
                print(f"Class {label}: {len(indices)} samples")

            min_class_count = min(len(indices) for indices in label_to_indices.values())
            lower_bound = max(0, int(min_class_count * (1 - tolerance)))
            upper_bound = int(min_class_count * (1 + tolerance))
            print(f"\nBalancing all classes to be within [{lower_bound}, {upper_bound}] samples")

            balanced_indices = []
            for label, indices in label_to_indices.items():
                if len(indices) > upper_bound:
                    balanced_indices.extend(random.sample(indices, upper_bound))
                    print(f"Class {label}: downsampled to {upper_bound}")
                elif len(indices) >= lower_bound:
                    print(f"Class {label}: kept all {len(indices)}")
                    balanced_indices.extend(indices)

            if not balanced_indices:
                raise RuntimeError("Balanced subset is empty.")

            random.shuffle(balanced_indices)
            return Subset(dataset, balanced_indices)
        except Exception as e:
            raise RuntimeError(f"Failed to create balanced subset: {e}")

    @staticmethod
    def get_label(timestamp: float, annotation: dict[str, Any]) -> int:
        """
        Get label for a timestamp based on annotation intervals.

        Args:
            timestamp: Time in seconds.
            annotation: Dictionary with keys "bumpers", "commercials", "content" each
                        potentially a list of [start, end] intervals or empty/missing.

        Returns:
            int: 2 for content, 0 for bumpers, 1 for commercials.
        """
        bumpers = annotation.get("bumpers") or []
        commercials = annotation.get("commercials") or []
        content = annotation.get("content") or []

        # Check bumpers intervals
        if any(start <= timestamp <= end for start, end in bumpers):
            return 1
        # Check commercials intervals
        if any(start <= timestamp <= end for start, end in commercials):
            return 2
        # Default to content (2) if no other intervals matched
        return 0

    def is_black_frame(self, frame: np.ndarray) -> bool:
        """
        Check if frame is black or lacks motion.

        Args:
            frame: BGR frame.

        Returns:
            True if black or low motion, else False.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_val = 0.0
        if self.prev_frame is not None:
            diff = cv2.absdiff(gray, self.prev_frame)
            motion_val = float(np.mean(diff))
        self.prev_frame = gray
        return gray.mean() < self.black_threshold or motion_val < self.motion_threshold
