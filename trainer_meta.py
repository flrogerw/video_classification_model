import os
import json
import cv2
import clip
from PIL import Image
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from torch import nn
from torch.utils.data import Dataset, random_split, DataLoader, Subset
from typing import List, Tuple, Dict, Any
from sklearn.metrics import confusion_matrix
from clip_frame_classifier import CLIPWithMetadataClassifier
from contact_sheet import VideoContactSheet

CONFUSION_MATRIX = False  # Whether to show the confusion matrix
EPOCH_COUNT = 12  # Number of times to loop the dataset
FRAME_BUFFER = 0  # Number of seconds on either side of an annotation timestamp
FPS = 0.3  # How often to grab a frame for the dataset
DATA_BATCH_SIZE = 32
LABEL_COUNT = 3  # 0 = normal content, 1 = bumpers, 2 = commercial
MODEL = "meta_clip_classifier.pt"
RETRAIN_MODEL = "meta_clip_classifier.pt"
CLIP_MODEL = "ViT-L/14"  # ViT-B/32
TARGET_CLASSES = [1, 2]  # 0 = normal content, 1 = bumpers, 2 = commercial
SKIP_BLACK = True  # Set to False when classifying commercials.
RETRAIN = False
NORMALIZED_STATS = "normalization_stats.pt"
BALANCE_TOLERANCE = 3  # The class tolerance for the balanced dataset.


def balanced_subset(dataset: Dataset, tolerance: float = BALANCE_TOLERANCE) -> Subset:
    """
    Returns a balanced subset where the number of samples for each class
    is within ±tolerance of the smallest class.

    Args:
        dataset: Input dataset (must return (image, metadata, label) tuples).
        tolerance: Allowed deviation from the smallest class (e.g., 0.1 = 10%).

    Returns:
        Subset: A balanced subset of the dataset.
    """
    try:
        label_to_indices = defaultdict(list)

        for idx in range(len(dataset)):
            sample = dataset[idx]
            if len(sample) == 3:
                _, label, _ = sample
            elif len(sample) == 2:
                _, label = sample
            else:
                raise ValueError(f"Unexpected dataset sample format at index {idx}: {sample}")

            label = int(label)
            if label < 0:
                continue
            label_to_indices[label].append(idx)

        print("Label distribution in full dataset:")
        for label, indices in sorted(label_to_indices.items()):
            print(f"Class {label}: {len(indices)} samples")

        existing_labels = [label for label in label_to_indices if len(label_to_indices[label]) > 0]

        if not existing_labels:
            raise RuntimeError("No class has any samples; cannot create a balanced subset.")

        min_class_count = min(len(label_to_indices[label]) for label in existing_labels)
        lower_bound = max(0, int(min_class_count * (1 - tolerance)))
        upper_bound = int(min_class_count * (1 + tolerance))

        print(f"\nBalancing all classes to be within [{lower_bound}, {upper_bound}] samples")

        balanced_indices = []

        for label in existing_labels:
            indices = label_to_indices[label]
            if len(indices) > upper_bound:
                sampled = random.sample(indices, upper_bound)
                print(f"Class {label}: downsampled to {upper_bound}")
            elif len(indices) < lower_bound:
                print(f"Class {label}: skipped (only {len(indices)} < {lower_bound})")
                continue
            else:
                sampled = indices
                print(f"Class {label}: kept all {len(indices)}")

            balanced_indices.extend(sampled)

        if not balanced_indices:
            raise RuntimeError("No samples were added to the balanced subset.")

        random.shuffle(balanced_indices)
        return Subset(dataset, balanced_indices)
    except Exception as e:
        raise RuntimeError(f"Failed to create balanced subset: {e}")


def analyze_dataset(dataset: Dataset) -> Counter:
    """
    Analyze and print the distribution of labels in a dataset.

    Args:
        dataset: A dataset or Subset containing (x, label) pairs.

    Returns:
        Counter: A counter object with label frequencies.
    """
    label_counts = Counter()
    try:
        if isinstance(dataset, Subset):
            indices = dataset.indices
            base_dataset = dataset.dataset
            for idx in indices:
                _, label, _ = base_dataset[idx]
                label_counts[int(label)] += 1
        else:
            for idx in range(len(dataset)):
                _, label = dataset[idx]
                label_counts[int(label)] += 1
        return label_counts
    except Exception as e:
        raise RuntimeError(f"Error analyzing dataset: {e}")


def compute_class_weights(counter: Counter, num_classes: int = 3) -> torch.Tensor:
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


def get_label(timestamp: float, annotation: Dict[str, Any]) -> int:
    """
    Assign a label to the frame timestamp based on annotation segments.

    Args:
        timestamp: Time in seconds of the current frame.
        annotation: Dictionary containing 'bumpers', and 'commercials' time segments.

    Returns:
        Integer label: 1 = bumpers, 2 = commercial, 0 = normal content.
    """
    commercials = annotation.get("commercials", [])
    bumpers = annotation.get("bumpers", [])

    if bumpers and any(start <= timestamp <= end for start, end in bumpers):
        return 1
    elif commercials and any(start <= timestamp <= end for start, end in commercials):
        return 2
    else:
        return 0


class VideoFrameClipDataset(Dataset):
    def __init__(self, annotation_dir: str, interval_sec: float = 1.0):
        self.annotation_dir = annotation_dir
        self.interval_sec = interval_sec
        self.samples: List[Dict] = []
        self.device = "cuda" if torch.cuda.is_available() else "mps"

        try:
            self.model, self.preprocess = clip.load(CLIP_MODEL, device=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model: {e}")

        annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith(".json")]

        for annotation_file in annotation_files:
            annotation_path = os.path.join(annotation_dir, annotation_file)

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

            try:
                cap = cv2.VideoCapture(video_file)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    raise ValueError("Invalid FPS")
                total_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
                cap.release()
            except Exception as e:
                print(f"Error probing video {video_file}: {e}")
                continue

            time_ranges = []

            for key in ("bumpers", "commercials", "content"):
                segments = annotation.get(key)

                if not segments:
                    continue

                if isinstance(segments, list) and all(isinstance(x, (int, float)) for x in segments):
                    segments = [segments]

                for seg in segments:
                    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
                        continue

                    start, end = seg
                    start = max(0, (start or 0) - FRAME_BUFFER)
                    end = (end or 0) + FRAME_BUFFER  # <<---- WRONG
                    time_ranges.append((start, end))

            if not time_ranges:
                continue

            t = 0.0
            epsilon = 0.01
            while t < (total_duration - epsilon):
                if any(start <= t <= end for start, end in time_ranges):
                    label = get_label(t, annotation)
                    self.samples.append({
                        "video_path": video_file,
                        "timestamp": t,
                        "video_duration": total_duration,
                        "label": label
                    })
                t += self.interval_sec

    @staticmethod
    def normalize(value, min_val, max_val):
        if max_val == min_val:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        sample = self.samples[idx]
        video_path = sample["video_path"]
        video_duration = sample['video_duration']
        timestamp = sample["timestamp"]
        label = sample["label"]

        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Could not read frame at {timestamp}s in {video_path}")

        if SKIP_BLACK and self.is_black_frame(frame):
            return torch.zeros((512,), device=self.device), -1, {}

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = self.preprocess(image).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_tensor.unsqueeze(0)).squeeze(0)

        metadata = {
            "relative_position": timestamp / video_duration if video_duration > 0 else 0.0
        }

        return embedding, label, metadata

    @staticmethod
    def is_black_frame(frame: np.ndarray, threshold: int = 1.0) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray.mean() < threshold


def train(device: str, model: nn.Module) -> None:
    """
    Train the CLIP classifier using video frame embeddings and additional metadata features.
    """
    try:
        # Load and balance dataset
        full_dataset = VideoFrameClipDataset("dataset/annotations", interval_sec=FPS)
        balanced_dataset = balanced_subset(full_dataset)
        class_counts = analyze_dataset(balanced_dataset)
        class_weights = compute_class_weights(class_counts).to(device)

        # Split into train/val
        train_size = int(0.8 * len(balanced_dataset))
        val_size = len(balanced_dataset) - train_size
        train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=DATA_BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=DATA_BATCH_SIZE)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Slightly lower learning rate(e.g., 1e-4 → 5e-5)
        # Use AdamW instead of Adam for better generalization.
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        for epoch in range(EPOCH_COUNT):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for image_inputs, labels, metadata_inputs in train_loader:
                image_inputs = image_inputs.to(device).float()
                labels = labels.to(device)
                meta_tensor = torch.cat(
                    [metadata_inputs[k].unsqueeze(1).float().to(device) for k in metadata_inputs],
                    dim=1)

                preds = model(image_inputs, meta_tensor)
                loss = loss_fn(preds, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (preds.argmax(1) == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            print(f"Epoch {epoch + 1} - Train Loss: {total_loss:.4f} - Train Acc: {train_acc:.2f}")

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for image_inputs, labels, metadata_inputs in val_loader:
                    image_inputs = image_inputs.to(device).float()
                    meta_tensor = torch.cat(
                        [metadata_inputs[k].unsqueeze(1).float().to(device) for k in metadata_inputs],
                        dim=1)
                    labels = labels.to(device)

                    outputs = model(image_inputs, meta_tensor)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item() * image_inputs.size(0)
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    if CONFUSION_MATRIX:
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

            val_acc = val_correct / val_total if val_total > 0 else 0.0
            print(f"           Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}")

            if CONFUSION_MATRIX and all_preds and all_labels:
                cm = confusion_matrix(all_labels, all_preds)
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_CLASSES,
                            yticklabels=TARGET_CLASSES)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.show()

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = torch.load(RETRAIN_MODEL, weights_only=False) if RETRAIN else (
        # CLIPWithMetadataClassifier(meta_weight=0.9).to(device))
        CLIPWithMetadataClassifier(embedding_dim=768, metadata_dim=1, hidden_dim=128, num_classes=3,
                                   meta_weight=1).to(device))
    train(device, model)
    save_model = f"retrained_{MODEL}" if RETRAIN else MODEL
    torch.save(model, save_model)
