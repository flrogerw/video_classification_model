import os
import clip
import torch
from collections import Counter
from torch import nn
from torch.utils.data import random_split, DataLoader
from classes.utils import ConfusionMatrix, ConfidenceGraph
from classes.clip_classifier import ClipFrameClassifier, VideoFrameClipDataset
from dotenv import load_dotenv

# Load .env file
load_dotenv()

CONFUSION_MATRIX = os.getenv("CONFUSION_MATRIX", "False") == "True"

def train(device: str, model: nn.Module) -> None:
    """
    Train the CLIP classifier using video frame embeddings and additional metadata features.
    """
    try:
        # Load and balance datasets
        full_dataset = VideoFrameClipDataset(annotation_dir=os.getenv("ANNOTATIONS_DIR"),
                                             black_threshold=float(os.getenv("BLACK_THRESHOLD")),
                                             motion_threshold=float(os.getenv("MOTION_THRESHOLD")),
                                             clip_model=os.getenv("CLIP_MODEL"), interval_sec=float(os.getenv("FPS")))
        balanced_dataset = VideoFrameClipDataset.get_balanced_subset(dataset=full_dataset,
                                                                     tolerance=float(os.getenv("BALANCE_TOLERANCE")))
        class_counts = VideoFrameClipDataset.analyze_dataset(balanced_dataset)
        class_weights = VideoFrameClipDataset.compute_class_weights(counter=class_counts, num_classes=2).to(device)

        # Split into train/val
        train_size = int(0.8 * len(balanced_dataset))
        val_size = len(balanced_dataset) - train_size
        train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=int(os.getenv("DATA_BATCH_SIZE")), shuffle=True,
                                  pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=int(os.getenv("DATA_BATCH_SIZE")), pin_memory=False)

        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        val_accuracies = []
        val_confidences = []

        for epoch in range(int(os.getenv("EPOCH_COUNT"))):
            model.train()
            total_loss = 0
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                preds = model(inputs)
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
                    _, predicted = outputs.max(1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

                    if CONFUSION_MATRIX:
                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                average_confidence = sum(all_confidences) / len(all_confidences)

            val_acc = val_correct / val_total if val_total > 0 else 0.0
            val_accuracies.append(val_acc)
            val_confidences.append(average_confidence)
            val_loss /= val_total

            print(f"           Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}")

            if CONFUSION_MATRIX and all_preds and all_labels:
                plotter = ConfusionMatrix(target_classes=['Content', 'Bumpers'])
                plotter.set_labels(all_labels, all_preds)
                plotter.plot(epoch=3)

        graph = ConfidenceGraph()
        graph.set_data(val_accuracies, val_confidences)
        graph.plot()

    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")
