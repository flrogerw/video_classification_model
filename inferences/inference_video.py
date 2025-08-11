"""
Video Segment Prediction Module

This module loads a trained CLIP-based classifier model and uses it to detect
specific segments in videos. It can:
    - Load CLIP and classifier models
    - Detect black frames
    - Extract frame embeddings with CLIP
    - Predict classes for video frames
    - Group predictions into continuous segments
    - Convert time formats

Environment variables are read from `.env` using `python-dotenv`.
"""

import os
import torch
import clip
import cv2
from PIL import Image
from collections import defaultdict
from classes.clip_classifier import ClipFrameClassifier
from dotenv import load_dotenv
from typing import Optional, Callable

# Load environment variables
load_dotenv()

# Environment settings (loaded once)
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL")
FPS_INTERVAL = float(os.getenv("FPS", 0.3))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.9))


def load_trained_model(device: str, path: Optional[str] = None) -> torch.nn.Module:
    """
    Load a trained classifier model from a given path.

    Args:
        device: Device string ("cuda", "mps", "cpu") to load the model onto.
        path: Optional path to the saved model file.

    Returns:
        The loaded and evaluation-ready model.

    Raises:
        RuntimeError: If loading fails.
    """
    try:
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load trained model from {path}: {e}")


def load_clip_model(device: str) -> tuple[torch.nn.Module, Callable]:
    """
    Load the CLIP model and preprocessing pipeline.

    Args:
        device: Device string ("cuda", "mps", "cpu").

    Returns:
        Tuple of (clip_model, preprocess_function).

    Raises:
        RuntimeError: If CLIP loading fails.
    """
    try:
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        return clip_model, preprocess
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model '{CLIP_MODEL_NAME}': {e}")


def is_black_frame(frame, threshold: float = 1) -> bool:
    """
    Detect black or near-black frames by average pixel intensity.

    Args:
        frame: A single video frame (NumPy array).
        threshold: Mean pixel value below which the frame is considered black.

    Returns:
        True if frame is black, otherwise False.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold


def predict_video_segments(
    video_path: str,
    model: torch.nn.Module,
    clip_model: torch.nn.Module,
    preprocess: Callable,
    device: str,
    target_classes: Optional[list[int]] = None
) -> list[tuple[float, int]]:
    """
    Predict segments in a video where specific classes appear.

    Args:
        video_path: Path to the video file.
        model: Trained classifier model.
        clip_model: Pretrained CLIP model.
        preprocess: CLIP preprocessing function.
        device: Device string ("cuda", "mps", "cpu").
        target_classes: List of class IDs to track.

    Returns:
        List of (timestamp_seconds, predicted_class_id) tuples.

    Raises:
        RuntimeError: If video reading or prediction fails.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamped_classes = []
        frame_idx = 0

        success, frame = cap.read()
        while success:
            # Process frame at defined interval
            if frame_idx % int(fps * FPS_INTERVAL) == 0:
                # Skip black frames
                if is_black_frame(frame, threshold=1):
                    success, frame = cap.read()
                    frame_idx += 1
                    continue

                # Convert to PIL and preprocess
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_input = preprocess(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    embedding = clip_model.encode_image(image_input)
                    output = model(embedding)
                    probs = torch.softmax(output, dim=1)

                max_prob, predicted_class = torch.max(probs, dim=1)

                print(
                    f"{max_prob.item():.4f} :: Class {predicted_class.item()} :: Time {frame_idx / fps:.2f}s"
                )

                # Store timestamp if it meets target and confidence threshold
                if (
                    target_classes
                    and predicted_class.item() in target_classes
                    and max_prob.item() > CONFIDENCE_THRESHOLD
                ):
                    timestamped_classes.append((frame_idx / fps, predicted_class.item()))

            success, frame = cap.read()
            frame_idx += 1

        cap.release()
        return timestamped_classes

    except Exception as e:
        raise RuntimeError(f"Error processing video {video_path}: {e}")


def group_segments(
    timestamped_classes: list[tuple[float, int]],
    max_gap: float = 5.0
) -> dict[int, list[tuple[float, float]]]:
    """
    Group timestamps by class into continuous segments, ignoring
    any segments of 1 second or less.

    Args:
        timestamped_classes: List of (timestamp_seconds, class_id) tuples.
        max_gap: Max gap between consecutive timestamps (in seconds) to be grouped.

    Returns:
        Dictionary mapping class_id to list of (start_time, end_time) tuples.
    """
    class_groups = defaultdict(list)

    # Group timestamps by class
    for ts, cls in timestamped_classes:
        class_groups[cls].append(ts)

    grouped_segments = {}
    for cls, timestamps in class_groups.items():
        timestamps.sort()
        segments = []
        start = timestamps[0]
        prev = timestamps[0]

        for t in timestamps[1:]:
            if t - prev > max_gap:
                if prev - start > 1.0:  # Ignore short segments
                    segments.append((start, prev))
                start = t
            prev = t

        # Add final segment if valid
        if prev - start > 1.0:
            segments.append((start, prev))

        grouped_segments[cls] = segments

    return grouped_segments


def seconds_to_min_sec(seconds: float) -> str:
    """
    Convert seconds to MM:SS format.

    Args:
        seconds: Time in seconds.

    Returns:
        String formatted as "M:SS".
    """
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes}:{secs:02d}"
