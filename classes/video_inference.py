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
import ast
import os
import torch
import clip
import cv2
from PIL import Image
from collections import defaultdict
from typing import Optional, Callable, List, Tuple, Dict
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()


class VideoSegmentPredictor:
    """
    Class to load models and predict video segments using a CLIP-based classifier.
    """

    def __init__(self):
        """
        Initialize environment settings and placeholders for models.
        """
        self.clip_model_name: str = os.getenv("CLIP_MODEL", "")
        self.fps_interval: float = float(os.getenv("FPS", 0.3))
        self.confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.9))
        self.black_threshold: float = float(os.getenv("BLACK_THRESHOLD", 1))
        self.read_length: int = int(os.getenv("READ_LENGTH", 60))
        self.class_mapping = {0: "content", 1: "bumper", 2: "commercial"}

        self.clip_model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[Callable] = None
        self.classifier_model: Optional[torch.nn.Module] = None

    def load_clip_model(self, device: str) -> None:
        """
        Load the CLIP model and preprocessing pipeline.

        Args:
            device: Device string ("cuda", "mps", "cpu").

        Raises:
            RuntimeError: If CLIP loading fails.
        """
        try:
            self.clip_model, self.preprocess = clip.load(self.clip_model_name, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load CLIP model '{self.clip_model_name}': {e}")

    def load_trained_model(self, device: str, path: Optional[str]) -> None:
        """
        Load a trained classifier model from a given path.

        Args:
            device: Device string ("cuda", "mps", "cpu") to load the model onto.
            path: Path to the saved model file.

        Raises:
            RuntimeError: If loading fails.
        """
        try:
            self.classifier_model = torch.load(path, map_location=device, weights_only=False)
            self.classifier_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model from {path}: {e}")

    def predict_video_segments(
            self,
            video_path: str,
            device: str,
            target_classes: list[int] | None = None,
    ) -> List[Tuple[float, int]]:
        """
        Predict segments in a video where specific classes appear.

        Args:
            video_path: Path to the video file.
            device: Device string ("cuda", "mps", "cpu").

        Returns:
            List of (timestamp_seconds, predicted_class_id) tuples.

        Raises:
            RuntimeError: If video reading or prediction fails.
        """
        if self.clip_model is None or self.preprocess is None or self.classifier_model is None:
            raise RuntimeError("Models not loaded. Call load_clip_model and load_trained_model first.")

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps

            start_window_end = self.read_length
            end_window_start = duration - self.read_length

            timestamped_classes: List[Tuple[float, int]] = []
            frame_idx = 0

            success, frame = cap.read()
            while success:
                timestamp_sec = frame_idx / fps

                # Only process frames in the first N or last N seconds
                if timestamp_sec <= start_window_end or timestamp_sec >= end_window_start:
                    if self.is_black_frame(frame, threshold=self.black_threshold):
                        success, frame = cap.read()
                        frame_idx += 1
                        continue

                    # Convert to PIL and preprocess
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image_input = self.preprocess(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = self.clip_model.encode_image(image_input)
                        output = self.classifier_model(embedding)
                        probs = torch.softmax(output, dim=1)

                    max_prob, predicted_class = torch.max(probs, dim=1)

                    print(
                        f"{max_prob.item():.4f} :: {self.class_mapping.get(predicted_class.item())} :: Time {self.seconds_to_min_sec(frame_idx / fps)}"
                    )

                    # Store timestamp if it meets target and confidence threshold
                    if (
                            target_classes
                            and predicted_class.item() in target_classes
                            and max_prob.item() > self.confidence_threshold
                    ):
                        timestamped_classes.append((frame_idx / fps, predicted_class.item()))

                success, frame = cap.read()
                frame_idx += 1

            cap.release()
            return timestamped_classes

        except Exception as e:
            raise RuntimeError(f"Error processing video {video_path}: {e}")

    @staticmethod
    def group_segments(
            timestamped_classes: List[Tuple[float, int]],
            max_gap: float = 5.0,
    ) -> Dict[int, List[Tuple[float, float]]]:
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

        grouped_segments: Dict[int, List[Tuple[float, float]]] = {}
        for cls, timestamps in class_groups.items():
            timestamps.sort()
            segments: List[Tuple[float, float]] = []
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

    @staticmethod
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


    @staticmethod
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
