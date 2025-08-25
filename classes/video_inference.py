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
        self.clip_model_name: str = os.getenv("MODEL_CLIP_BASE", "")
        self.fps_interval: float = float(os.getenv("FPS", 0.3))
        self.confidence_threshold: float = float(os.getenv("THRESHOLD_CONFIDENCE", 0.9))
        self.black_threshold: float = float(os.getenv("THRESHOLD_BLACK", 1))
        self.read_length: int = int(os.getenv("BUFFER_READ_LENGTH", 60))
        self.class_mapping = {0: "content", 1: "bumper", 2: "commercial"}

        self.clip_model: Optional[torch.nn.Module] = None
        self.preprocess: Optional[Callable] = None
        self.classifier_model: Optional[torch.nn.Module] = None

    def trim_black_frames(self, video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = round(total_frames / fps, 2)

            # --- STEP 1: Find first non-black frame ---
            start_frame_idx = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while start_frame_idx < total_frames:
                success, frame = cap.read()
                if not success:
                    break
                if not self.is_black_frame(frame, threshold=self.black_threshold):
                    break
                start_frame_idx += 1
            start_time = start_frame_idx / fps

            # --- STEP 2: Find last non-black frame ---
            end_frame_idx = total_frames - 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            while end_frame_idx > start_frame_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_idx)
                success, frame = cap.read()
                if not success:
                    break
                if not self.is_black_frame(frame, threshold=self.black_threshold):
                    break
                end_frame_idx -= 1
            end_time = round(end_frame_idx / fps, 2)

            return max(0, round(start_time, 2)), min(end_time, duration)
        except Exception as e:
            raise RuntimeError(f"Error trimming black frames: video {video_path}: {e}")

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
            target_classes=None,
    ) -> List[Tuple[float, int, float]]:
        if target_classes is None:
            target_classes = []
        if self.clip_model is None or self.preprocess is None or self.classifier_model is None:
            raise RuntimeError("Models not loaded. Call load_clip_model and load_trained_model first.")

        CLASS_TO_EXTEND = 1  # the class that drives boundary extension
        extend_buffer_sec = 1.0  # stop only after this long without seeing CLASS_TO_EXTEND

        print(f"\nProcessing: {video_path}", flush=True)

        def infer_on_frame(frame):
            """Run model and return (pred_cls:int, conf:float) or (None, None) if black."""
            if self.is_black_frame(frame, threshold=self.black_threshold):
                return None, None
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = self.clip_model.encode_image(image_input)
                output = self.classifier_model(embedding)
                probs = torch.softmax(output, dim=1)
            max_prob, predicted_class = torch.max(probs, dim=1)
            return predicted_class.item(), max_prob.item()

        def maybe_record(ts, pred_cls, conf, sink):
            """Record only if within targets and above threshold."""
            if pred_cls is None:
                return
            if target_classes and pred_cls in target_classes and conf > self.confidence_threshold:
                sink.append((ts, pred_cls, conf))

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Unable to open video file: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise RuntimeError("Invalid FPS from video.")
            start_time, end_time = self.trim_black_frames(video_path=video_path)
            if start_time is None or end_time is None or end_time <= start_time:
                raise RuntimeError("trim_black_frames returned invalid bounds.")

            results: List[Tuple[float, int, float]] = []

            # ---------------------------
            # PASS 1: Forward from start
            # ---------------------------
            start_idx = int(start_time * fps)
            end_idx_max = int(end_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

            read_window_end = start_time + self.read_length
            last_class1_seen_at = None

            frame_idx = start_idx
            while True:
                if frame_idx > end_idx_max:
                    break
                success, frame = cap.read()
                if not success:
                    break
                ts = frame_idx / fps
                if ts > end_time:
                    break

                pred_cls, conf = infer_on_frame(frame)
                if pred_cls is not None:
                    maybe_record(ts, pred_cls, conf, results)
                    if pred_cls == CLASS_TO_EXTEND and conf > self.confidence_threshold:
                        last_class1_seen_at = ts

                # --- Stopping condition ---
                if ts >= read_window_end:
                    if last_class1_seen_at is not None and (ts - last_class1_seen_at) < extend_buffer_sec:
                        # Saw Class 1 recently → extend
                        read_window_end += extend_buffer_sec
                    elif pred_cls is None:
                        # Black frame → don’t stop yet
                        pass
                    elif pred_cls == 0:
                        # Found Class 0 → stop cleanly
                        break
                    else:
                        # No recent Class 1, not black, not Class 0 → stop
                        break

                frame_idx += 1

            cap.release()

            # ---------------------------
            # PASS 2: Backward from end
            # ---------------------------
            cap2 = cv2.VideoCapture(video_path)
            if not cap2.isOpened():
                raise RuntimeError(f"Unable to re-open video file: {video_path}")

            end_idx = int(end_time * fps)
            min_idx = int(start_time * fps)
            target_span_start_idx = max(min_idx, end_idx - int(self.read_length * fps))

            frames_since_class1 = int(1e9)  # big number: "haven't seen Class 1 yet"
            frame_idx = end_idx

            while frame_idx >= min_idx:
                cap2.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = cap2.read()
                if not success:
                    frame_idx -= 1
                    continue

                ts = frame_idx / fps
                if ts < start_time:
                    break

                pred_cls, conf = infer_on_frame(frame)
                if pred_cls is not None:
                    maybe_record(ts, pred_cls, conf, results)

                # Update Class 1 recency
                if pred_cls == CLASS_TO_EXTEND and conf is not None and conf > self.confidence_threshold:
                    frames_since_class1 = 0
                else:
                    frames_since_class1 += 1

                # --- Stopping condition ---
                if frame_idx <= target_span_start_idx:
                    if frames_since_class1 <= int(extend_buffer_sec * fps):
                        # Saw Class 1 recently -> keep going
                        pass
                    elif pred_cls is None:
                        # Black frame -> keep going until we hit a real class
                        pass
                    elif pred_cls == 0:
                        # Found real Class 0 -> safe to stop
                        break
                    else:
                        # Some other class, no recent Class 1 -> stop
                        break

                frame_idx -= 1

            cap2.release()

            # ---------------------------
            # De-duplicate & sort results
            # ---------------------------
            # Dedup by (rounded timestamp, class) to avoid double-count from overlaps.
            seen = set()
            unique = []
            for ts, cls_id, conf in sorted(results, key=lambda x: x[0]):
                key = (round(ts, 3), cls_id)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ts, cls_id, conf))

            return unique

        except Exception as e:
            raise RuntimeError(f"Error processing video {video_path}: {e}")


        except Exception as e:
            raise RuntimeError(f"Error processing video {video_path}: {e}")

    @staticmethod
    def group_segments(
            timestamped_classes: List[Tuple[float, int, float]],
            max_gap: float = 5.0,
    ) -> Dict[int, List[Tuple[float, float, float]]]:
        """
        Group timestamps by class into continuous segments, ignoring
        any segments of 1 second or less. Computes average confidence per segment.

        Args:
            timestamped_classes: List of (timestamp_seconds, class_id, confidence) tuples.
            max_gap: Max gap between consecutive timestamps (in seconds) to be grouped.

        Returns:
            Dictionary mapping class_id to list of (start_time, end_time, avg_confidence) tuples.
        """
        class_groups = defaultdict(list)

        # Group timestamps by class
        for ts, cls, conf in timestamped_classes:
            class_groups[cls].append((ts, conf))

        grouped_segments: Dict[int, List[Tuple[float, float, float]]] = {}
        for cls, values in class_groups.items():
            values.sort(key=lambda x: x[0])  # sort by timestamp
            segments: List[Tuple[float, float, float]] = []
            start = values[0][0]
            prev = values[0][0]
            confs = [values[0][1]]

            for t, conf in values[1:]:
                if t - prev > max_gap:
                    if prev - start > 1.0:  # Ignore short segments
                        avg_conf = sum(confs) / len(confs)
                        segments.append((start, prev, avg_conf))
                    start = t
                    confs = [conf]
                else:
                    confs.append(conf)
                prev = t

            # Add final segment if valid
            if prev - start > 1.0:
                avg_conf = sum(confs) / len(confs)
                segments.append((start, prev, avg_conf))

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
