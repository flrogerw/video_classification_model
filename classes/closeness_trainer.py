import glob
import json
import os

import cv2
import numpy as np
import xgboost as xgb

from classes.video_classifier import VideoFrameClipDataset


class StartDurationClosenessTrainer:
    def __init__(self, neg_ratio=5, random_state=42):
        self.neg_ratio = neg_ratio
        self.random_state = random_state
        self.black_threshold = float(os.getenv('THRESHOLD_BLACK'))

        self.model = None

    def find_black_segments(self, video_path: str, n_seconds: int = 30, sample_rate: float = 0.3):
        """
        Analyze the first and last `n_seconds` of a video to detect black frame regions.

        Args:
            video_path: Path to the video file.
            n_seconds: Number of seconds from the start and end to analyze.
            sample_rate: Interval in seconds between samples (e.g., 0.5 = 2 fps).

        Returns:
            (start_black_end, end_black_start)
            start_black_end: Timestamp (in seconds) where black frames stop at the beginning.
            end_black_start: Timestamp (in seconds) where black frames start at the end.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / video_fps

        start_black_end = 0.0
        end_black_start = duration

        # --- Analyze first n_seconds ---
        start_frames = int(min(n_seconds * video_fps, frame_count))
        for i in range(0, start_frames, int(video_fps * sample_rate)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            ts = i / video_fps
            if not VideoFrameClipDataset.is_black_frame(frame, threshold=self.black_threshold):
                start_black_end = ts
                break

        # --- Analyze last n_seconds ---
        end_start_frame = max(0, frame_count - int(n_seconds * video_fps))
        for i in range(frame_count - 1, end_start_frame, -int(video_fps * sample_rate)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue
            ts = i / video_fps
            if not VideoFrameClipDataset.is_black_frame(frame, threshold=self.black_threshold):
                end_black_start = ts
                break

        cap.release()
        return round(start_black_end, 2), round(end_black_start, 2)

    def train(self, start_times, durations):
        """
        Train a closeness model on normalized start_times and durations.

        Args:
            start_times: array-like of normalized start times (0-1)
            durations: array-like of normalized durations (0-1)
        """
        rng = np.random.default_rng(self.random_state)

        # Positive samples
        pos = np.column_stack([start_times, durations])
        n_pos = len(pos)

        # Negative samples uniformly in [0,1] x [0,1]
        n_neg = int(self.neg_ratio * n_pos)
        neg = rng.uniform(0, 1, size=(n_neg, 2))

        # Combine
        X_train = np.vstack([pos, neg])
        y_train = np.hstack([np.ones(n_pos), np.zeros(n_neg)])

        dtrain = xgb.DMatrix(X_train, label=y_train)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 3,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 1.0,
            'seed': self.random_state,
            'tree_method': 'hist'
        }

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            verbose_eval=False
        )

    def predict(self, start_times, durations):
        """
        Predict closeness probability for given start_time and duration arrays.

        Returns:
            Array of probabilities in [0,1]
        """
        X = np.column_stack([start_times, durations])
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

    @staticmethod
    def calculate_content_segments(annotation: dict, video_duration: float) -> list[list[float]]:
        """
        Calculate content segments by finding gaps between bumpers and commercials.
        Ignores the 'content' property in the annotation.

        Args:
            annotation: Annotation dictionary containing 'bumpers' and 'commercials'.
            video_duration: Duration of the video in seconds.

        Returns:
            A list of [start, end] time ranges for content segments.
        """
        bumpers = annotation.get("bumpers") or []
        commercials = annotation.get("commercials") or []

        # Ensure all entries are lists of floats
        blocked_segments = []
        for seg in bumpers + commercials:
            if seg and isinstance(seg, list) and len(seg) == 2:
                blocked_segments.append([float(seg[0]), float(seg[1])])

        # Sort by start time
        blocked_segments.sort(key=lambda x: x[0])

        content_segments = []
        last_end = 0.0

        for start, end in blocked_segments:
            if start > last_end:
                content_segments.append([last_end, start])
            last_end = max(last_end, end)

        # Add final segment if there's content after the last blocked segment
        if last_end < video_duration:
            content_segments.append([last_end, video_duration])

        return content_segments

    def load_annotations(self, annotation_dir: str):
        start_times = []
        durations = []

        for file in glob.glob(f"{annotation_dir}/*.json"):
            with open(file, "r") as f:
                data = json.load(f)

            for label_name in ["bumpers", "commercials"]:
                for seg in data.get(label_name) or []:
                    start_time, end_time = seg

                    rel_start, rel_duration = self.get_features(start_time=start_time, end_time=end_time,
                                                                video_path=data.get('file_path'),
                                                                duration=data.get('video_duration'))
                    start_times.append(rel_start)
                    durations.append(rel_duration)

        return np.array(start_times), np.array(durations)

    def get_features(self, start_time: float, end_time: float, duration: float, black_start_end: float | None = None,
                     black_end_start: float | None = None, video_path: str | None = None) -> tuple:

        # Find and remove beginning/end black screens.  Foobars closeness when large amounts of black frames.
        if video_path:
            black_start_end, black_end_start = self.find_black_segments(video_path, n_seconds=30)

        closer_to = min((0, duration), key=lambda v: abs(start_time - v))
        if closer_to == 0:
            virt_start_time = start_time - black_start_end
            virt_end_time = end_time - black_start_end
        else:
            virt_start_time = start_time - (duration - black_end_start)
            virt_end_time = end_time - (duration - black_end_start)

        virt_duration = (black_end_start - black_start_end)

        rel_start, rel_end = StartDurationClosenessTrainer.normalize_times(virt_start_time, virt_end_time, virt_duration)
        rel_duration = (virt_end_time - virt_start_time) / virt_duration
        return rel_start, rel_duration

    @staticmethod
    def normalize_times(start: float, end: float, duration: float) -> tuple[float, float]:
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        start_norm = max(0, min(1, start / duration))
        end_norm = max(0, min(1, end / duration))
        return start_norm, end_norm

    def save_model(self, filepath):
        """Save the trained XGBoost model to file."""
        if self.model is not None:
            self.model.save_model(filepath)
        else:
            raise ValueError("Model has not been trained yet.")

    def load_model(self, filepath):
        """Load a saved XGBoost model from file."""
        self.model = xgb.Booster()
        self.model.load_model(filepath)
