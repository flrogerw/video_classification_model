import json
import glob
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split


class SegmentMetaTrainer:
    def __init__(self):
        self.model = None

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
        features = []
        labels = []

        for file in glob.glob(f"{annotation_dir}/*.json"):
            with open(file, "r") as f:
                data = json.load(f)

            for label_name in ["bumpers", "commercials", "content"]:
                class_id = self.class_to_id(label_name)

                if label_name == "content":
                    segments = self.calculate_content_segments(data, data.get('video_duration'))
                elif label_name == "commercials":
                    segments = [[0.0, 0.0]]
                else:
                    segments = data.get(label_name) or []
                for seg in segments:
                    start_time, end_time = seg
                    seg_features = self.get_features(start_time, end_time, data.get('video_duration'))
                    features.append(seg_features)
                    labels.append(class_id)

        return np.array(features), np.array(labels)

    @staticmethod
    def get_features(start_time: float, end_time: float, video_duration: float) -> list:
        rel_start, rel_end = SegmentMetaTrainer.normalize_times(start_time, end_time, video_duration)
        rel_duration = (end_time - start_time) / video_duration
        normalized_duration = min(video_duration, 7200) / 7200
        return [normalized_duration, rel_start, rel_end, rel_duration]

    @staticmethod
    def class_to_id(label):
        mapping = {"content": 0, "bumpers": 1, "commercials": 2}
        return mapping[label]

    @staticmethod
    def normalize_times(start: float, end: float, duration: float) -> tuple[float, float]:
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        start_norm = max(0, min(1, start / duration))
        end_norm = max(0, min(1, end / duration))
        return start_norm, end_norm

    def train(self, features, labels, column_weights=None):
        """
        Train the XGBoost model with optional column weighting.

        Args:
            features (pd.DataFrame): Feature matrix with column names.
            labels (array-like): Target values.
            column_weights (dict): Optional mapping of column names to scale factors.
                                   Example: {'start_time': 5.0}
        """
        X = features.copy()

        # Apply column weighting if provided
        if column_weights:
            for col, factor in column_weights.items():
                if col in X.columns:
                    X[col] = X[col] * factor
                else:
                    raise ValueError(f"Column '{col}' not found in features")

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(X, labels, test_size=0.2, random_state=42)

        # Create DMatrix objects
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'seed': 42,
            'tree_method': 'hist'
        }

        evals = [(dtrain, 'train'), (dval, 'validation')]

        # Train the model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=evals,
            early_stopping_rounds=10,
            verbose_eval=True
        )

    def predict(self, features):
        dtest = xgb.DMatrix(features)
        return self.model.predict(dtest)

    def save_model(self, file_path: str) -> None:
        """
        Save the trained XGBoost model to the given file path.
        """
        if self.model is None:
            raise RuntimeError("No model to save. Train the model first.")
        self.model.save_model(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path: str) -> None:
        """
        Load a trained XGBoost model from the given file path.
        """
        self.model = xgb.Booster()
        self.model.load_model(file_path)
        print(f"Model loaded from {file_path}")
