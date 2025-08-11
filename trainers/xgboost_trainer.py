import json
import glob

import cv2
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split



class SegmentMetaTrainer:
    def __init__(self):
        self.model = None

    def load_annotations(self, annotation_dir: str):
        features = []
        labels = []

        for file in glob.glob(f"{annotation_dir}/*.json"):
            with open(file, "r") as f:
                data = json.load(f)

            video_duration = self.get_video_duration(data.get("file_path"))
            for label_name in ["bumpers", "commercials"]:
                class_id = self.class_to_id(label_name)
                for seg in (data.get(label_name) or []):
                    start_time, end_time = seg
                    rel_start, rel_end = self.normalize_times(start_time, end_time, video_duration)
                    rel_duration = (end_time - start_time) / video_duration
                    normalized_duration = min(video_duration, 7200) / 7200
                    features.append([normalized_duration, rel_start, rel_end, rel_duration])
                    print([normalized_duration, rel_start, rel_end, rel_duration])
                    labels.append(class_id)

        return np.array(features), np.array(labels)

    @staticmethod
    def class_to_id(label):
        mapping = {"bumpers": 1, "commercials": 2, "content": 0}
        return mapping[label]

    @staticmethod
    def get_video_duration(video_file: str) -> float:
        # Get video properties
        try:
            cap = cv2.VideoCapture(video_file)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                raise ValueError("Invalid FPS value")
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
            cap.release()
            return duration
        except Exception as e:
            print(f"Error probing video {video_file}: {e}")
            return 0.0

    @staticmethod
    def normalize_times(start: float, end: float, duration: float)-> tuple[float, float]:
        if duration <= 0:
            raise ValueError("Duration must be greater than 0")

        start_norm = max(0, min(1, start / duration))
        end_norm = max(0, min(1, end / duration))
        return start_norm, end_norm

    def train(self, features, labels):
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        params = {
            'objective': 'multi:softmax',
            'num_class': 3,
            'eval_metric': 'mlogloss',
            'max_depth': 4,
            'eta': 0.1,
            'seed': 42
        }

        evals = [(dtrain, 'train'), (dval, 'validation')]
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