"""
Video Segment Prediction and Classification Script

This script:
1. Loads a trained CLIP-based video segment classifier.
2. Predicts video segments for specific target classes.
3. Groups predicted segments into continuous time intervals.
4. Uses a meta-classifier (XGBoost) to classify each segment.

Configuration:
    - Model paths and target classes are loaded from environment variables.
    - Requires `.env` file with configuration values.

Modules:
    - classes.video_inference.VideoSegmentPredictor
    - classes.xgboost_trainer.SegmentMetaTrainer
"""

import ast
import os
import torch

from classes.closeness_trainer import StartDurationClosenessTrainer
from classes.video_annotations import VideoAnnotationGenerator
from classes.video_inference import VideoSegmentPredictor
from classes.xgboost_trainer import SegmentMetaTrainer

# Select computation device (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class_mapping = {0: "content", 1: "bumper", 2: "commercial"}

try:
    # Load target classes from environment variable
    target_classes = ast.literal_eval(os.getenv("TARGET_CLASSES"))

    # Step 1: Initialize and load the CLIP-based video segment predictor
    clip_predictor = VideoSegmentPredictor()
    clip_predictor.load_clip_model(device=device)
    clip_predictor.load_trained_model(
        device=device,
        path=os.getenv("INFERENCE_CLIP")
    )

    # Step 2: Initialize and load the meta-classifier
    meta_predictor = SegmentMetaTrainer()
    meta_predictor.load_model(os.getenv("INFERENCE_BOOST"))

    # Step 2.5: Initialize and load the closeness-classifier
    closeness = StartDurationClosenessTrainer()
    closeness.load_model(os.getenv("INFERENCE_CLOSENESS"))


    # Step 3: List of videos for inference
    inference_videos = [
        "/Volumes/TTBS/time_traveler/90s/92/Baywatch_Point_Doom.mp4"
    ]

    for video_path in inference_videos:
        try:
            # Get video duration
            video_duration = VideoAnnotationGenerator.get_video_length(filename=video_path)

            # Predict segments for target classes
            segments = clip_predictor.predict_video_segments(
                video_path,
                device=device,
                target_classes=[1,2]
            )

            # Group predicted frames into continuous time intervals
            grouped_segments = clip_predictor.group_segments(segments)

            # Process each class's grouped segments
            for cls, segs in grouped_segments.items():
                print(f"{class_mapping.get(cls)}:")
                for start, end in segs:
                    start_str = clip_predictor.seconds_to_min_sec(start)
                    end_str = clip_predictor.seconds_to_min_sec(end)
                    print(f"  {start_str} - {end_str}")

                    # Extract features for the meta-classifier
                    seg_features = meta_predictor.get_features(
                        start_time=start,
                        end_time=end,
                        video_duration=video_duration
                    )

                    # Predict label for this segment
                    predictions = meta_predictor.predict([seg_features])
                    print(f"  Meta-classifier prediction: {predictions}")

                    # Predict closeness for this segment
                    rel_start, rel_duration = closeness.get_features(start, end, video_duration)
                    probs = closeness.predict(start_times=[rel_start], durations=[rel_duration])
                    print(f"  Closeness-classifier prediction: {probs}")

        except FileNotFoundError:
            print(f"Video file not found: {video_path}")
        except Exception as vid_err:
            print(f"Error processing {video_path}: {vid_err}")

except ValueError as val_err:
    print(f"Invalid configuration value: {val_err}")
except FileNotFoundError as fnf_err:
    print(f"Model file not found: {fnf_err}")
except RuntimeError as rt_err:
    print(f"Runtime error: {rt_err}")
except Exception as e:
    print(f"Unexpected error: {e}")
