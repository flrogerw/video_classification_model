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
from pathlib import Path

import torch

from classes.closeness_trainer import StartDurationClosenessTrainer
from classes.video_annotations import VideoAnnotationGenerator
from classes.video_inference import VideoSegmentPredictor
from classes.video_utils import VideoContactSheet

# Select computation device (CUDA if available, else CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

class_mapping = {0: "content", 1: "bumper", 2: "commercial"}
SHOW_ID = 97 #6 abbot 97 martian
DEV_MODE = True
MODEL_INFERENCE_CLIP = os.path.expandvars(os.getenv("MODEL_INFERENCE_CLIP"))

try:
    # Load target classes from environment variable
    target_classes = ast.literal_eval(os.getenv("TARGET_CLASSES"))

    # Step 1: Initialize and load the CLIP-based video segment predictor
    clip_predictor = VideoSegmentPredictor()
    clip_predictor.load_clip_model(device=device)
    clip_predictor.load_trained_model(
        device=device,
        path=MODEL_INFERENCE_CLIP
    )

    # Step 2: Initialize and load the closeness-classifier
    closeness = StartDurationClosenessTrainer()
    closeness.load_model(os.getenv("MODEL_CLOSENESS"))

    # Step 3: List of videos for inference
    generator = VideoAnnotationGenerator()
    inference_videos = generator.get_inference_filenames(show_id=SHOW_ID)

    for video_path in inference_videos:
        samples = []
        try:
            # Get video duration
            video_duration = save_end = generator.get_video_length(filename=video_path)
            save_start = 0

            # Predict segments for target classes
            segments = clip_predictor.predict_video_segments(
                video_path,
                device=device,
                target_classes=[1, 2]
            )

            # Group predicted frames into continuous time intervals
            grouped_segments = clip_predictor.group_segments(segments)

            # Removes beginning and end black frames
            if grouped_segments[-1] and (save_start == 0 or save_end == video_duration):
                black_outs = grouped_segments[-1]
                closer = min((0, video_duration), key=lambda v: abs(black_outs[0][0] - v))
                if closer == 0:
                    if black_outs[0][0] == 0:
                        save_start = round(black_outs[0][1], 2)
                if int(save_end) == int(black_outs[-1][1]) == int(video_duration):
                    save_end = round(black_outs[-1][0], 2)

            # Process each class's grouped segments
            for cls, segs in grouped_segments.items():
                if cls == -1:
                    continue

                for start, end, confidence in segs:

                    start_str = clip_predictor.seconds_to_min_sec(start)
                    end_str = clip_predictor.seconds_to_min_sec(end)

                    # Predict closeness for this segment
                    rel_start, rel_duration = closeness.get_features(start_time=start, end_time=end,
                                                                     black_start_end=save_start,
                                                                     black_end_start=save_end,
                                                                     # video_path=video_path,
                                                                     duration=video_duration)

                    probs = closeness.predict(start_times=[rel_start], durations=[rel_duration])

                    total_confidence = (confidence + probs[0]) / 2

                    outcome = 'TRUE' if confidence > .985 else 'FALSE'
                    print(
                        f"  {start_str}({round(start, 2)}) - {end_str}({round(end, 2)}) Confidence: {total_confidence} Video: {confidence} Closeness: {round(probs[0], 5)}  {outcome}")

                    if confidence > .985:
                        closer = min((0, video_duration), key=lambda v: abs(start - v))
                        if closer == 0:
                            save_start = round(end, 2)
                        else:
                            save_end = round(start, 2)

                        if save_start > 0:
                            samples.append((max(0.0, save_start - 2), save_start))
                        samples.append((save_start, save_start + 2))

                        samples.append((max(0.0, save_end - 2), save_end))
                        if save_end <= video_duration + 2:
                            samples.append((save_end, save_end + 2))

            if not samples:
                samples.append((0.0, 2.0))
                samples.append((video_duration - 2, video_duration))

            vcs = VideoContactSheet(video_path, cols=6)
            vcs.extract_frames_intervals(segments=samples, interval_sec=0.5)
            filename = Path(video_path).stem
            vcs.save_contact_sheet(f"contact_sheets/{filename}")

            print(f"    Final: {save_start} - {save_end}")
            if not DEV_MODE:
                generator.update_episode_start_end(save_start, save_end, os.path.basename(video_path))


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
