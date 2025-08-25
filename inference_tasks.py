import os
import ast
from pathlib import Path

import numpy as np
import torch

from classes.closeness_trainer import StartDurationClosenessTrainer
from classes.video_annotations import VideoAnnotationGenerator
from classes.video_inference import VideoSegmentPredictor
from classes.video_utils import VideoContactSheet

# ===== Singleton Model Loading =====
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    target_classes = ast.literal_eval(os.getenv("TARGET_CLASSES"))

    clip_predictor = VideoSegmentPredictor()
    clip_predictor.load_clip_model(device=device)
    clip_predictor.load_trained_model(
        device=device,
        path=os.getenv("MODEL_INFERENCE_CLIP")
    )

    closeness = StartDurationClosenessTrainer()
    closeness.load_model(os.getenv("MODEL_CLOSENESS"))

    generator = VideoAnnotationGenerator()

except Exception as e:
    raise RuntimeError(f"Model initialization failed: {e}")


# ===== Helper =====
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


# ===== Main Worker Function =====
def process_video(video_path: str, episode_id: int = None, dev_mode: bool = True):
    """
    Process a single video using the global models.
    Designed to be called from multiple threads safely.
    """
    try:
        # video_duration = generator.get_video_length(filename=video_path)
        save_start, save_end = clip_predictor.trim_black_frames(video_path)
        video_duration = save_end - save_start
        samples = []

        # Predict segments
        segments = clip_predictor.predict_video_segments(
            video_path,
            device=device,
            target_classes=[1, 2]
        )
        grouped_segments = clip_predictor.group_segments(segments)

        results = []

        # Process each class's grouped segments
        for cls, segs in grouped_segments.items():
            if cls == -1:
                continue
            for start, end, confidence in segs:

                # Predict closeness
                rel_start, rel_duration = closeness.get_features(
                    start_time=start,
                    end_time=end,
                    black_start_end=save_start,
                    black_end_start=save_end,
                    duration=video_duration,
                )
                probs = closeness.predict(
                    start_times=[rel_start], durations=[rel_duration]
                )

                total_confidence = (confidence + probs[0]) / 2
                outcome = "TRUE" if confidence > 0.985 else "FALSE"

                segment_result = {
                    "start": start,
                    "end": end,
                    "confidence": float(total_confidence),
                    "video_confidence": confidence,
                    "closeness_confidence": float(probs[0]),
                    "outcome": outcome,
                }

                results.append(segment_result)

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

            vcs = VideoContactSheet(video_path, matlib_show=False, cols=16)
            vcs.extract_frames_intervals(segments=samples, interval_sec=0.5)
            filename = Path(video_path).stem
            vcs.save_contact_sheet(f"contact_sheets/{filename}")

        start_str = clip_predictor.seconds_to_min_sec(save_start)
        end_str = clip_predictor.seconds_to_min_sec(save_end)

        final_result = {
            "episode_id": episode_id,
            "video": video_path,
            "final_range": (save_start, save_end),
            "time_range": (start_str, end_str),
            "segments": results
        }

        if not dev_mode:
            generator.update_episode_start_end(save_start, save_end, os.path.basename(video_path))

        return final_result

    except FileNotFoundError:
        print(f"[{video_path}] File not found")
        return {"episode_id": episode_id, "video": video_path, "error": "File not found"}
    except Exception as e:
        print(f"[{video_path}] Exception: {e}")
        return {"episode_id": episode_id, "video": video_path, "error": str(e)}
