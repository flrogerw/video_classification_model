import os
import ast

import numpy as np
import torch
from celery import states
from celery.exceptions import Ignore

from celery_app import celery_app
from classes.closeness_trainer import StartDurationClosenessTrainer
from classes.video_annotations import VideoAnnotationGenerator
from classes.video_inference import VideoSegmentPredictor

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load global models once (not per task)
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

def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    elif isinstance(obj, np.generic):  # float32, int64, etc.
        return obj.item()
    else:
        return obj

@celery_app.task(bind=True)
def process_video(self, video_path: str, episode_id: int = None, dev_mode: bool = True):
    """
    Celery task for processing a single video.
    Streams progress/results back via task.update_state.
    """
    try:
        # Video metadata
        video_duration = save_end = generator.get_video_length(filename=video_path)
        save_start = 0

        # Predict segments
        segments = clip_predictor.predict_video_segments(
            video_path,
            device=device,
            target_classes=[1, 2]
        )
        grouped_segments = clip_predictor.group_segments(segments)

        # Remove black frames at edges
        if grouped_segments[-1] and (save_start == 0 or save_end == video_duration):
            black_outs = grouped_segments[-1]
            closer = min((0, video_duration), key=lambda v: abs(black_outs[0][0] - v))
            if closer == 0 and black_outs[0][0] == 0:
                save_start = round(black_outs[0][1], 2)
            if int(save_end) == int(black_outs[-1][1]) == int(video_duration):
                save_end = round(black_outs[-1][0], 2)

        results = []

        # Process each class's grouped segments
        for cls, segs in grouped_segments.items():
            if cls == -1:
                continue

            for start, end, confidence in segs:
                start_str = clip_predictor.seconds_to_min_sec(start)
                end_str = clip_predictor.seconds_to_min_sec(end)

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

                # Stream progress back
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "video": str(video_path),
                        "segment": sanitize(segment_result),
                    },
                )

                if confidence > 0.985:
                    closer = min((0, video_duration), key=lambda v: abs(start - v))
                    if closer == 0:
                        save_start = round(end, 2)
                    else:
                        save_end = round(start, 2)

        final_result = {"video": video_path, "final_range": (save_start, save_end), "segments": results}

        if not dev_mode:
            generator.update_episode_start_end(save_start, save_end, os.path.basename(video_path))

        return final_result

    except FileNotFoundError:
        self.update_state(state=states.FAILURE, meta={"exc": "File not found"})
        raise Ignore()
    except Exception as e:
        self.update_state(state=states.FAILURE, meta={"exc": str(e)})
        raise Ignore()
