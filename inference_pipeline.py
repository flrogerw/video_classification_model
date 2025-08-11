import ast
import os
import torch
from inferences.inference_video import predict_video_segments, group_segments, seconds_to_min_sec, load_clip_model, \
    load_trained_model


device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_trained_model(device=device, path=os.getenv("INFERENCE_MODEL"))
target_classes = ast.literal_eval(os.getenv("TARGET_CLASSES"))
clip_model, preprocess = load_clip_model(device=device)

inference_videos = ["/Volumes/TTBS/time_traveler/60s/68/Batman_The_Jokers_Flying_Saucer.mp4"]  ## 5YhL0iW2kk.json

for video_path in inference_videos:
    timestamped_classes = predict_video_segments(video_path=video_path, model=model, clip_model=clip_model,
                                                 preprocess=preprocess, device=device, target_classes=target_classes)
    segments = group_segments(timestamped_classes, max_gap=5.0)

    label_map = {1: "bumper", 2: "commercial"}
    print(f"Processing: {video_path}")
    print(f"Detections: {segments}")

    for cls, segs in segments.items():
        print(f"  Detected {label_map.get(cls, 'unknown')} segments:")
        for start, end in segs:
            print(f"    - From {seconds_to_min_sec(start)} to {seconds_to_min_sec(end)}\n")
