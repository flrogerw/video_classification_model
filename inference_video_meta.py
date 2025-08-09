import torch
import clip
import cv2
from PIL import Image
from collections import defaultdict
from typing import Optional, Tuple

MODEL = "meta_clip_classifier.pt"
CLIP_MODEL = "ViT-L/14" #"ViT-B/32"
FPS = 0.3
TARGET_CLASSES = [1, 2]  # 0 = normal content, 1 = bumper, 2 = commercial
CONFIDENCE_THRESHOLD = 0.4

# Detect device
device = "cuda" if torch.cuda.is_available() else "mps"


def load_trained_model(path: str = MODEL) -> torch.nn.Module:
    """
    Load a trained PyTorch model from disk.

    Args:
        path (str): Path to the saved model file.

    Returns:
        torch.nn.Module: The loaded model in evaluation mode.
    """
    try:
        model = torch.load(path, map_location=device, weights_only=False)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from '{path}': {e}")


def load_clip_model() -> Tuple[torch.nn.Module, object]:
    """
    Load a CLIP model and its preprocessing pipeline.

    Returns:
        tuple: (clip_model, preprocess_function)
    """
    try:
        clip_model, preprocess = clip.load(CLIP_MODEL, device=device)
        return clip_model, preprocess
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model '{CLIP_MODEL}': {e}")


def normalize_metadata(meta_dict: dict[str, float]) -> torch.Tensor:
    """
    Normalize metadata values into a tensor.

    Args:
        meta_dict (dict): dictionary containing metadata features.

    Returns:
        torch.Tensor: 1D tensor of metadata features.
    """
    return torch.tensor(
        [meta_dict.get("relative_position", 0.0)],
        dtype=torch.float32
    )


def predict_video_segments(
    video_path: str,
    model: torch.nn.Module,
    clip_model: torch.nn.Module,
    preprocess,
    target_classes: Optional[list[int]] = None
) -> list[Tuple[float, int]]:
    """
    Predicts segments in a video that belong to specific target classes.

    Args:
        video_path (str): Path to the video file.
        model (torch.nn.Module): Trained classification model.
        clip_model (torch.nn.Module): CLIP model for frame embedding.
        preprocess (callable): CLIP preprocessing function.
        target_classes (list[int], optional): list of class IDs to detect.

    Returns:
        list[tuple]: list of (timestamp, class_id) detections.
    """
    if target_classes is None:
        target_classes = TARGET_CLASSES

    timestamped_classes = []

    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps if fps > 0 else 0
    except Exception as e:
        raise RuntimeError(f"Failed to open video '{video_path}': {e}")

    frame_idx = 0
    success, frame = cap.read()

    while success:
        # Process frames at fixed intervals
        if fps > 0 and frame_idx % int(fps * FPS) == 0:
            timestamp = frame_idx / fps

            try:
                # 1. Image embedding using CLIP
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_input = preprocess(img).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    embedding = clip_model.encode_image(image_input)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                # 2. Metadata vector
                meta_dict = {
                    "relative_position": timestamp / duration if duration > 0 else 0.0
                }
                meta_tensor = normalize_metadata(meta_dict).unsqueeze(0).to(device).float()

                # 3. Model prediction
                with torch.no_grad():
                    output = model(embedding, meta_tensor)
                    probs = torch.softmax(output, dim=1)

                max_prob, predicted_class = torch.max(probs, dim=1)
                print(max_prob.item(), " :: ", predicted_class.item(), " :: ", frame_idx / fps)

                # Save only confident predictions in target classes
                if predicted_class.item() in target_classes and max_prob.item() > CONFIDENCE_THRESHOLD:
                    timestamped_classes.append((timestamp, predicted_class.item()))
            except Exception as e:
                print(f"Frame {frame_idx}: prediction error - {e}")

        success, frame = cap.read()
        frame_idx += 1

    cap.release()
    return timestamped_classes


def group_segments(timestamped_classes: list[Tuple[float, int]], max_gap: float = 2.0) -> dict[int, list[dict[str, float]]]:
    """
    Groups timestamps into contiguous segments for each class.

    Args:
        timestamped_classes (list[tuple]): list of (timestamp, class_id) detections.
        max_gap (float): Max time gap (seconds) between consecutive timestamps to group.

    Returns:
        dict: {class_id: [ {"start": float, "end": float, "duration": float}, ... ]}
    """
    class_groups = defaultdict(list)
    for ts, cls in timestamped_classes:
        class_groups[cls].append(ts)

    grouped_segments = {}

    for cls, timestamps in class_groups.items():
        if not timestamps:
            continue

        timestamps.sort()
        segments = []
        start = timestamps[0]
        prev = timestamps[0]

        for t in timestamps[1:]:
            if t - prev > max_gap:
                duration = prev - start
                segments.append({
                    "start": start,
                    "end": prev,
                    "duration": duration
                })
                start = t
            prev = t

        # Add the last segment
        duration = prev - start
        segments.append({
            "start": start,
            "end": prev,
            "duration": duration
        })

        grouped_segments[cls] = segments

    return grouped_segments


if __name__ == "__main__":
    try:
        model = load_trained_model(MODEL)
        clip_model, preprocess = load_clip_model()

        videos = [
            "/Volumes/TTBS/time_traveler/60s/63/My_Favorite_Martian_Rocket_to_Mars.mp4"
        ]

        for video_path in videos:
            print(f"\nProcessing: {video_path}")
            timestamped_classes = predict_video_segments(video_path, model, clip_model, preprocess, TARGET_CLASSES)
            print("Detections:", timestamped_classes)

            segments = group_segments(timestamped_classes, max_gap=2.0)
            label_map = {1: "bumper", 2: "commercial"}
            print(segments)
            for cls, segs in segments.items():
                print(f"\nDetected {label_map.get(cls, 'unknown')} segments:")
                for seg in segs:
                    print(f" - From {seg['start']:.2f}s to {seg['end']:.2f}s (Duration: {seg['duration']:.2f}s)")

    except Exception as e:
        print(f"Fatal error: {e}")
