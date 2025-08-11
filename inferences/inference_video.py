import os
import torch
import clip
import cv2
from PIL import Image
from collections import defaultdict
from classes.clip_classifier import ClipFrameClassifier
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load trained classifier
def load_trained_model(device, path: str | None = None):
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model


# Load CLIP
def load_clip_model(device):
    clip_model, preprocess = clip.load(os.getenv("CLIP_MODEL"), device=device)
    return clip_model, preprocess


def is_black_frame(frame, threshold=1):
    """
    Detects black or near-black frames by average pixel intensity.
    threshold: mean pixel value below which the frame is considered black.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold


# Predict segments in video and collect (timestamp, predicted_class)
def predict_video_segments(video_path, model, clip_model, preprocess, device, target_classes: None | list = None):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamped_classes = []

    frame_idx = 0
    success, frame = cap.read()
    while success:
        if frame_idx % int(fps * float(os.getenv("FPS"))) == 0:

            # ---- Skip black frames ----
            if is_black_frame(frame, threshold=1):
                success, frame = cap.read()
                frame_idx += 1
                continue

            # Convert frame to PIL Image and preprocess
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = preprocess(img).unsqueeze(0).to(device)  # shape: [1, 3, H, W]

            with torch.no_grad():
                # Get embedding from CLIP
                embedding = clip_model.encode_image(image_input)  # shape: [1, 512]
                # embedding = embedding / embedding.norm(dim=-1, keepdim=True)

                # Pass embedding to your classifier model
                output = model(embedding)  # shape: [1, num_classes]

                probs = torch.softmax(output, dim=1)

            max_prob, predicted_class = torch.max(probs, dim=1)

            print(max_prob.item(), " :: ", predicted_class.item(), " :: ", frame_idx / fps)

            if predicted_class.item() in target_classes and max_prob.item() > float(os.getenv("CONFIDENCE_THRESHOLD")):
                timestamp = frame_idx / fps
                timestamped_classes.append((timestamp, predicted_class.item()))

        success, frame = cap.read()
        frame_idx += 1

    cap.release()
    return timestamped_classes


# Group timestamps by class, then merge close timestamps into segments
def group_segments(timestamped_classes: list[tuple[float, str]], max_gap: float = 5.0) -> dict[
    str, list[tuple[float, float]]]:
    """
    Groups timestamps by class into continuous segments, ignoring any
    segments that are 1 second or less in duration.

    Args:
        timestamped_classes: List of (timestamp, class) tuples.
        max_gap: Maximum allowed gap (in seconds) between consecutive timestamps
                 to consider them part of the same segment.

    Returns:
        Dictionary mapping class labels to lists of (start_time, end_time) segments.
    """
    class_groups = defaultdict(list)

    # Group timestamps by class
    for ts, cls in timestamped_classes:
        class_groups[cls].append(ts)

    grouped_segments = {}

    for cls, timestamps in class_groups.items():
        timestamps.sort()
        segments = []
        start = timestamps[0]
        prev = timestamps[0]

        for t in timestamps[1:]:
            if t - prev > max_gap:
                if prev - start > 1.0:  # Ignore <= 1 second segments
                    segments.append((start, prev))
                start = t
            prev = t

        # Add last segment if > 1 second
        if prev - start > 1.0:
            segments.append((start, prev))

        grouped_segments[cls] = segments

    return grouped_segments


def seconds_to_min_sec(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes}:{secs:02d}"


