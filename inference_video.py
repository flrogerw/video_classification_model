import torch
import clip
import cv2
from PIL import Image
from torch import nn
from collections import defaultdict
from contact_sheet import VideoContactSheet

MODEL = "models/clip_classifier.pt"
CLIP_MODEL = "ViT-B/32" #"ViT-B/32" # ViT-L/14
FPS = 0.3
TARGET_CLASSES = [1]  # 0 = normal content, 1 = bumper
CONFIDENCE_THRESHOLD = 0.4 # Minimum confidence before using in calculations

device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------
# Simple Classifier Head
# ------------------------------
class ClipFrameClassifier(nn.Module):
    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


# Load trained classifier
def load_trained_model(path=MODEL):
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    return model


# Load CLIP
def load_clip_model():
    clip_model, preprocess = clip.load(CLIP_MODEL, device=device)
    return clip_model, preprocess

def is_black_frame(frame, threshold=1):
    """
    Detects black or near-black frames by average pixel intensity.
    threshold: mean pixel value below which the frame is considered black.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray.mean() < threshold

# Predict segments in video and collect (timestamp, predicted_class)
def predict_video_segments(video_path, model, clip_model, preprocess, target_classes=TARGET_CLASSES):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamped_classes = []

    frame_idx = 0
    success, frame = cap.read()
    while success:
        if frame_idx % int(fps * FPS) == 0:

            # ---- Skip black frames ----
            if is_black_frame(frame, threshold=10):
                success, frame = cap.read()
                frame_idx += 1
                continue

            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = clip_model.encode_image(image_input)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                output = model(embedding)
                probs = torch.softmax(output, dim=1)

            max_prob, predicted_class = torch.max(probs, dim=1)

            print(max_prob.item(), " :: ", predicted_class.item(), " :: ", frame_idx / fps)

            if predicted_class.item() in target_classes and max_prob.item() > CONFIDENCE_THRESHOLD:
                timestamp = frame_idx / fps
                timestamped_classes.append((timestamp, predicted_class.item()))

        success, frame = cap.read()
        frame_idx += 1

    cap.release()
    return timestamped_classes


# Group timestamps by class, then merge close timestamps into segments
def group_segments(timestamped_classes, max_gap=2.0):
    class_groups = defaultdict(list)

    # Separate timestamps by class
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
                segments.append((start, prev))
                start = t
            prev = t

        segments.append((start, prev))
        grouped_segments[cls] = segments

    return grouped_segments


if __name__ == "__main__":
    model = load_trained_model(MODEL)
    clip_model, preprocess = load_clip_model()

    videos = ["/Volumes/TTBS/time_traveler/60s/63/Petticoat_Junction_Spur_Line_to_Shady_Rest.mp4"] ## 5YhL0iW2kk.json

    for video_path in videos:
        timestamped_classes = predict_video_segments(video_path, model, clip_model, preprocess)

        segments = group_segments(timestamped_classes, max_gap=2.0)

        label_map = {1: "bumper", 2: "commercial"}
        print(video_path)
        for cls, segs in segments.items():
            print(f"\nDetected {label_map.get(cls, 'unknown')} segments:")
            for start, end in segs:
                print(f" - From {start:.2f}s to {end:.2f}s")
