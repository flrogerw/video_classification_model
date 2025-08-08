# Video Segment Classifier (CLIP + Metadata)

This project uses a **CLIP-based model** combined with **video metadata** to detect and classify segments in video files.  
It is designed for tasks such as automatically identifying **bumpers**, **commercials**, and other custom-defined video segments.

---

## Features

- **Frame-level video analysis** using [OpenAI CLIP](https://github.com/openai/CLIP) embeddings.
- **Metadata-aware classification** (e.g., relative position in the video).
- **Customizable target classes** (default: bumpers & commercials).
- **Configurable confidence threshold** for filtering predictions.
- **Segment grouping** to merge consecutive detections.
- **Robust error handling** with `try/except` blocks.
- Runs on **CPU** or **GPU** automatically.

---

## Requirements

- Python 3.9+
- PyTorch (with CUDA support if using GPU)
- OpenAI CLIP
- OpenCV
- Pillow

Install dependencies with:

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python pillow
```
Project Structure
bash
Copy
Edit
.
├── meta_clip_classifier.pt       # Your trained model
├── main.py                        # Main script (detection + grouping)
└── README.md                      # This file
Configuration
You can change detection settings by editing the constants at the top of the script:

```python
Copy
Edit
MODEL = "meta_clip_classifier.pt"    # Path to trained model
CLIP_MODEL = "ViT-B/32"              # CLIP variant
FPS = 0.3                            # Seconds between analyzed frames
TARGET_CLASSES = [1, 2]              # Classes to detect
CONFIDENCE_THRESHOLD = 0.7           # Min probability to count as a detection
```
Default class meanings:  0 → Normal content  1 → Bumper  2 → Commercial

Usage
Place your trained model file (meta_clip_classifier.pt) in the project folder.

Run the script with:

```bash
python main.py
```
Edit the videos list in the __main__ block to point to your own video files:

```python
Copy
Edit
videos = [
    "/path/to/video.mp4",
    "/another/video.mp4"
]
```

Processing: /path/to/video.mp4
Detections: [(12.0, 1), (13.0, 1), (40.5, 2)]

Detected bumper segments:
 - From 12.00s to 13.00s (Duration: 1.00s)

Detected commercial segments:
 - From 40.50s to 42.00s (Duration: 1.50s)
How It Works
Video Reading
Uses OpenCV to read frames at a fixed interval (FPS).

Frame Embedding
Converts each frame to RGB and passes it through the CLIP model to get an image embedding.

Metadata Encoding
Adds contextual metadata such as relative position in the video.

Classification
Passes the embedding and metadata to your trained classifier to predict class probabilities.

Filtering
Keeps only predictions with a probability above the confidence threshold.

Grouping
Groups close detections into continuous time segments.

Training Your Model
This script expects a model trained to accept:

CLIP image embeddings

Metadata vector (e.g., relative position)

You will need to train such a model separately before using this detection script.

Troubleshooting
CLIP not found: Make sure you installed it via
pip install git+https://github.com/openai/CLIP.git

CUDA out of memory: Lower FPS or run on CPU by forcing device = "cpu".

Wrong predictions: Lower CONFIDENCE_THRESHOLD to include more uncertain detections for review.

License
MIT License.
See LICENSE for details.
