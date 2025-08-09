# Video Segment Classifier (CLIP + Metadata)

This project uses a **CLIP-based model** combined with **video metadata**(optional) to detect and classify segments in video files.  
It is designed for tasks such as automatically identifying **intros/outros**, **commercials**, and other custom-defined video segments.

---

## Features

- **Frame-level video analysis** using [OpenAI CLIP](https://github.com/openai/CLIP) embeddings.
- **Metadata-aware classification** (e.g., relative position in the video).
- **Customizable target classes** (default: bumpers & commercials).
- **Configurable confidence threshold** for filtering predictions.
- **Segment grouping** to merge consecutive detections.
- **Robust error handling** with `try/except` blocks.
- Runs on **CPU**, **GPU** or **MPS** automatically.
- **Lazy Loading** of annotations for less resources required.

---

## Requirements

- Python 3.11+
- PyTorch (with CUDA support if using GPU)
- OpenAI CLIP
- OpenCV
- Pillow

Install dependencies with:

```bash
pip install -r requirements.txt
```
Project Structure
```bash
├── datasets
   └── annotations                # Location of annotation files
├── meta_clip_classifier.pt       # The trained model
├── inference_video.py            # Main script (detection + grouping)
└── README.md                     
```
### Configuration
You can change detection settings by editing the constants at the top of the script:

```python
MODEL = "meta_clip_classifier.pt"    # Path to trained model
CLIP_MODEL = "ViT-B/32"              # CLIP variant
FPS = 0.3                            # Seconds between analyzed frames
TARGET_CLASSES = [1, 2]              # Classes to detect
CONFIDENCE_THRESHOLD = 0.4           # Min probability to count as a detection
```
#### Default class meanings:  
0 → Normal content  1 → Bumper  2 → Commercial

#### Usage  
Place the trained model file (meta_clip_classifier.pt) in the project folder.

Run the script with:

```bash
python inference_video_meta.py     # with Metadata
python inference_video.py          # without Metadata
```
Edit the videos list in the __main__ block to point to your own video files:

```python
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

### How It Works
**Video Reading**  
Uses OpenCV to read frames at a fixed interval (FPS).

**Frame Embedding**  
Converts each frame to RGB and passes it through the CLIP model to get an image embedding.

**Metadata Encoding**  
Adds contextual metadata such as relative position in the video.

**Classification**  
Passes the embedding and metadata to the trained classifier to predict class probabilities.

**Filtering**  
Keeps only predictions with a probability above the confidence threshold.

**Grouping**  
Groups close detections into continuous time segments.

**Training the Model**  
This script expects a model trained to accept:

- CLIP image embeddings
- Metadata vector (e.g., relative position)

You will need to train such a model separately before using this detection script.

### Troubleshooting  
**CLIP not found:**  
Make sure you installed it via pip install git+https://github.com/openai/CLIP.git

**CUDA out of memory:**   
Lower FPS or run on CPU by forcing device = "cpu".

**Wrong predictions:**  
Lower CONFIDENCE_THRESHOLD to include more uncertain detections for review.


