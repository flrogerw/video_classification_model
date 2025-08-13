# Video Segment Classifier (CLIP + XGBoost)

This project uses a **CLIP-based model** combined with **XGBoost** and a custom **closeness model** to detect and classify segments in video files. It is designed for tasks such as automatically identifying intros, outros, and other custom-defined video segments.


---

## Features

- **Frame-level video analysis** using [OpenAI CLIP](https://github.com/openai/CLIP) embeddings.
- **Metadata-aware XGBoost classification** (e.g., relative position in the video).
- **Customizable target classes** (default: bumpers & commercials).
- **Configurable confidence threshold** for filtering predictions.
- **Segment grouping** to merge consecutive detections.
- **Robust error handling** with `try/except` blocks.
- Runs on **CPU**, **GPU** or **MPS** automatically.
- **Lazy Loading** of annotations for less resources required.
- **Filters Unwanted Frames** Blacks frames, redundant no motion frames.
- **Confusion Matrix** Includes a per Epoch matrix. (optional)
- **Accuracy/Confidence Graph** Line graph to plot accuracy and confidence throughout the process.
- **Configruable Dataset Balancing** Set the ratio of good to bad samples.

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
├── classes                             # Location of class files
│   ├── closeness_trainer.py            # Closeness classifier.
│   ├── video_annotations.py            # Generates annotation files.
│   ├── video_classifier.py             # Metadata aware classifier head
│   ├── video_inference.py              # Standard classification inference classifier
│   ├── video_trainer.py                # Standard classification trainer
│   ├── video_utils.py                  # Utility classes
│   └── xgboost_trainer.py              # XGBoost training class
├── datasets                            # Location of datasets files
│   └── annotations                     # Location of annotation files
├── models                              # Location of models
│   ├── clip_classifier.pt              # Standard classification trained model
│   ├── closeness_model.json            # Closeness classification trained model
│   └── segment_meta_model_lg.json      # XGBoost trained model
├── README.md 
├── pipeline_training.py                # Runs the trainng pipeline scripts
├── pipeline_inference.py               # Runs the inference pipeline scripts
└── requirements.txt                    # Python requirements file                     
```

## Model Training
### Configuration
You can change detection settings by editing the constants in the .env file:

```python
## DATABASE
DB_NAME=XXXXXX
DB_USER=XXXXXX
DB_PASSWORD=XXXXXX
DB_HOST=XXXXXX
DB_PORT=XXXXXX

## SHARED ##
# Location of where to save/access the trained model.
MODEL=models/clip_classifier.pt
# XGBoost model location
BOOST_MODEL=models/segment_meta_model.json
# How often to grab a frame for the datasets
FPS=0.3
# "ViT-B/32" or ViT-L/14
CLIP_MODEL=ViT-B/32
# Location of annotation files
ANNOTATIONS_DIR=./datasets/annotations
# Classes to process
TARGET_CLASSES=[0,1,2]
# mean brightness below this = black
BLACK_THRESHOLD=1.0

## CREATE DATASET ##
ROOT_DIR=/Volumes/TTBS/time_traveler
# Number of samples from each group (number of episodes from a show)
SAMPLE_COUNT=2
# How much content to grab
CONTENT_BUFFER=180

## TRAINER ##
# Whether to show the confusion matrix
CONFUSION_MATRIX=False
# Number of times to loop the datasets
EPOCH_COUNT=10
# Number of samples sent to model
DATA_BATCH_SIZE=32
# Number of labels we are interested in
LABEL_COUNT=2
# Is this a retrining run
RETRAIN=False
# The class tolerance for the balanced datasets.
BALANCE_TOLERANCE=8.0
# mean frame difference below this = low motion
MOTION_THRESHOLD=2.0

## INFERENCE ##
# Minimum confidence before using in calculations
CONFIDENCE_THRESHOLD=0.9
#  Model to use for video inference pipeline
INFERENCE_MODEL=models/clip_classifier.pt
```

### Annotations Format

Each annotation file is a JSON object describing the locations of bumpers, commercials, and content in a video.

#### Example
```json
{
    "file_path": "/file_to_process.mp4",
    "bumpers": [[1538.73, 1545.93]],
    "commercials": [[395.0, 397.1], [600.0, 602.5]],
    "content": [[0,120],[1407.27, 1527.27]],
    "video_duration": 1546.53
}
```

#### Field Descriptions
- **file_path** *(string)* – Path to the video file.
- **bumpers** *(array of [start, end])* – Intro/Outro segments in seconds.
- **commercials** *(array of [start, end])* – Commercial segments in seconds.
- **content** *(array of [start, end])* – Main program segments in seconds.

#### Notes
- Times are in seconds from the start of the video.
- `[start, end]` defines the start and end time of each segment.
- Any list may be empty if no segments of that type exist.
- The content values are taken from the start and end of the video  
to capture credits, which often resemble bumpers in training.

### Training

```bash
python trainer_pipeline.py
```

## Video Inference

### Configuration
You can change detection settings by editing the constants in the .env file:

```python
MODEL = "clip_classifier.pt"            # Path to trained model
CLIP_MODEL = "ViT-B/32"                 # CLIP variant
FPS = 0.3                               # Seconds between analyzed frames
TARGET_CLASSES = [1]                    # Classes to detect
CONFIDENCE_THRESHOLD = 0.9              # Min probability to count as a detection
```
#### Default class meanings: 
0 → Normal content
1 → Bumper
2 → Commercial

#### Usage
Run the script with:

```bash
python pipeline_inference.py
```
Edit the videos list in the __main__ block to point to your own video files:

```python
inference_videos = [
    "/path/to/video.mp4",
    "/another/video.mp4"
]
```
#### Outcome:
```
Processing: /path/to/video.mp4  
Detections: [(12.0, 1), (13.0, 1)]   
Detected bumper segments:  
 - From 12.00s to 13.00s  
```
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


