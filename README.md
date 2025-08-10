# Video Segment Classifier (CLIP + Metadata)

This project uses a **CLIP-based model** combined with **video metadata**(optional) to detect and classify segments in video files.  It is designed for tasks such as automatically identifying **intros**, **outros** and other custom-defined video segments.

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
│   └── clip_frame_meta_classifier.py   # Metadata aware classifier head
│   └── contact_sheet.py                # Create contact sheets of frames
├── datasets                            # Location of datasets files
│   └── annotations                     # Location of annotation files
├── inferences                          # Location of inference scripts
│   └── inference_video.py              # Standard classification inference classifier
│   └── inference_video_meta.py         # Metadata aware classification inference classifier
├── models                              # Location of models
│   └── clip_classifier.pt              # Standard classification trained model
│   └── meta_clip_classifier.pt         # Metadata aware classification trained model
├── trainers                            # Location of traing scripts
│   └── trainer.py                      # Standard classification trainer
│   └── trainer_meta.py                 # Metadata aware classification trainer
├── README.md   
└── requirements.txt                    # Python requirements file                     
```

## Model Training
### Configuration
You can change detection settings by editing the constants at the top of the script:

```python
CONFUSION_MATRIX = False                # Whether to show the confusion matrix
EPOCH_COUNT = 20                        # Number of times to loop the datasets
FRAME_BUFFER = 0                        # Number of seconds on either side of an annotation timestamp
FPS = 0.3                               # How often to grab a frame for the datasets
DATA_BATCH_SIZE = 32                    # How many samples per batch to load 
LABEL_COUNT = 2                         # 0 = normal content, 1 = bumpers
MODEL = "models/clip_classifier.pt"     # What name to save the model as.
CLIP_MODEL = "ViT-B/32"                 # Which CLIP model to use ViT-B/32 or ViT-L/14
TARGET_CLASSES = [1]                    # Which classes to use in the Confusion Matrix
RETRAIN = False                         # Is this a first run or a retraining run.
BALANCE_TOLERANCE = 2.0                 # The class tolerance for the balanced datasets.
BLACK_THRESHOLD = 1.0                   # mean brightness below this = black
MOTION_THRESHOLD = 20.0                 # mean frame difference below this = low motion
```

### Annotations Format

Each annotation file is a JSON object describing the locations of bumpers, commercials, and content in a video.

#### Example
```json
{
    "file_path": "/file_to_process.mp4",
    "bumpers": [[1538.73, 1545.93]],
    "commercials": [[395.0, 397.1], [600.0, 602.5]],
    "content": [[0,120],[1407.27, 1527.27]]
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
#### Running the Classifier With or Without Metadata

You can run the frame-based classifier alone or combine it with metadata filtering.  
- **Without metadata** – Use `trainer.py` to run classification without keeping location metadata.  
- **With metadata** – Use `trainer_meta.py` to run classification with keeping location metadata.

```bash
python trainer.py           # Without metadata.
python trainer_meta.py      # with metadata.
```


## Video Inference

### Configuration
You can change detection settings by editing the constants at the top of the script:

```python
MODEL = "meta_clip_classifier.pt"       # Path to trained model
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
python inference_video_meta.py        # with Metadata
python inference_video.py             # without Metadata
```
Edit the videos list in the __main__ block to point to your own video files:

```python
inference_videos = [
    "/path/to/video.mp4",
    "/another/video.mp4"
]
```
```
Processing: /path/to/video.mp4  
Detections: [(12.0, 1), (13.0, 1)]   
Detected bumper segments:  
 - From 12.00s to 13.00s (Duration: 1.00s)  
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


