"""
Main Training Pipeline

This script orchestrates the video classification training process. It:
1. Generates video annotations from a database and video files.
2. Loads and trains a CLIP-based frame classifier.
3. Trains a secondary XGBoost meta-classifier using the generated annotations.

Configuration:
    - All required paths, model names, and flags are loaded from environment variables.
    - Requires `.env` file to be configured.

Modules:
    - classes.video_annotations.VideoAnnotationGenerator
    - classes.video_trainer.ClipClassifierTrainer
    - classes.video_classifier.ClipFrameClassifier
    - classes.xgboost_trainer.SegmentMetaTrainer
"""

import os
import clip
import torch

from classes.closeness_trainer import StartDurationClosenessTrainer
from classes.video_classifier import ClipFrameClassifier
from classes.video_trainer import ClipClassifierTrainer
from classes.video_annotations import VideoAnnotationGenerator

# Determine whether to retrain existing model based on environment variable
RETRAIN = os.getenv("BOOL_RETRAIN", "False") == "True"
MODEL_DIMENSION = int(os.getenv('MODEL_CLIP_DIMENSION'))

# Select device: CUDA if available, otherwise MPS (Apple Silicon) or CPU
device = "cuda" if torch.cuda.is_available() else "mps"

# Define what steps in the pipeline to preform.
# Step 1: Generate video annotations
# Step 2: Train CLIP-based classifier
# Step 3: Train XGBoost meta and closeness classifiers
steps_to_run = [2]

try:
    # Step 1: Generate video annotations
    if 1 in steps_to_run:
        generator = VideoAnnotationGenerator()
        generator.get_model_annotations()

    # Step 2: Train CLIP-based classifier
    if 2 in steps_to_run:
        clip_trainer = ClipClassifierTrainer()

        # Load CLIP model and preprocessing pipeline
        clip_model, preprocess = clip.load(os.getenv("MODEL_CLIP_BASE"), device=device)

        # If retraining, load saved model; otherwise, create a new classifier instance
        model = (
            torch.load(os.path.expandvars(os.getenv("MODEL_CLIP")), weights_only=False)
            if RETRAIN
            else ClipFrameClassifier(input_dim=MODEL_DIMENSION, num_classes=2).to(device)
        )

        # Train the model
        clip_trainer.train(device, model)

        # Save the trained model (prefix "retrained_" if retraining)
        save_model = (
            os.path.expandvars(os.getenv("MODEL_RETRAIN")) if RETRAIN else os.path.expandvars(os.getenv("MODEL_CLIP"))
        )
        torch.save(model, save_model)

    # Step 3: Train XGBoost meta-classifier and closeness-classifier
    if 3 in steps_to_run:
        # Load the classifier trainers
        closeness = StartDurationClosenessTrainer()

        # Create feature sets
        start_times, durations = closeness.load_annotations(os.getenv("DIR_ANNOTATIONS"))
        # start_times, durations = closeness.load_annotations(os.getenv("DIR_ANNOTATIONS_ALL"))

        # Train the classifiers
        closeness.train(start_times, durations)

        # Save the trained classifiers
        closeness.save_model(os.getenv("MODEL_CLOSENESS"))

except FileNotFoundError as fnf_err:
    print(f"File not found: {fnf_err}")
except ValueError as val_err:
    print(f"Invalid value encountered: {val_err}")
except RuntimeError as rt_err:
    print(f"Runtime error: {rt_err}")
except Exception as e:
    print(f"Unexpected error: {e}")
