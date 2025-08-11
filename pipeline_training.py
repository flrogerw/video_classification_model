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
from classes.video_classifier import ClipFrameClassifier
from classes.video_trainer import ClipClassifierTrainer
from classes.xgboost_trainer import SegmentMetaTrainer
from classes.video_annotations import VideoAnnotationGenerator

# Determine whether to retrain existing model based on environment variable
RETRAIN = os.getenv("RETRAIN", "False") == "True"

# Select device: CUDA if available, otherwise MPS (Apple Silicon) or CPU
device = "cuda" if torch.cuda.is_available() else "mps"

steps_to_run = [2, 3]

try:
    # Step 1: Generate video annotations
    if 1 in steps_to_run:
        generator = VideoAnnotationGenerator()
        generator.get_model_annotations(5)

    # Step 2: Train CLIP-based classifier
    if 2 in steps_to_run:
        clip_trainer = ClipClassifierTrainer()

        # Load CLIP model and preprocessing pipeline
        clip_model, preprocess = clip.load(os.getenv("CLIP_MODEL"), device=device)

        # If retraining, load saved model; otherwise, create a new classifier instance
        model = (
            torch.load(os.getenv("MODEL"), weights_only=False)
            if RETRAIN
            else ClipFrameClassifier(input_dim=512, num_classes=2).to(device)
        )

        # Train the model
        clip_trainer.train(device, model)

        # Save the trained model (prefix "retrained_" if retraining)
        save_model = (
            f'retrained_{os.getenv("MODEL")}' if RETRAIN else os.getenv("MODEL")
        )
        torch.save(model, save_model)

    # Step 3: Train XGBoost meta-classifier
    if 3 in steps_to_run:
        meta_trainer = SegmentMetaTrainer()
        features, labels = meta_trainer.load_annotations(os.getenv("ANNOTATIONS_DIR"))
        meta_trainer.train(features, labels)

        # Save the trained meta-classifier
        meta_trainer.save_model(os.getenv("BOOST_MODEL"))

except FileNotFoundError as fnf_err:
    print(f"File not found: {fnf_err}")
except ValueError as val_err:
    print(f"Invalid value encountered: {val_err}")
except RuntimeError as rt_err:
    print(f"Runtime error: {rt_err}")
except Exception as e:
    print(f"Unexpected error: {e}")
