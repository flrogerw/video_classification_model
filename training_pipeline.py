import os
import clip
import torch
from classes.clip_classifier import ClipFrameClassifier
from trainers.clip_trainer import train

RETRAIN = os.getenv("RETRAIN", "False") == "True"

device = "cuda" if torch.cuda.is_available() else "mps"
clip_model, preprocess = clip.load(os.getenv("CLIP_MODEL"), device=device)
model = torch.load(os.getenv("MODEL"), weights_only=False) if RETRAIN else ClipFrameClassifier(
    input_dim=512, num_classes=2).to(device)

train(device, model)
save_model = f'retrained_{os.getenv("MODEL")}' if RETRAIN else os.getenv("MODEL")
torch.save(model, save_model)