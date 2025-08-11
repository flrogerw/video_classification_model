import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

from dotenv import load_dotenv
from trainers.xgboost_trainer import SegmentMetaTrainer

if __name__ == "__main__":
    # Usage example:
    #trainer = SegmentMetaTrainer()
    #features, labels = trainer.load_annotations("./sandbox/annotations")
    #trainer.train(features, labels)
    #trainer.save_model("models/segment_meta_model.json")

    #normalized_duration = min(video_duration, 7200) / 7200
    #[normalized_duration, rel_start, rel_end, rel_duration]
    """
    [0.21289814814814814, 0.9965446005305963, 1, 0.003457574044274315]
    [0.2084074074074074, 0.9967322729696109, 0.9999977785676204, 0.0032655055980095058]
    [0.3856296296296296, 0.9961450729927007, 0.9999987994621591, 0.0038537264694584157]
    """
    trainer = SegmentMetaTrainer()
    trainer.load_model("models/segment_meta_model.json")
    predictions = trainer.predict([[0.21289814814814814, 0.9965446005305963, 1, 0.003457574044274315]])
    print(predictions)
