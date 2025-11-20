import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

MODEL_NAME = "superb/wav2vec2-base-superb-er"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
id2label = model.config.id2label
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(_device)

def predict_emotion_from_audio(file_path):
    try:
        speech, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(speech) < 4000 or np.mean(np.abs(speech)) < 0.005:
            print("⚠️ Audio not clear or too short")
            return "UNCLEAR"
        inputs = feature_extractor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(_device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        emotion = id2label[predicted_id]
        return emotion.upper()
    except Exception as e:
        print(f"❌ Error in predicting emotion: {e}")
        return "UNCLEAR"
