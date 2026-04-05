import librosa
import numpy as np
import tf_keras

model = tf_keras.models.load_model("cnn.h5", compile=False)

emotion_labels = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

def extract_feature(data, sr):

    result = np.array([])

    stft = np.abs(librosa.stft(data))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)

    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)

    result = np.hstack((mfcc, chroma, mel))

    return result

def predict_emotion_from_audio(file_path):

    try:

        data, sr = librosa.load(file_path, sr=22050)

        if len(data) < 4000 or np.mean(np.abs(data)) < 0.005:
            print("⚠️ Audio not clear or too short")
            return "UNCLEAR"

        feature = extract_feature(data, sr)

        feature = feature.reshape(1, 180, 1)

        prediction = model.predict(feature)

        predicted_index = np.argmax(prediction)

        emotion = emotion_labels[predicted_index]

        return emotion.upper()

    except Exception as e:
        print(f"❌ Error in predicting emotion: {e}")
        return "UNCLEAR"
    
if __name__ == "__main__":
    print("Model loaded successfully")