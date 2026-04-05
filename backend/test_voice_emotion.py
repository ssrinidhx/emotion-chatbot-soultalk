import sounddevice as sd
import numpy as np
import librosa
import tf_keras
import tempfile
import soundfile as sf

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

    stft = np.abs(librosa.stft(data))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)

    return np.hstack((mfcc, chroma, mel))

def predict_emotion(file_path):

    data, sr = librosa.load(file_path, sr=22050)

    feature = extract_feature(data, sr)
    feature = feature.reshape(1, 180, 1)

    prediction = model.predict(feature)

    predicted_index = np.argmax(prediction)

    return emotion_labels[predicted_index]

def record_audio(seconds=4, sr=22050):

    print("\n🎤 Speak now...")

    recording = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    sd.wait()

    recording = recording.flatten()

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp_file.name, recording, sr)

    return tmp_file.name

if __name__ == "__main__":

    print("Voice Emotion Test Started")
    print("Press Ctrl+C to stop\n")

    while True:

        input("Press ENTER to record...")

        audio_path = record_audio()

        emotion = predict_emotion(audio_path)

        print(f"\nDetected Emotion: {emotion.upper()}\n")