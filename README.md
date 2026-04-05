# SoulTalk – Emotional Wellness Chatbot

SoulTalk is a web-based emotional wellness chatbot designed to help users reflect on and manage their emotions through natural conversation. It combines **text understanding and voice-tone analysis** to detect emotional states using advanced **NLP and speech processing pipelines**, and responds empathetically with **CBT-inspired supportive messages**.

The system integrates **text emotion detection, voice emotion recognition, speech transcription, and AI-driven responses** to create a conversational companion focused on emotional awareness and mental well-being.

## Features

### Text & Voice Interaction

- Accepts both **typed and spoken inputs**.
- Detects emotional state from:
  - **Text** using DistilBERT.
  - **Voice tone** using a CNN model trained on the **RAVDESS emotional speech dataset**.
- Generates supportive and empathetic responses.

### Session Management

- Persistent chat sessions stored in **MongoDB**.
- Users can:
  - view previous conversations
  - continue earlier sessions
- Each message stores:
  - text
  - detected emotion
  - optional audio file

### AI & Emotion Fusion

SoulTalk combines multiple AI components:

| Input Type   | Processing                                     |
| ------------ | ---------------------------------------------- |
| Text         | Emotion detection using DistilBERT             |
| Voice        | Emotion recognition using CNN + audio features |
| Conversation | Context-aware reply using Cohere LLM           |

Session titles are automatically generated based on the conversation content.

## Audio Processing Pipeline

Voice messages go through two stages:

### 1. Voice Emotion Recognition

Implemented in `emotion_predictor.py`.

Pipeline:

```
Audio (.wav)
   ↓
Librosa audio loading
   ↓
Feature Extraction
   • MFCC
   • Chroma
   • Mel Spectrogram
   ↓
Feature vector (180 dimensions)
   ↓
CNN model (trained on RAVDESS dataset)
   ↓
Predicted emotion
```

Supported emotions:

```
neutral
calm
happy
sad
angry
fearful
disgust
surprised
```

The trained model is stored as:

```
cnn.h5
```

### 2. Speech-to-Text (ASR)

Voice messages are transcribed using **Whisper**.

Model used:

```
openai/whisper-base
```

Pipeline:

```
Audio file
   ↓
FFmpeg decoding
   ↓
Whisper ASR
   ↓
Transcribed text
```

The transcription is used for generating the chatbot response.

## Backend API Endpoints

| Endpoint                | Method | Description                                                     |
| ----------------------- | ------ | --------------------------------------------------------------- |
| `/api/session/new`      | POST   | Create a new chat session                                       |
| `/api/session/list`     | POST   | List all user sessions                                          |
| `/api/session/messages` | POST   | Fetch messages for a session                                    |
| `/api/message`          | POST   | Send a text message and receive AI reply                        |
| `/api/voice-message`    | POST   | Upload voice message, return emotion + transcription + AI reply |
| `/uploads/audio/`       | GET    | Serve stored audio files                                        |
| `/api/session/delete`   | POST   | Delete a session                                                |
| `/api/session/rename`   | POST   | Rename a session                                                |

## Tech Stack

| Category           | Technologies                              | Role                              |
| ------------------ | ----------------------------------------- | --------------------------------- |
| Frontend           | React.js, HTML5, CSS3, Bootstrap          | Chat UI and user interaction      |
| Backend            | Python, Flask, Flask-CORS                 | API handling and AI pipeline      |
| Database           | MongoDB                                   | Stores sessions and chat history  |
| Text Emotion       | DistilBERT                                | Detects emotion from user text    |
| Voice Emotion      | CNN model trained on RAVDESS              | Detects emotion from speech tone  |
| Speech Recognition | Whisper (HuggingFace Transformers)        | Converts speech to text           |
| LLM Response       | Cohere API (`command-xlarge-nightly`)     | Generates empathetic replies      |
| Audio Processing   | Librosa                                   | Audio feature extraction          |
| Embeddings         | SentenceTransformers (`all-MiniLM-L6-v2`) | Semantic context understanding    |
| Audio Decoding     | FFmpeg                                    | Audio format decoding for Whisper |

## Project Structure

```
SoulTalk/
│
├── backend/
│   ├── app.py
│   ├── emotion_predictor.py
│   ├── cnn.h5
│   ├── requirements.txt
│   ├── tools/
│   │   └── ffmpeg/
│   ├── uploads/
│   │   └── audio/
│   ├── .env
│   └── venv/
│
├── frontend/
│   ├── public/
│   │   ├── index.html
│   │   └── SoulTalk.png
│   ├── src/
│   │   ├── App.js
│   │   ├── App.css
│   │   ├── firebase.js
│   │   ├── index.js
│   │   └── components/
│   │       ├── ChatBox.js
│   │       ├── GoogleLoginButton.js
│   │       ├── Sidebar.js
│   │       └── Topbar.js
│   ├── package.json
│   └── package-lock.json
│
└── README.md
```

## How It Works

### 1. User Login

Users authenticate using **Google Sign-In**.

### 2. Text Message Flow

```
User message
      ↓
DistilBERT emotion classification
      ↓
Session title generation (if missing)
      ↓
Conversation history retrieval
      ↓
Cohere LLM generates empathetic response
      ↓
Message stored in MongoDB
```

### 3. Voice Message Flow

```
Voice recording
      ↓
Saved in uploads/audio
      ↓
Emotion detection (CNN model)
      ↓
Speech-to-text (Whisper ASR)
      ↓
Session title generation (if missing)
      ↓
Cohere LLM generates response
      ↓
Message stored in MongoDB
```

## Environment Variables

Create a `.env` file in the backend folder.

```
COHERE_API_KEY=your_cohere_api_key
MONGO_URI=your_mongodb_connection_uri
```

## Credits

- **Cohere API** – Conversational response generation
- **Hugging Face Transformers** – NLP models and Whisper ASR
- **RAVDESS Dataset** – Training data for voice emotion recognition
- **Librosa / NumPy / scikit-learn** – Audio feature extraction
- **React.js** – Frontend interface
- **MongoDB** – Session and message storage
