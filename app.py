import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import pickle

# Load model
with open("emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature extraction function
def extract_features(audio, sr=22050):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])

# Audio recording
def record_audio(duration=3, sr=22050):
    st.info("🎙️ Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    st.success("✅ Recording complete")
    return recording.flatten(), sr

# Streamlit UI
st.title("🎧 Voice Emotion Detector")
st.write("Record your voice or upload a WAV file to predict the emotion.")

# Mic Input
if st.button("Start Recording"):
    audio, sr = record_audio()
    features = extract_features(audio, sr)
    prediction = model.predict([features])[0]
    st.markdown(f"### 🧠 Predicted Emotion (Mic): `{prediction.capitalize()}`")

    # Show prediction confidence
    probs = model.predict_proba([features])[0]
    st.subheader("🔍 Prediction Probabilities (Mic):")
    for emotion, prob in zip(model.classes_, probs):
        st.write(f"{emotion.capitalize()}: {prob:.2f}")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a .wav audio file", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    audio_data, sr = librosa.load(uploaded_file, sr=22050)
    features = extract_features(audio_data, sr)
    prediction = model.predict([features])[0]
    st.markdown(f"### 🧠 Predicted Emotion (File): `{prediction.capitalize()}`")

    # Show prediction confidence
    probs = model.predict_proba([features])[0]
    st.subheader("🔍 Prediction Probabilities (File):")
    for emotion, prob in zip(model.classes_, probs):
        st.write(f"{emotion.capitalize()}: {prob:.2f}")
