📘 Voice Emotion Detection from Audio — Training Script with Explanations
📦 Step 1: Install Required Libraries
These libraries are used for audio processing (librosa, soundfile, resampy), machine learning (scikit-learn), and progress tracking (tqdm).

python
Copy
Edit
!pip install librosa soundfile resampy scikit-learn tqdm
💽 Step 2: Mount Google Drive
We mount Google Drive to access the emotion_detection.zip file stored there.

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
📂 Step 3: Extract the Zip File
We extract the dataset emotion_detection.zip into a folder called /content/emotion_detection.

python
Copy
Edit
import zipfile
import os

zip_path = '/content/drive/MyDrive/emotion_detection.zip'  # Path in Google Drive
extract_path = "/content/emotion_detection"

os.makedirs(extract_path, exist_ok=True)

try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Extraction complete")
except FileNotFoundError:
    print(f"❌ Error: The file {zip_path} was not found.")
except Exception as e:
    print(f"❌ An error occurred during extraction: {e}")
🔤 Step 4: Emotion Mapping
CREMA-D files use short codes for emotions. This dictionary helps map them to full emotion labels.

python
Copy
Edit
emotion_map = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}
🧠 Step 5: Feature Extraction Function
We extract 3 sets of audio features:

MFCCs for tone quality

Chroma for pitch-based tone

Mel spectrogram for power across frequencies

python
Copy
Edit
import librosa
import numpy as np

def extract_features(file_path):
    X, sr = librosa.load(file_path, res_type='kaiser_fast', duration=3)
    mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=X, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel])
🔄 Step 6: Load and Process the Dataset
We go through all .wav files in the dataset, extract features, and assign their correct emotion label.

python
Copy
Edit
from tqdm import tqdm

X, y = [], []

for root, _, files in os.walk(extract_path):
    for file in tqdm(files):
        if file.endswith(".wav"):
            try:
                file_path = os.path.join(root, file)
                emotion_code = file.split("_")[2]  # e.g. "HAP" from "1001_TIE_HAP_XX.wav"
                emotion = emotion_map.get(emotion_code)
                if emotion:
                    features = extract_features(file_path)
                    X.append(features)
                    y.append(emotion)
            except Exception as e:
                print(f"Error with file {file}: {e}")
🧪 Step 7: Train the Model
Split the data into training and testing, train a Random Forest classifier, and print the performance.

python
Copy
Edit
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📄 Classification Report:\n", classification_report(y_test, y_pred))
💾 Step 8: Save the Model
Save the trained model to a .pkl file, so it can be reused in the Streamlit app.

python
Copy
Edit
import pickle

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as emotion_model.pkl")
📥 Step 9: Download the Model Locally
This allows you to download emotion_model.pkl to your local machine for later use.

python
Copy
Edit
from google.colab import files
files.download("emotion_model.pkl")
