import os
import numpy as np
import librosa
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

DATASET_PATH = r"C:\ai training\dataset"
SAMPLE_RATE = 16000
CLIP_DURATION = 5


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=CLIP_DURATION)
    audio = librosa.util.normalize(audio)

    target_len = SAMPLE_RATE * CLIP_DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

    return np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(chroma, axis=1)
    ])


def build_and_train():
    X, y = [], []
    categories = {"real": 0, "scam": 1}

    for label_name, label_id in categories.items():
        folder = os.path.join(DATASET_PATH, label_name)
        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        for file_name in os.listdir(folder):
            if file_name.endswith((".wav", ".mp3", ".flac")):
                file_path = os.path.join(folder, file_name)
                feat = extract_features(file_path)
                X.append(feat)
                y.append(label_id)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb = XGBClassifier(eval_metric="logloss")

    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft"
    )

    ensemble.fit(X_scaled, y)

    joblib.dump(ensemble, "voice_scam_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    print("Training complete!")


if __name__ == "__main__":
    build_and_train()