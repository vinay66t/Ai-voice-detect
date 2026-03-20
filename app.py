from flask import Flask, render_template, request, jsonify
import os
import joblib
import librosa
import numpy as np
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model + scaler
model = joblib.load("voice_scam_model.pkl")
scaler = joblib.load("scaler.pkl")

SAMPLE_RATE = 16000
CLIP_DURATION = 5


# ==============================
# FEATURE EXTRACTION
# ==============================
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


# ==============================
# ROUTES
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["audio"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
        features_scaled = scaler.transform(features.reshape(1, -1))
        proba = model.predict_proba(features_scaled)[0]
        ai_conf = float(proba[1])

        if ai_conf > 0.80:
            result = "🚨 High Risk: AI Deepfake"
            explanation = "Synthetic frequency patterns detected."
        elif ai_conf > 0.55:
            result = "⚠️ Suspicious"
            explanation = "Minor spectral anomalies detected."
        else:
            result = "✅ Authentic"
            explanation = "Natural human vocal variance verified."

        return jsonify({
            "result": result,
            "confidence": round(ai_conf * 100, 2),
            "explanation": explanation
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == "__main__":
    app.run(debug=True)