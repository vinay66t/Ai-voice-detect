import sys
import os
import numpy as np
import librosa
import joblib
import scipy.signal as signal
import sounddevice as sd
from scipy.io.wavfile import write

# ==============================
# SETTINGS
# ==============================
SAMPLE_RATE = 16000
DURATION = 10  # seconds
TEMP_FILE = "temp_recording.wav"

# ==============================
# LOAD MODEL
# ==============================
try:
    model = joblib.load("voice_scam_model.pkl")
except Exception as e:
    print("❌ Could not load model:", e)
    sys.exit()

# ==============================
# BANDPASS FILTER
# ==============================
def bandpass_filter(audio, sr):
    low = 300
    high = 3400
    b, a = signal.butter(4, [low/(sr/2), high/(sr/2)], btype='band')
    return signal.lfilter(b, a, audio)

# ==============================
# FEATURE EXTRACTION
# ==============================
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio)

        if len(audio) < DURATION * SAMPLE_RATE:
            audio = np.pad(audio, (0, DURATION*SAMPLE_RATE - len(audio)))
        else:
            audio = audio[:DURATION*SAMPLE_RATE]

        audio = bandpass_filter(audio, sr)

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.std(delta, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ])

        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

        extra = np.array([zcr, spectral_centroid, spectral_bandwidth, spectral_flatness])

        return np.hstack((features, extra))

    except Exception as e:
        print("❌ Error processing audio:", e)
        return None

# ==============================
# LIVE RECORDING FUNCTION
# ==============================
def record_from_mic():
    print("\n🎙️ Recording for", DURATION, "seconds...")
    print("Speak now...")

    recording = sd.rec(int(DURATION * SAMPLE_RATE),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32')

    sd.wait()

    print("✅ Recording complete.\n")

    write(TEMP_FILE, SAMPLE_RATE, recording)
    return TEMP_FILE

# ==============================
# PREDICTION FUNCTION
# ==============================
proba = model.predict_proba(features)[0]

ai_confidence = proba[1]
human_confidence = proba[0]

threshold = 0.65  # adjustable

if ai_confidence > threshold:
    result = "AI Generated Voice (Scam)"
else:
    result = "Human Voice"

confidence = max(ai_confidence, human_confidence)

    print(f"Confidence (Real): {real_prob}%")
    print(f"Confidence (Scam): {scam_prob}%")
    print("============================\n")


# ==============================
# MAIN
# ==============================
if len(sys.argv) == 2 and sys.argv[1] == "--live":
    file_path = record_from_mic()
    predict_audio(file_path)

elif len(sys.argv) == 2:
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("❌ File not found:", file_path)
        sys.exit()

    predict_audio(file_path)

else:
    print("\nUsage:")
    print("  python test_voice.py path_to_audio.wav")
    print("  python test_voice.py --live\n")