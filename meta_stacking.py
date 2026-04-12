import os
import numpy as np
import cv2
import pandas as pd
from skimage import feature
from joblib import load, dump
import librosa
import parselmouth
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================================================
# 🔧 CONTROL FLAG (SET TRUE ONLY ONCE)
# =====================================================
EVALUATE_META_MODEL_ONCE = True   # 👉 After checking accuracy, set to False

# =============================
# 1️⃣ Load Base Models
# =============================
spiral_model = load("spiral_model.pkl")
speech_model = load("parkinsons_best_model.pkl")
speech_scaler = load("speech_scaler.pkl")
feature_order = load("feature_order.pkl")

print("[INFO] Base models loaded")

# =============================
# 2️⃣ Spiral Feature Extraction
# =============================
def quantify_image(image):
    return feature.hog(
        image,
        orientations=9,
        pixels_per_cell=(10, 10),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L1"
    )

# =============================
# 3️⃣ Speech Feature Extraction
# =============================
SAMPLE_RATE = 16000
N_MFCC = 13
LPC_ORDER = 16

def extract_speech_features_from_wav(wav_path):
    features = {}

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"MFCC{i+1}_mean"] = np.mean(mfcc[i])
        features[f"MFCC{i+1}_var"] = np.var(mfcc[i])
        features[f"MFCC{i+1}_skew"] = skew(mfcc[i])
        features[f"MFCC{i+1}_kurt"] = kurtosis(mfcc[i])

    lpc = librosa.lpc(y, order=LPC_ORDER)
    for i, coef in enumerate(lpc):
        features[f"LPC{i+1}"] = coef

    spectrum = np.abs(np.fft.fft(y))
    cepstrum = np.fft.ifft(np.log(spectrum + 1e-10)).real
    features["Cep_mean"] = np.mean(cepstrum)
    features["Cep_var"] = np.var(cepstrum)

    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch()
    features["Pitch_mean"] = np.nanmean(pitch.selected_array['frequency'])
    features["Pitch_std"] = np.nanstd(pitch.selected_array['frequency'])

    pp = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    features["Jitter_local"] = parselmouth.praat.call(
        pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["Shimmer_local"] = parselmouth.praat.call(
        [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    features["RMS_energy"] = np.mean(librosa.feature.rms(y=y))

    df = pd.DataFrame([features])
    df = df.reindex(columns=feature_order, fill_value=0)
    return speech_scaler.transform(df)

# =============================
# 4️⃣ Load Training Data
# =============================
def load_training_data():
    spiral_base = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\spiral\training"
    audio_pd = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\PD_AH"
    audio_hc = r"C:\Users\DELL\OneDrive\Desktop\datasetnew\HC_AH"

    spiral_paths, audio_paths, labels = [], [], []

    for label, cls in [(1, "parkinson"), (0, "healthy")]:
        spiral_dir = os.path.join(spiral_base, cls)
        audio_dir = audio_pd if label == 1 else audio_hc

        for img_file, wav_file in zip(
            sorted(os.listdir(spiral_dir)),
            sorted(os.listdir(audio_dir))
        ):
            spiral_paths.append(os.path.join(spiral_dir, img_file))
            audio_paths.append(os.path.join(audio_dir, wav_file))
            labels.append(label)

    return spiral_paths, audio_paths, labels

# =============================
# 5️⃣ Train & Evaluate Meta-Model
# =============================
def train_and_evaluate_meta_model():
    spiral_paths, audio_paths, labels = load_training_data()

    speech_probs, spiral_probs, y_meta = [], [], []

    for img_path, wav_path, label in zip(spiral_paths, audio_paths, labels):

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (200, 200))
        img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )[1]

        spiral_feat = quantify_image(img).reshape(1, -1)
        spiral_prob = spiral_model.predict_proba(spiral_feat)[0, 1]

        speech_feat = extract_speech_features_from_wav(wav_path)
        speech_prob = speech_model.predict_proba(speech_feat)[0, 1]

        speech_probs.append(speech_prob)
        spiral_probs.append(spiral_prob)
        y_meta.append(label)

    X_meta = np.column_stack((speech_probs, spiral_probs))
    y_meta = np.array(y_meta)

    X_train, X_test, y_train, y_test = train_test_split(
        X_meta, y_meta, test_size=0.2, random_state=42, stratify=y_meta
    )

    meta_model = LogisticRegression()
    meta_model.fit(X_train, y_train)

    y_pred = meta_model.predict(X_test)

    print("\n📊 META-MODEL PERFORMANCE (ONE-TIME)")
    print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    dump(meta_model, "meta_stacking_model.pkl")
    print("[INFO] Meta-model saved")

# =============================
# 6️⃣ Test Single Sample
# =============================
def test_single_input(spiral_file, audio_file):
    meta_model = load("meta_stacking_model.pkl")

    img = cv2.imread(spiral_file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (200, 200))
    img = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )[1]

    spiral_feat = quantify_image(img).reshape(1, -1)
    spiral_prob = spiral_model.predict_proba(spiral_feat)[0, 1]

    speech_feat = extract_speech_features_from_wav(audio_file)
    speech_prob = speech_model.predict_proba(speech_feat)[0, 1]

    fused_prob = meta_model.predict_proba(
        np.array([[speech_prob, spiral_prob]])
    )[0, 1]

    print("\n🔹 Speech Probability :", round(speech_prob, 3))
    print("🔹 Spiral Probability :", round(spiral_prob, 3))
    print("🟢 Final Probability :", round(fused_prob, 3))
    print("🟢 Prediction :",
          "Parkinson’s Detected" if fused_prob > 0.5 else "Healthy Control")

# =============================
# 🚀 MAIN
# =============================
if __name__ == "__main__":

    if EVALUATE_META_MODEL_ONCE:
        train_and_evaluate_meta_model()
    elif not os.path.exists("meta_stacking_model.pkl"):
        train_and_evaluate_meta_model()

    # 🔹 Normal inference
    test_single_input(
        r"C:\Users\DELL\OneDrive\Desktop\datasetnew\spiral\testing\healthy\V09HE01.png",
        r"C:\Users\DELL\OneDrive\Desktop\datasetnew\PD_AH\AH_545713224-1B3708B0-8792-4FEE-B03B-C7CB9CB03D58.wav"
    )
