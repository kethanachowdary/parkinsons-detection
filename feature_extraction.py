# ============================================================
# Feature Extraction from Raw WAV files
# Parkinson’s Disease Speech Analysis
# Output: Excel + CSV
# ============================================================

import os
import numpy as np
import pandas as pd
import librosa
import parselmouth
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
N_MFCC = 13
LPC_ORDER = 16

# ============================================================
# FEATURE EXTRACTION FOR ONE WAV FILE
# ============================================================

def extract_features_from_wav(wav_path):
    features = {}

    # ---------------- Load audio ----------------
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    # ---------------- MFCC features ----------------
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    for i in range(N_MFCC):
        features[f"MFCC{i+1}_mean"] = np.mean(mfcc[i])
        features[f"MFCC{i+1}_var"]  = np.var(mfcc[i])
        features[f"MFCC{i+1}_skew"] = skew(mfcc[i])
        features[f"MFCC{i+1}_kurt"] = kurtosis(mfcc[i])

    # ---------------- LPC coefficients ----------------
    lpc = librosa.lpc(y, order=LPC_ORDER)
    for i, coef in enumerate(lpc):
        features[f"LPC{i+1}"] = coef

    # ---------------- Cepstral features ----------------
    spectrum = np.abs(np.fft.fft(y))
    cepstrum = np.fft.ifft(np.log(spectrum + 1e-10)).real
    features["Cep_mean"] = np.mean(cepstrum)
    features["Cep_var"]  = np.var(cepstrum)

    # ---------------- Parselmouth (Praat-based) ----------------
    snd = parselmouth.Sound(wav_path)

    pitch = snd.to_pitch()
    features["Pitch_mean"] = np.nanmean(pitch.selected_array['frequency'])
    features["Pitch_std"]  = np.nanstd(pitch.selected_array['frequency'])

    point_process = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    features["Jitter_local"] = parselmouth.praat.call(
        point_process, "Get jitter (local)",
        0, 0, 0.0001, 0.02, 1.3
    )

    features["Shimmer_local"] = parselmouth.praat.call(
        [snd, point_process], "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # ---------------- Energy ----------------
    features["RMS_energy"] = np.mean(librosa.feature.rms(y=y))

    return features

# ============================================================
# BATCH EXTRACTION FROM DATASET
# ============================================================

def extract_from_dataset(base_folder):
    rows = []

    label_map = {
        "HC_AH": "Healthy",
        "PD_AH": "PD"
    }

    for folder, label in label_map.items():
        folder_path = os.path.join(base_folder, folder)

        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(".wav"):
                wav_path = os.path.join(folder_path, file)
                try:
                    feats = extract_features_from_wav(wav_path)
                    feats["Sample"] = file
                    feats["Label"] = label
                    rows.append(feats)
                    print(f"✔ Extracted: {file}")
                except Exception as e:
                    print(f"❌ Failed: {file} → {e}")

    return pd.DataFrame(rows)

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    base_folder = r"C:\Users\DELL\OneDrive\Desktop\datasetnew"

    df = extract_from_dataset(base_folder)

    df.to_excel("speech_features_from_wav.xlsx", index=False)
    df.to_csv("speech_features_from_wav.csv", index=False)

    print("\n✅ Feature extraction completed successfully")
    print("📁 Saved files:")
    print("   - speech_features_from_wav.xlsx")
    print("   - speech_features_from_wav.csv")
    print("📊 Total samples extracted:", len(df))
