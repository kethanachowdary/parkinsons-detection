# =========================================================
# TEST SCRIPT
# Input:
#   - Raw WAV file
#   - Spiral image
# Output:
#   - Speech prob
#   - Spiral prob
#   - Dynamic weights
#   - Final prediction
# =========================================================

import numpy as np
import torch
import torch.nn as nn
import joblib
import cv2
from skimage import feature
import librosa
import parselmouth
from scipy.stats import skew, kurtosis

# =========================================================
# LOAD TRAINED MODELS
# =========================================================

speech_model = joblib.load("parkinsons_best_model.pkl")
speech_scaler = joblib.load("speech_scaler.pkl")
feature_order = joblib.load("feature_order.pkl")

spiral_model = joblib.load("spiral_model.pkl")

# =========================================================
# LOAD TRAINED FUSION MODEL
# =========================================================

class CrossModalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,2)
        )

    def forward(self, ps, pi):
        x = torch.cat([ps, pi], dim=1)
        weights = torch.softmax(self.attn(x), dim=1)
        ws = weights[:,0:1]
        wi = weights[:,1:2]
        fused = ws*ps + wi*pi
        return fused, ws, wi


fusion_model = CrossModalAttention()
fusion_model.load_state_dict(torch.load("fusion_model.pth"))
fusion_model.eval()

# =========================================================
# SPEECH FEATURE EXTRACTION
# =========================================================

def extract_speech_features(wav_path):

    SAMPLE_RATE = 16000
    N_MFCC = 13
    LPC_ORDER = 16

    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        features[f"MFCC{i+1}_mean"] = np.mean(mfcc[i])
        features[f"MFCC{i+1}_var"]  = np.var(mfcc[i])
        features[f"MFCC{i+1}_skew"] = skew(mfcc[i])
        features[f"MFCC{i+1}_kurt"] = kurtosis(mfcc[i])

    lpc = librosa.lpc(y, order=LPC_ORDER)
    for i, coef in enumerate(lpc):
        features[f"LPC{i+1}"] = coef

    spectrum = np.abs(np.fft.fft(y))
    cep = np.fft.ifft(np.log(spectrum+1e-10)).real
    features["Cep_mean"] = np.mean(cep)
    features["Cep_var"] = np.var(cep)

    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch()
    features["Pitch_mean"] = np.nanmean(pitch.selected_array['frequency'])
    features["Pitch_std"] = np.nanstd(pitch.selected_array['frequency'])

    pp = parselmouth.praat.call(snd,"To PointProcess (periodic, cc)",75,500)
    features["Jitter_local"] = parselmouth.praat.call(
        pp,"Get jitter (local)",0,0,0.0001,0.02,1.3)
    features["Shimmer_local"] = parselmouth.praat.call(
        [snd,pp],"Get shimmer (local)",0,0,0.0001,0.02,1.3,1.6)

    features["RMS_energy"] = np.mean(librosa.feature.rms(y=y))

    # reorder to match training
    feat_vec = np.array([features.get(f,0) for f in feature_order]).reshape(1,-1)
    feat_vec = speech_scaler.transform(feat_vec)

    prob = speech_model.predict_proba(feat_vec)[0][1]
    return prob

# =========================================================
# SPIRAL PROBABILITY
# =========================================================

def spiral_prob(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(200,200))
    img = cv2.threshold(img,0,255,
                        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)[1]

    hog_feat = feature.hog(
        img,
        orientations=9,
        pixels_per_cell=(10,10),
        cells_per_block=(2,2),
        transform_sqrt=True,
        block_norm="L1"
    ).reshape(1,-1)

    return spiral_model.predict_proba(hog_feat)[0][1]

# =========================================================
# PREDICT FUNCTION
# =========================================================

def predict(audio_path, spiral_path):

    ps = extract_speech_features(audio_path)
    pi = spiral_prob(spiral_path)

    ps_t = torch.tensor([[ps]],dtype=torch.float32)
    pi_t = torch.tensor([[pi]],dtype=torch.float32)

    fused, ws, wi = fusion_model(ps_t,pi_t)
    final_prob = fused.item()

    print("\n========= RESULT =========")
    print(f"Speech Prob : {ps:.4f}")
    print(f"Spiral Prob : {pi:.4f}")
    print(f"Weight Speech : {ws.item():.3f}")
    print(f"Weight Spiral : {wi.item():.3f}")
    print(f"Final Prob : {final_prob:.4f}")

    label = "Parkinson Detected" if final_prob > 0.5 else "Healthy"
    print("Prediction :", label)


# =========================================================
# RUN HERE
# =========================================================

if __name__ == "__main__":
    audio_file = r"audio_data\HC_AH\AH_333L_6C551A6E-CC47-410E-AA49-2DC0A86E6489.wav"
    spiral_file = r"data2\spiral\testing\parkinson\V07PE01.png"
    predict(audio_file, spiral_file)
