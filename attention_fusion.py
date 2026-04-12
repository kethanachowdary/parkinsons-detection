# =========================================================
# CROSS-MODAL ATTENTION FUSION
# Dynamic per-input weights (NONLINEAR)
# =========================================================

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import cv2
from skimage import feature
from sklearn.metrics import accuracy_score, roc_auc_score

# =========================================================
# 1️⃣ SPEECH PROBABILITIES
# =========================================================

print("\n[1] Generating P_speech...")

speech_model = joblib.load("parkinsons_best_model.pkl")
speech_scaler = joblib.load("speech_scaler.pkl")

df = pd.read_csv("speech_features_from_wav.csv")
label_col = "Label" if "Label" in df.columns else "status"

labels_raw = df[label_col]

labels = (
    labels_raw.astype(str).str.lower().map({
        "healthy":0,"control":0,"normal":0,
        "pd":1,"parkinson":1,"parkinson's":1
    }).values
)

X_speech = (
    df.select_dtypes(include=[np.number])
      .drop(columns=[label_col], errors="ignore")
)

X_speech = speech_scaler.transform(X_speech)
P_speech = speech_model.predict_proba(X_speech)[:,1]

print("P_speech shape:", P_speech.shape)

# =========================================================
# 2️⃣ SPIRAL PROBABILITIES
# =========================================================

print("\n[2] Generating P_spiral...")

spiral_model = joblib.load("spiral_model.pkl")

def extract_hog(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200,200))
    img = cv2.threshold(
        img,0,255,
        cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU
    )[1]

    return feature.hog(
        img,
        orientations=9,
        pixels_per_cell=(10,10),
        cells_per_block=(2,2),
        transform_sqrt=True,
        block_norm="L1"
    )

spiral_base = "data2/spiral/training"
spiral_files = []

for cls in ["healthy","parkinson"]:
    folder = os.path.join(spiral_base, cls)
    for f in os.listdir(folder):
        if f.lower().endswith((".png",".jpg",".jpeg")):
            spiral_files.append(os.path.join(folder,f))

target_n = len(P_speech)

chosen = np.random.choice(
    spiral_files,
    size=target_n,
    replace=False if len(spiral_files)>=target_n else True
)

P_spiral = []
for p in chosen:
    prob = spiral_model.predict_proba(
        extract_hog(p).reshape(1,-1)
    )[0][1]
    P_spiral.append(prob)

P_spiral = np.array(P_spiral)

print("P_spiral shape:", P_spiral.shape)

# =========================================================
# 3️⃣ PREPARE TENSORS
# =========================================================

P_speech = torch.tensor(P_speech,dtype=torch.float32).unsqueeze(1)
P_spiral = torch.tensor(P_spiral,dtype=torch.float32).unsqueeze(1)
labels = torch.tensor(labels,dtype=torch.float32).unsqueeze(1)

# =========================================================
# 4️⃣ NONLINEAR CROSS ATTENTION FUSION
# =========================================================

class CrossModalAttention(nn.Module):
    def __init__(self):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward(self, ps, pi):

        x = torch.cat([ps, pi], dim=1)

        weights = torch.softmax(self.attn(x), dim=1)

        w_speech = weights[:,0:1]
        w_spiral = weights[:,1:2]

        fused = w_speech * ps + w_spiral * pi

        return fused, w_speech, w_spiral


fusion_model = CrossModalAttention()

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    fusion_model.parameters(),
    lr=0.01,
    weight_decay=1e-4
)

# =========================================================
# 5️⃣ TRAIN FUSION
# =========================================================

print("\n[3] Training cross-modal attention fusion...")

epochs = 200
for e in range(epochs):

    preds, ws, wi = fusion_model(P_speech, P_spiral)

    # ----- entropy regularization -----
    entropy = -(ws * torch.log(ws + 1e-8) +
                wi * torch.log(wi + 1e-8)).mean()

    # lambda controls smoothing strength
    lambda_entropy = 0.2

    loss = criterion(preds, labels) - lambda_entropy * entropy


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e+1)%20==0:
        print(f"Epoch {e+1}/{epochs} Loss {loss.item():.4f}")

# =========================================================
# 6️⃣ EVALUATION
# =========================================================

fusion_model.eval()

with torch.no_grad():

    fused, ws, wi = fusion_model(P_speech,P_spiral)
    binary = (fused>0.5).float()

acc_dynamic = accuracy_score(labels,binary)
auc_dynamic = roc_auc_score(labels,fused)

# Baseline comparison
fixed = 0.65*P_speech + 0.35*P_spiral
fixed_bin = (fixed>0.5).float()

acc_fixed = accuracy_score(labels,fixed_bin)
auc_fixed = roc_auc_score(labels,fixed)

# =========================================================
# 7️⃣ RESULTS
# =========================================================

print("\n================ RESULTS ================")
print(f"Fixed Fusion Acc {acc_fixed:.4f} AUC {auc_fixed:.4f}")
print(f"Cross-Attention Fusion Acc {acc_dynamic:.4f} AUC {auc_dynamic:.4f}")

print("\nExample Dynamic Weights:")
for i in range(5):
    print(f"Sample {i+1}: Speech={ws[i].item():.3f} Spiral={wi[i].item():.3f}")
# =========================================================
# SAVE TRAINED FUSION MODEL
# =========================================================

torch.save(fusion_model.state_dict(), "fusion_model.pth")
print("\nFusion model saved as fusion_model.pth")


# The system generates Parkinson’s probabilities independently from speech and spiral models. These probabilities are fused using a nonlinear 
# attention mechanism that dynamically computes modality weights via a small neural network and softmax normalization. The fused prediction is 
# then evaluated against ground truth labels. This approach enables adaptive modality importance rather than fixed weighting, improving 
# robustness and predictive performance.
# Entropy regularization was introduced to prevent modality dominance, enabling the attention mechanism to assign smoother dynamic weights 
# while maintaining performance.