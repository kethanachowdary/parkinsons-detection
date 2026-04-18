Parkinson’s Disease Detection via fusion of speech and spiral features using machine learning

This project focuses on detecting Parkinson’s Disease by combining information from speech signals (audio) and hand-drawn spiral images using advanced ensemble learning techniques.

🚀 Overview
Parkinson’s disease affects both speech patterns and motor control.
To capture this, the system uses a multi-modal approach:

Audio (.wav files) for speech feature extraction
Spiral drawings for image-based feature extraction (HOG)
Fusion models to combine both modalities for final prediction

Methodology
1. Feature Extraction
🎤 Audio Features
Extracted from .wav files
Captures voice impairments such as tremors and instability

Features include:
MFCC (Mel-Frequency Cepstral Coefficients)
Pitch, jitter, shimmer

✍️ Spiral Image Features
Parkinson’s patients often show irregular hand movements
Histogram of Oriented Gradients (HOG) is used to extract shape and texture features

3. Fusion Approaches
🔹 Weighted Averaging
Combines predictions from audio and image models
Uses predefined weights
Serves as a baseline method
🔹 Attention-Based Fusion
Learns the importance of each modality dynamically
Assigns adaptive weights to audio and image features
🔹 Meta-Stacking
Uses outputs of individual models as input features
Trains a meta-learner such as Logistic Regression or XGBoost
Captures complex relationships between predictions

📂 Project Structure
├── data/
│   ├── audio_data/
│   └── spiral_data/
├── src/
│   ├── audio.py
│   ├── spiral_binary.py
│   └── feature_extraction.py
│   ├── models.py
│   ├── weighted_average.py
│   ├── attention_fusion.py
│   ├── meta_stacking.py
├── notebooks/
└── README.md

⚙️ Installation
git clone https://github.com/kethanachowdary/parkinsons-detection
cd parkinsons-detection


## Results

| Fusion Method       | Performance |
|--------------------|------------|
| Weighted Averaging | Baseline   |
| Attention-Based    | Improved   |
| Meta-Stacking      | Best       |


A simple web application was developed to make the model accessible for real-time predictions.
https://parkinsons-detection-system.vercel.app
