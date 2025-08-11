import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle  # for saving/loading model

def extract_features_from_file(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    # data is list of dicts, each with 'keypoints', 'score', 'box', etc.
    
    keypoints_list = [item['keypoints'] for item in data if 'keypoints' in item]
    
    if len(keypoints_list) == 0:
        return None  # no data
    
    flattened = np.array(keypoints_list).flatten()
    return flattened

def load_dataset(base_folder):
    features = []
    labels = []
    
    for class_folder in os.listdir(base_folder):
        class_path = os.path.join(base_folder, class_folder)
        if not os.path.isdir(class_path):
            continue
        
        for root, dirs, files in os.walk(class_path):
            for filename in files:
                if filename.endswith('.json'):
                    filepath = os.path.join(root, filename)
                    feature_vector = extract_features_from_file(filepath)
                    if feature_vector is not None:
                        features.append(feature_vector)
                        labels.append(class_folder)  # folder name as label
    
    X = pd.DataFrame(features)
    y = pd.Series(labels)
    return X, y

def main():
    base_folder = 'data'
    
    print("Loading dataset...")
    X, y = load_dataset(base_folder)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Labels distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Save the trained model
    with open('rf_parkinsons_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print("Model saved as rf_parkinsons_model.pkl")
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()