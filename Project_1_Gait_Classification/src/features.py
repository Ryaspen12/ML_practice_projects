import pandas as pd 
import numpy as np

def get_window_label(window):
    return window["activity"].mode()[0]

def extract_features(window):
    features = {}
    sensor_cols = [c for c in window.columns if c not in ["timestamp", "activity", "heart_rate", "subject"]]
    
    for col in sensor_cols:
        x = window[col].values
        features[f"{col}_mean"] = np.mean(x)
        features[f"{col}_std"] = np.std(x)
        features[f"{col}_min"] = np.min(x)
        features[f"{col}_max"] = np.max(x)
        features[f"{col}_rms"] = np.sqrt(np.mean(x**2))

    return features

ACTIVITY_LABEL_MAP = {
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping"
}

def pull_features_per_subject(df, window_size=500, step_size=250):
    """Extract features from the DataFrame using a sliding window approach (5s at 100Hz), 
    ensuring that windows are created separately for each subject to prevent data leakage."""
    X, y, subjects = [], [], []
    for subject in df['subject'].unique(): #split by subject
        subject_data = df[df['subject'] == subject].reset_index(drop=True)
        for i in range(0, len(subject_data) - window_size, step_size):
            window = subject_data.iloc[i:i+window_size]
            X.append(extract_features(window))
            y.append(get_window_label(window))
            subjects.append(subject)

    X = pd.DataFrame(X)
    y = pd.Series(y)
    subjects = pd.Series(subjects)
    return X, y, subjects