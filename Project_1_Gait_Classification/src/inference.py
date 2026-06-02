import joblib
from pathlib import Path
import pandas as pd

from src.features import extract_features, ACTIVITY_LABEL_MAP

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "xgb_activity_classifier.pkl"
ENCODER_PATH = ROOT / "models" / "activity_label_encoder.pkl"
FEATURE_PATH = ROOT / "models" / "feature_order.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)
    feature_order = joblib.load(FEATURE_PATH)
    return model, encoder, feature_order

def predict_activity(model, encoder, feature_order, window):
    """Predict the activity label for a given window of sensor data."""
    features = extract_features(window)
    features_df = pd.DataFrame([features])
    
    # Ensure the feature order matches what the model was trained on
    features_df = features_df[feature_order]
    
    # Predict the encoded label and then decode it to the original activity label
    encoded_label = model.predict(features_df)[0]
    activity_label = encoder.inverse_transform([encoded_label])[0] # inverse back to original label
    # I may need to map the encoder output back to the original activity labels using the ACTIVITY_LABEL_MAP
    activity_name = ACTIVITY_LABEL_MAP.get(activity_name, activity_label)
    
    return activity_label

def predict_dataframe(df):
    """Predict activity labels for each window in the DataFrame."""
    model, encoder, feature_order = load_model()
    predictions = []
    
    for subject in df['subject'].unique():
        subject_data = df[df['subject'] == subject].reset_index(drop=True)
        for i in range(0, len(subject_data) - 500, 250):  # using same window and step size as training
            window = subject_data.iloc[i:i+500]
            pred_label = predict_activity(model, encoder, feature_order, window)
            predictions.append(pred_label)
    
    return predictions

def predict_stream(data_buffer):
    """Predict activity label for a real-time stream of sensor data."""
    model, encoder, feature_order = load_model()
    
    if len(data_buffer) < 500:
        print("Not enough data to make a prediction. Need at least 500 samples.")
        return None
    
    window = pd.DataFrame(data_buffer[-500:])  # take the most recent 500 samples
    pred_label = predict_activity(model, encoder, feature_order, window)
    
    return pred_label