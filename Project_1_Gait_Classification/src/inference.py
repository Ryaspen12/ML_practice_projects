import joblib
import pandas as pd

from src.features import extract_features

def load_model(model_path="../models/xgb_activity_classifier.pkl", encoder_path="../models/activity_label_encoder.pkl", feature_order_path="../models/feature_order.pkl"):
    """Load the trained model, label encoder, and feature order."""
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    feature_order = joblib.load(feature_order_path)
    return model, encoder, feature_order

def predict_activity(model, encoder, feature_order, window):
    """Predict the activity label for a given window of sensor data."""
    features = extract_features(window)
    features_df = pd.DataFrame([features])
    
    # Ensure the feature order matches what the model was trained on
    features_df = features_df[feature_order]
    
    # Predict the encoded label and then decode it to the original activity label
    encoded_label = model.predict(features_df)[0]
    activity_label = encoder.inverse_transform([encoded_label])[0]
    # I may need to map the encoder output back to the original activity labels using the ACTIVITY_LABEL_MAP
    # activity_name = src.features.ACTIVITY_LABEL_MAP.get(activity_name, activity_label)
    
    return activity_label