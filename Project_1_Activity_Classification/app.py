from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import List

from src.inference import load_model
from src.features import extract_features, ACTIVITY_LABEL_MAP

app = FastAPI()

model, encoder, feature_order = load_model()

class WindowRequest(BaseModel):
    columns: List[str]
    window: List[List[float]]

@app.post("/predict")
def predict(payload: WindowRequest):
    df = pd.DataFrame(payload.window, columns=payload.columns)

    features = extract_features(df)
    features_df = pd.DataFrame([features])
    features_df = features_df[feature_order]

    encoded_label = model.predict(features_df)[0]
    activity_label = encoder.inverse_transform([encoded_label])[0]
    activity_name = ACTIVITY_LABEL_MAP.get(activity_label, activity_label)

    return {"activity": activity_name}