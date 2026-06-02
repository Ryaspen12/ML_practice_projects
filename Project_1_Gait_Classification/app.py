from fastapi import FastAPI
import pandas as pd

from src.inference import load_model, predict_activity

app = FastAPI()

model, encoder, feature_order = load_model()

@app.post("/predict")
def predict(window: dict):

    df = pd.DataFrame(window)

    label = predict_activity(model, encoder, feature_order, df)

    return {"activity": label}