"""Turn validated inputs into a prediction and a probability."""
import os

import pandas as pd

from diabetes_app.features import FEATURES
from diabetes_app.model import load_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(ROOT, "artifacts", "model.joblib")


class DiabetesPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model = load_model(model_path)

    def predict(self, values: dict) -> dict:
        row = pd.DataFrame([values], columns=FEATURES)
        probability = float(self.model.predict_proba(row)[0][1])
        return {"label": int(probability >= 0.5),
                "probability_percent": round(probability * 100, 1)}
