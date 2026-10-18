"""Feature definitions, input validation, and data cleaning."""
import numpy as np
import pandas as pd

# Column name in the data, label shown to users, minimum, maximum, expected type
FEATURE_SPECS = [
    ("Pregnancies", "Number of pregnancies", 0, 20, int),
    ("Glucose", "Glucose (mg/dL)", 40, 300, float),
    ("BloodPressure", "Blood pressure (mmHg)", 30, 200, float),
    ("SkinThickness", "Skin thickness (mm)", 5, 100, float),
    ("Insulin", "Insulin (IU/mL)", 10, 900, float),
    ("BMI", "Body mass index (kg/m²)", 10, 70, float),
    ("DiabetesPedigreeFunction", "Diabetes pedigree function", 0.05, 3.0, float),
    ("Age", "Age (years)", 18, 120, int),
]
FEATURES = [spec[0] for spec in FEATURE_SPECS]
TARGET = "Outcome"

# In the source data, 0 means "not measured" for these columns, not a true zero.
ZERO_MEANS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


class ValidationError(ValueError):
    pass


def validate_inputs(raw: dict) -> dict:
    """Convert raw form values to typed, range-checked numbers.

    Raises ValidationError with a message a user can read.
    """
    clean = {}
    for name, label, low, high, kind in FEATURE_SPECS:
        text = str(raw.get(name, "")).strip()
        if text == "":
            raise ValidationError(f"{label} is required.")
        try:
            number = float(text)
        except ValueError:
            raise ValidationError(f"{label} must be a number.")
        if kind is int:
            if number != int(number):
                raise ValidationError(f"{label} must be a whole number.")
            number = int(number)
        if not low <= number <= high:
            raise ValidationError(f"{label} must be between {low} and {high}.")
        clean[name] = number
    return clean


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where impossible zeros are marked as missing (NaN)."""
    out = df.copy()
    out[ZERO_MEANS_MISSING] = out[ZERO_MEANS_MISSING].replace(0, np.nan)
    return out
