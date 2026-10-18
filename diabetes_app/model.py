"""Build, save, and load the Random Forest pipeline."""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANDOM_SEED = 2026


def build_pipeline() -> Pipeline:
    """Median imputation followed by a Random Forest classifier.

    The imputer sits inside the pipeline, so during cross-validation each fold
    learns its medians from its own training data only.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )),
    ])


def save_model(pipeline, path: str) -> None:
    joblib.dump(pipeline, path)


def load_model(path: str):
    return joblib.load(path)
