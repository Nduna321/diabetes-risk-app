"""Train the diabetes risk model and record how well it performs.

Usage:  python train_model.py data/diabetes.csv
Writes: artifacts/model.joblib and artifacts/metrics.json
"""
import json
import os
import sys

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from diabetes_app.features import FEATURES, TARGET, prepare_frame
from diabetes_app.model import RANDOM_SEED, build_pipeline, save_model

ARTIFACT_DIR = "artifacts"


def main(csv_path: str) -> None:
    df = prepare_frame(pd.read_csv(csv_path))
    X, y = df[FEATURES], df[TARGET]

    # Hold-out test set: 20% of rows, split with the class balance preserved
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED)

    model = build_pipeline().fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    holdout = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, prob), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_rows": int(len(y_test)),
    }

    # 5-fold cross-validation on all rows
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(build_pipeline(), X, y, cv=cv,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
    cross_val = {
        name.replace("test_", ""): {
            "mean": round(float(scores[name].mean()), 4),
            "std": round(float(scores[name].std()), 4),
        }
        for name in scores if name.startswith("test_")
    }

    # Final model is trained on every row and saved for the web app
    final = build_pipeline().fit(X, y)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    save_model(final, os.path.join(ARTIFACT_DIR, "model.joblib"))

    report = {"rows": int(len(df)), "features": FEATURES,
              "holdout": holdout, "cross_validation": cross_val}
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python train_model.py data/diabetes.csv")
    main(sys.argv[1])
