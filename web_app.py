"""Flask front end: collects patient measurements and shows the prediction."""
import os

from flask import Flask, render_template, request

from diabetes_app.features import FEATURE_SPECS, ValidationError, validate_inputs
from diabetes_app.predictor import DiabetesPredictor

app = Flask(__name__)
_predictor = None


def get_predictor() -> DiabetesPredictor:
    """Load the model on first use, so the app starts even before training."""
    global _predictor
    if _predictor is None:
        _predictor = DiabetesPredictor()
    return _predictor


@app.route("/", methods=["GET"])
def form():
    return render_template("form.html", fields=FEATURE_SPECS, values={}, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    values = request.form.to_dict()
    try:
        clean = validate_inputs(values)
    except ValidationError as err:
        return render_template("form.html", fields=FEATURE_SPECS,
                               values=values, error=str(err)), 400
    result = get_predictor().predict(clean)
    return render_template("result.html", result=result, inputs=clean)


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
