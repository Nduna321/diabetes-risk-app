# Diabetes Risk Prediction: Machine Learning Web Application

**Author:** Proud Ndlovu

An end-to-end machine learning application that estimates diabetes risk from patient diagnostic measurements. A Random Forest classifier is trained with scikit-learn, evaluated with held-out and cross-validated metrics, and served through a Flask web form.

## Overview
The application uses eight diagnostic features, including number of pregnancies, glucose, insulin level, BMI, and age, to predict whether a patient is likely to have diabetes. It covers the full workflow: data cleaning, model training, evaluation, saving the model, and serving predictions to users.

The dataset is the Pima Indians Diabetes Database, a public healthcare dataset of 768 patients (female, at least 21 years old, of Pima Indian heritage).

## Features
| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration, 2 hours into an oral glucose test (mg/dL) |
| BloodPressure | Diastolic blood pressure (mmHg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (IU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Score reflecting family history of diabetes |
| Age | Age in years |

## How it works
1. **Cleaning.** In this dataset a value of 0 for Glucose, BloodPressure, SkinThickness, Insulin, or BMI means the measurement is missing, so those zeros are converted to missing values.
2. **Imputation.** Missing values are filled with each column's median. The imputer is part of the model pipeline, so it learns medians only from training data. This prevents information from the test data leaking into training.
3. **Model.** A Random Forest with 300 trees, minimum 2 samples per leaf, and balanced class weights. Balanced weights account for the dataset having more non-diabetic than diabetic patients.
4. **Evaluation.** A stratified 80/20 train-test split gives a held-out estimate. Five-fold stratified cross-validation gives a more stable estimate across the full dataset.
5. **Serving.** The web app validates each input (required, numeric, within an allowed range), runs the saved model, and shows the predicted label, the probability, and the values used.

## Technologies
- **Python 3**
- **Pandas**: loading and cleaning data
- **NumPy**: numerical operations and missing values
- **scikit-learn**: imputation, Random Forest classifier, pipelines, cross-validation, and metrics
- **joblib**: saving and loading the trained model
- **Flask**: web application and routing

## Project structure
```
diabetes-risk-app/
├── diabetes_app/
│   ├── features.py       # Feature definitions, input validation, cleaning
│   ├── model.py          # Pipeline definition, save and load
│   └── predictor.py      # Loads the model and makes predictions
├── templates/
│   ├── form.html         # Input form
│   └── result.html       # Prediction result
├── data/                 # Place diabetes.csv here (not committed)
├── train_model.py        # Trains, evaluates, and saves the model
├── web_app.py            # Flask application
├── requirements.txt
└── .gitignore
```

## Setup
1. Download the Pima Indians Diabetes dataset from Kaggle and save it as `data/diabetes.csv`.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Train and evaluate the model:
   ```bash
   python train_model.py data/diabetes.csv
   ```
   This writes `artifacts/model.joblib` and `artifacts/metrics.json`.
4. Run the web application:
   ```bash
   python web_app.py
   ```
   Open http://127.0.0.1:5000 in a browser.

## Results
Results from running `train_model.py` on the 768-row Pima Indians Diabetes dataset, with the Python 3.15 environment described in Setup.

**Held-out test set (20%, 154 patients)**

| Metric | Value |
|---|---|
| Accuracy | 0.714 |
| Precision | 0.581 |
| Recall | 0.667 |
| F1-score | 0.621 |
| ROC-AUC | 0.757 |

Confusion matrix on the test set: 74 true negatives, 26 false positives, 18 false negatives, 36 true positives.

**5-fold cross-validation (all 768 patients, mean ± standard deviation)**

| Metric | Result |
|---|---|
| Accuracy | 0.754 ± 0.022 |
| Precision | 0.631 ± 0.035 |
| Recall | 0.721 ± 0.055 |
| F1-score | 0.671 ± 0.026 |
| ROC-AUC | 0.830 ± 0.021 |

## Limitations
- This is a learning project. It is not a medical device and must not be used for diagnosis.
- The dataset is small and covers one population, so results may not generalise to other groups.
- Median imputation is a simple method and may not reflect the true values of missing measurements.
- Recall and precision matter more than accuracy in screening tasks. Review the confusion matrix in `metrics.json` before drawing conclusions.

## Data source
Pima Indians Diabetes Database, originally from the National Institute of Diabetes and Digestive and Kidney Diseases, available on Kaggle. Check the dataset page for its current license before publishing.
