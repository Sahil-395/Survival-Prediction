# Heart Failure Survival Prediction System

**Project Developer:** Sahil  
**Roll Number:** 16

## Project overview

This is an academic machine-learning project that uses the UCI Heart Failure Clinical Records dataset to classify the recorded `DEATH_EVENT` outcome. The notebook explores the data, performs basic cleaning and EDA, trains three classification models, compares their results, and saves the selected model. A simple Streamlit application accepts the same 11 baseline clinical features and displays one result: the model-estimated chance of no death event during the dataset follow-up, shown out of 100.

> **Important:** This is an educational demonstration, not a medical diagnosis or clinically validated prediction system. It does not estimate a person's lifespan or guarantee a real-world outcome. It must not be used for healthcare or treatment decisions.

## Dataset

- Dataset: Heart Failure Clinical Records
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- File: `data/heart_failure_clinical_records_dataset.csv`
- Records: 299
- Target: `DEATH_EVENT` (0 = no death event recorded during follow-up; 1 = death event recorded during follow-up)
- Original columns: 13

The `time` column represents the follow-up period and is excluded from model training and app inputs to avoid using follow-up information as a baseline predictor. The app converts the model probability for class 0 into a simple score out of 100; this is a model estimate from historical records, not a real or time-specific personal survival probability. The model uses 11 baseline features: `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, and `smoking`.

Dataset attribution: Davide Chicco and Giuseppe Jurman, *Machine learning can predict the survival of patients with heart failure from serum creatinine and ejection fraction alone*, BMC Medical Informatics and Decision Making (2020), dataset hosted by the UCI Machine Learning Repository. See the UCI page for the dataset licence and citation details.

## Machine-learning models

1. Random Forest Classifier
2. Logistic Regression (with StandardScaler in a pipeline)
3. Decision Tree Classifier

Models are compared using stratified cross-validation on the training data and test-set Accuracy, Precision, Recall, F1-score, and ROC-AUC. The best model is selected automatically using the highest mean cross-validation F1-score for class 1. Actual scores are generated when the notebook is run; they are not hardcoded in this README.

## Project structure

```text
survival_prediction/
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── survival_prediction.ipynb
├── app.py
├── best_model.joblib
├── requirements.txt
└── README.md
```

## Installation

Use a compatible Python environment (Python 3.11 or newer is recommended). Open a terminal in the `survival_prediction` project folder and run:

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
.venv\Scripts\activate.bat
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The `scikit-learn` version is pinned because loading a joblib model across different scikit-learn versions is not officially supported. If you change the scikit-learn version, rerun the notebook to regenerate `best_model.joblib` in that environment.

## Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `survival_prediction.ipynb` and run all cells from top to bottom. This trains and evaluates the models and regenerates `best_model.joblib`.

## Run the Streamlit application

After successfully running the notebook, start the app from the same project folder:

```bash
streamlit run app.py
```

The browser should open the local Streamlit application. If `best_model.joblib` is missing, run the notebook first.

## Developer

**Sahil — Roll No. 16**
