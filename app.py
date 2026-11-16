"""Simple Heart Failure Survival Prediction System - academic demonstration."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "best_model.joblib"

FEATURE_NAMES = [
    "age",
    "anaemia",
    "creatinine_phosphokinase",
    "diabetes",
    "ejection_fraction",
    "high_blood_pressure",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "sex",
    "smoking",
]

st.set_page_config(
    page_title="Survival Prediction",
    layout="centered",
)


@st.cache_resource(show_spinner="Loading model...")
def load_model_artifact(model_path: str):
    return joblib.load(model_path)


def main():
    st.title("Survival Prediction")
    st.caption("Enter the patient's details to get a simple model estimate.")
    st.caption("Developed by Sahil | Roll No. 16")

    if not MODEL_PATH.exists():
        st.error("best_model.joblib was not found. Run the Jupyter Notebook first.")
        st.stop()

    try:
        artifact = load_model_artifact(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
        st.info("Install the packages in requirements.txt, including the specified scikit-learn version.")
        st.stop()

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_names" not in artifact:
        st.error("The saved model file has an unexpected format. Please run the notebook again.")
        st.stop()

    saved_features = list(artifact["feature_names"])
    if saved_features != FEATURE_NAMES:
        st.error("The model input columns do not match the app. Please regenerate best_model.joblib.")
        st.stop()

    with st.form("patient_details"):
        st.subheader("Patient Details")
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=60, step=1)
            anaemia = st.selectbox("Anaemia", ["No", "Yes"])
            cpk = st.number_input(
                "Creatinine Phosphokinase (mcg/L)",
                min_value=0.0, max_value=50000.0, value=250.0, step=10.0,
            )
            diabetes = st.selectbox("Diabetes", ["No", "Yes"])
            ejection_fraction = st.number_input(
                "Ejection Fraction (%)", min_value=0.0, max_value=100.0,
                value=38.0, step=1.0,
            )
            high_blood_pressure = st.selectbox("High Blood Pressure", ["No", "Yes"])

        with right:
            platelets = st.number_input(
                "Platelets (kiloplatelets/mL)", min_value=0.0,
                max_value=2000000.0, value=262000.0, step=1000.0,
            )
            serum_creatinine = st.number_input(
                "Serum Creatinine (mg/dL)", min_value=0.0,
                max_value=50.0, value=1.1, step=0.1, format="%.2f",
            )
            serum_sodium = st.number_input(
                "Serum Sodium (mEq/L)", min_value=80.0,
                max_value=200.0, value=137.0, step=1.0,
            )
            sex = st.selectbox("Sex", ["Female", "Male"])
            smoking = st.selectbox("Smoking", ["No", "Yes"])

        submitted = st.form_submit_button("Check Survival Estimate", type="primary", use_container_width=True)

    if submitted:
        try:
            # Dataset encoding: No/Female = 0; Yes/Male = 1.
            patient = {
                "age": float(age),
                "anaemia": 1 if anaemia == "Yes" else 0,
                "creatinine_phosphokinase": float(cpk),
                "diabetes": 1 if diabetes == "Yes" else 0,
                "ejection_fraction": float(ejection_fraction),
                "high_blood_pressure": 1 if high_blood_pressure == "Yes" else 0,
                "platelets": float(platelets),
                "serum_creatinine": float(serum_creatinine),
                "serum_sodium": float(serum_sodium),
                "sex": 1 if sex == "Male" else 0,
                "smoking": 1 if smoking == "Yes" else 0,
            }
            input_df = pd.DataFrame([patient], columns=saved_features)
            model = artifact["model"]
            probabilities = model.predict_proba(input_df)[0]
            class_probabilities = {
                int(label): float(probability)
                for label, probability in zip(model.classes_, probabilities)
            }

            # Class 0 means no death event was recorded during the dataset's follow-up.
            survival_estimate = class_probabilities.get(0)
            if survival_estimate is None:
                st.error("The saved model does not contain the expected survival class (class 0).")
                st.stop()

            survival_score = round(survival_estimate * 100)
            st.divider()
            st.subheader("Estimated Survival")
            st.metric("Model-estimated chance of survival", f"{survival_score} / 100")
            st.progress(max(0.0, min(1.0, survival_estimate)))
            st.caption(
                "This is the model's estimate for class 0 (no death event recorded during the "
                "dataset's follow-up period). It is not a guaranteed or time-specific medical prognosis."
            )

        except (ValueError, TypeError, KeyError) as exc:
            st.error(f"Please check the entered details. Prediction could not be completed: {exc}")
        except Exception as exc:
            st.error(f"Prediction error: {exc}")

    st.divider()
    st.caption(
        "Academic project by Sahil | Roll No. 16. For educational use only. This application is "
        "not a medical diagnosis tool and must not be used for treatment or healthcare decisions. "
        "The dataset is small and contains historical records of patients with heart failure. "
        "The displayed score is a machine-learning estimate, not an actual personal survival chance. "
        "Please consult a qualified healthcare professional for medical advice."
    )


if __name__ == "__main__":
    main()
