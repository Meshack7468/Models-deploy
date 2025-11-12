

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Molecular Subtype Prediction": {
            "model": joblib.load("models/molecular_subtype_model.joblib"),
            "le": joblib.load("models/molecular_le.joblib")
        },
        "Survival Status Prediction": {
            "model": joblib.load("models/survival_status_model.joblib")
        },
        "Vital Status Prediction": {
            "model": joblib.load("models/vital_status_model.joblib")
        }
    }
    return models

models = load_models()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Breast Cancer Prediction Suite", layout="wide")
st.title("Breast Cancer Multi-Model Prediction Suite")

st.write(
    "This app uses patient and tumor details to predict breast cancer **molecular subtype**, "
    "**survival/vital status**, then recommend appropriate treatment approach."
)

# -----------------------------
# Model Selection
# -----------------------------
st.sidebar.header("Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction:",
    list(models.keys())
)

selected_model_info = models[model_choice]
st.markdown(f"###  **{model_choice}**")
st.info("Provide patient and tumor information below to get predictions. All models use the same input features.")

# -----------------------------
# Input Features
# -----------------------------
st.subheader("Patient & Tumor Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age at Diagnosis", 0, 120, 50)
    surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
    er_status = st.selectbox("ER Status", ["Positive", "Negative"])
    her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
    grade = st.selectbox("Neoplasm Histologic Grade", ["Grade 1", "Grade 2", "Grade 3"])
    tmb = st.number_input("TMB (nonsynonymous)", 0.0, 1000.0, 10.0)
    stage = st.selectbox("Tumor Stage", ["0", "1", "2", "3", "4"])

with col2:
    subtype_3gene = st.selectbox(
        "3-Gene Classifier Subtype",
        ["ER+/HER2- LOW PROLIF", "ER+/HER2- HIGH PROLIF", "ER-/HER2-", "HER2+"]
    )
    pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
    lymph_nodes = st.number_input("Lymph Nodes Examined Positive", 0, 50, 1)
    cluster = st.selectbox("Integrative Cluster", [str(i) for i in range(1, 11)])
    hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
    npi = st.number_input("Nottingham Prognostic Index", 0.0, 10.0, 3.5)
    histologic_subtype = st.selectbox(
        "Tumor Other Histologic Subtype",
        ["Ductal", "Lobular", "Medullary", "Mucinous", "Tubular/Cribriform", "Mixed", "Other"]
    )

# -----------------------------
# Prepare Input Data
# -----------------------------
input_dict = {
    "Age at Diagnosis": [age],
    "Type of Breast Surgery": [surgery],
    "ER Status": [er_status],
    "HER2 Status": [her2_status],
    "Neoplasm Histologic Grade": [grade],
    "TMB (nonsynonymous)": [tmb],
    "Tumor Stage": [stage],
    "3-Gene classifier subtype": [subtype_3gene],
    "PR Status": [pr_status],
    "Lymph nodes examined positive": [lymph_nodes],
    "Integrative Cluster": [cluster],
    "Hormone Therapy": [hormone_therapy],
    "Nottingham prognostic index": [npi],
    "Tumor Other Histologic Subtype": [histologic_subtype]
}

input_data = pd.DataFrame.from_dict(input_dict)

# -----------------------------
# Prediction & Treatment Guidance
# -----------------------------
if st.button("Predict"):
    try:
        # -----------------------------
        # Molecular Subtype Prediction
        # -----------------------------
        if model_choice == "Molecular Subtype Prediction":
            model = selected_model_info["model"]
            le = selected_model_info["le"]

            pred_numeric = model.predict(input_data)[0]
            pred_label = le.inverse_transform([pred_numeric])[0]
            st.success(f"Predicted Molecular Subtype: {pred_label}")

            # Treatment Guidance
            if "LumA" in pred_label:
                st.info("Recommended Treatment: hormone therapy, radiotherapy, and surgery.")
            elif "LumB" in pred_label:
                st.info("Recommended Treatment: hormone therapy, chemotherapy, and surgery.")
            elif "Her2" in pred_label:
                st.info("Recommended Treatment: targeted therapy and chemotherapy.")
            elif "Basal" in pred_label:
                st.warning("Recommended Treatment: chemotherapy, targeted systemic therapy, surgery, and radiotherapy.")
            else:
                st.info("Recommended Treatment: Surgery and chemotherapy.")

        # -----------------------------
        # Survival Status Prediction
        # -----------------------------
        elif model_choice == "Survival Status Prediction":
            model = selected_model_info["model"]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                pred = np.argmax(proba)
                confidence = proba[pred] * 100
                output = "DECEASED" if pred == 1 else "LIVING"

                st.success(f"Predicted Survival Status: {output}")
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

                
            else:
                pred = model.predict(input_data)[0]
                output = "DECEASED" if pred == 1 else "LIVING"
                st.success(f"Predicted Survival Status: {output}")

        # -----------------------------
        # Vital Status Prediction
        # -----------------------------
        elif model_choice == "Vital Status Prediction":
            model = selected_model_info["model"]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                pred = np.argmax(proba)
                confidence = proba[pred] * 100

                # Interpret prediction codes
                if pred == 0:
                    status = "Died of the disease"
                elif pred == 1:
                    status = "Died of other causes"
                elif pred == 2:
                    status = "Living"
                else:
                    status = "Unknown"

                st.success(f"Predicted Vital Status: {status}")
                st.metric(label="Model Confidence", value=f"{confidence:.1f}%")

                
            else:
                pred = model.predict(input_data)[0]
                if pred == 0:
                    status = "Died of the disease"
                elif pred == 1:
                    status = "Died of other causes"
                elif pred == 2:
                    status = "Living"
                else:
                    status = "Unknown"
                st.success(f"Predicted Vital Status: {status}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Input preview:", input_data)
