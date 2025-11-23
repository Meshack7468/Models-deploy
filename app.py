

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Breast Cancer Image Classification":{
            "model":load_model("models/cnn_model.keras")
        },
        "Molecular Subtype Prediction": {
            "model": pickle.load(open("models/molecular_subtype_model.pkl", "rb")),
            "le": pickle.load(open("models/molecular_le.pkl", "rb"))
        },
        "Survival Status Prediction": {
            "model": pickle.load(open("models/survival_status_model.pkl", "rb"))
        },
        "Vital Status Prediction": {
            "model": pickle.load(open("models/vital_status_model.pkl", "rb")),
            "le": pickle.load(open("models/vitalstatus_le.pkl", "rb"))
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
    "This app uses patient details, tumor information, and medical imaging to make predictions "
    "related to breast cancer diagnosis and prognosis."
)

# --------------------
# Model Selection
# -------------------
st.sidebar.header("Select Prediction Type")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction:",
    [
        "Breast Cancer Image Classification",
        "Molecular Subtype Prediction",
        "Survival Status Prediction",
        "Vital Status Prediction"
    ]
)

selected_model_info = models[model_choice]
# ------------------------------
# IMAGE CLASSIFICATION PAGE
# -------------------------------

if model_choice == "Breast Cancer Image Classification":

    st.markdown("##  Breast Cancer Image Classification")
    st.info("Upload a microscopic image of breast tissue. The CNN model will classify it as *Benign or Malignant*.")

    uploaded_file = st.file_uploader(
        "Upload Microscopic Breast Tissue Image (JPG/PNG/JPEG)",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="Uploaded Image", width=300)

            img = img.resize((224, 224))
            img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

            cnn_model = selected_model_info["model"]
            prediction = cnn_model.predict(img_array)[0][0]

            if prediction > 0.5:
                result = "Malignant"
                color = "red"
            else:
                result = "Benign"
                color = "green"

            st.markdown(
                f"### Prediction: *<span style='color:{color}'>{result}</span>*",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Image classification error: {e}")

    st.stop()  # Prevents clinical inputs from showing

# ---------------------------------
# CLINICAL PREDICTION MODELS
# ----------------------------------

st.markdown(f"###  **{model_choice}**")
# Add informative model descriptions
if model_choice == "Molecular Subtype Prediction":
    st.info("""
    Predicts the molecular subtype of breast cancer (e.g., Luminal A, Luminal B, HER2-enriched, or Basal).  
    and recommends personalized treatment strategies.
    """)
elif model_choice == "Survival Status Prediction":
    st.info("""
    Predicts whether a patient is *Living* or *Deceased* based on clinical and pathological factors to support prognosis purposes.
    """)
elif model_choice == "Vital Status Prediction":
    st.info("""
    Predicts whether the patient is Living, Died of Disease, or Died of Other Causes.  
    
    """)

st.write("Provide patient and tumor information below to get predictions. These predictions come from statistical models and should be considered together with your doctor’s advice.")


# ------------------------
# Input Features
# ------------------------
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
            if "LUMA" in pred_label:
                st.info("Recommended Treatment: Hormone therapy, Radiotherapy.")
            elif "LUMB" in pred_label:
                st.info("Recommended Treatment: Hormone therapy, Chemotherapy, Radiotherapy.")
            elif "HER2" in pred_label:
                st.info("Recommended Treatment: Chemotherapy, Radiotherapy.")
            elif "BASAL" in pred_label:
                st.warning("Recommended Treatment: Chemotherapy, Radiotherapy.")
            elif "CLAUDIN-LOW" in pred_label:
                st.info("Recommended Treatment: Chemotherapy, Radiotherapy.")

            else:
                st.info("Recommended Treatment: Radiotherapy and chemotherapy.")

       
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

        # --------------------
        # Vital Status Prediction
        # --------------------


        elif model_choice == "Vital Status Prediction":
            model = selected_model_info["model"]
            pred = model.predict(input_data)[0]

            # Interpret prediction codes
            if pred == 0:
                status = "DIED OF THE DISEASE"
            elif pred == 1:
                status = "DIED OF OTHER CAUSES"
            elif pred == 2:
                status = "LIVING"
            

            st.success(f"Predicted Vital Status: {status}")


            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Input preview:", input_data)





     
     
     
