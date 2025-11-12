

#====================import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_models():
    models = {
        "Molecular Subtype Model": {
            "model": joblib.load("models/molecular_subtype_model.joblib"),
            "le": joblib.load("models/molecular_le.joblib")
        },
        "Survival Status Model": {
            "model": joblib.load("models/survival_status_model.joblib")
        },
        # "Vital Status Model": {
        #     "model": joblib.load("models/vital_status_model.joblib")
        # }
    }
    return models


models = load_models()

# -----------------------------
# Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Breast Cancer Prediction Suite", layout="wide")
st.title("🧬 Breast Cancer Multi-Model Prediction Suite")

# -----------------------------
# Select Model
# -----------------------------
st.sidebar.header("⚙ Select Model")
model_choice = st.sidebar.selectbox(
    "Choose a model for prediction:",
    list(models.keys())
)

selected_model_info = models[model_choice]

st.markdown(f"### Using **{model_choice}**")
st.info("Provide patient information below to get predictions. All models use the same input features.")

# -----------------------------
# Input Features (same for all models)
# -----------------------------
st.subheader("🧫 Patient & Tumor Information")

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
        "3-Gene classifier subtype",
        ["ER+/HER2- LOW PROLIF", "ER+/HER2- HIGH PROLIF", "ER-/HER2-", "HER2+"]
    )
    pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
    lymph_nodes = st.number_input("Lymph nodes examined positive", 0, 50, 1)
    cluster = st.selectbox("Integrative Cluster", [str(i) for i in range(1, 11)])
    hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
    npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 3.5)
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

st.write("### Input Summary")
st.dataframe(input_data)

# -----------------------------
# Treatment Recommendations
# -----------------------------
treatment_recommendations = {
    "Luminal A": "💊 **Responds best to hormone therapy.**",
    "Luminal B": "💉 **Best managed with combined hormone and chemotherapy.**",
    "HER2-enriched": "🎯 **Requires targeted therapy (e.g., trastuzumab, pertuzumab).** These significantly improve survival beyond surgery alone.",
    "Basal-like": "⚡ **Aggressive tumors. Require intensive targeted systemic therapy — surgery alone isn’t enough.**",
    "Claudin-low": "🧬 **Benefit most from biological treatments. Limited response to surgery or chemotherapy alone.**"
}

# -----------------------------
# Prediction
# -----------------------------
st.markdown("---")

if st.button("🔮 Predict"):
    try:
        # Molecular Subtype Model
        if model_choice == "Molecular Subtype Model":
            model = selected_model_info["model"]
            le = selected_model_info["le"]

            pred_numeric = model.predict(input_data)[0]
            pred_label = le.inverse_transform([pred_numeric])[0]

            st.success(f"🧠 Predicted Molecular Subtype: **{pred_label}**")

            # Personalized treatment guidance
            treatment = treatment_recommendations.get(
                pred_label,
                "No specific treatment guidance available for this subtype."
            )

            st.markdown("---")
            st.subheader("💡 Personalized Treatment Recommendation")
            st.info(treatment)

        # Survival Status Model
        elif model_choice == "Survival Status Model":
            model = selected_model_info["model"]
            pred = model.predict(input_data)[0]
            output = "🩺 **LIVING**" if pred == 0 else "⚰️ **DECEASED**"

            st.success(f"Predicted Survival Status: {output}")

        # Vital Status Model (optional)
        # elif model_choice == "Vital Status Model":
        #     model = selected_model_info["model"]
        #     pred = model.predict(input_data)[0]
        #     st.success(f"Predicted Vital Status: **{pred}**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("🔍 Input preview:", input_data)

#================================================================================================================
# import streamlit as st
# import joblib
# import pandas as pd

# # -----------------------------
# # Load Molecular Subtype Model (full pipeline) + LabelEncoder
# # -----------------------------
# @st.cache_resource
# def load_model_and_encoder():
#     model = joblib.load("models/molecular_subtype_model.joblib")  # full pipeline
#     le = joblib.load("models/molecular_le.joblib")  # LabelEncoder used during training
#     return model, le

# model, le = load_model_and_encoder()

# # -----------------------------
# # Streamlit Page Setup
# # -----------------------------
# st.set_page_config(page_title="Molecular Subtype Prediction", layout="wide")
# st.title("🧬 Molecular Subtype Prediction for Breast Cancer")

# # -----------------------------
# # Input Features
# # -----------------------------
# st.subheader("🔢 Patient & Tumor Information")

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age at Diagnosis", 0, 120, 50)
#     surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
#     er_status = st.selectbox("ER Status", ["Positive", "Negative"])
#     her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
#     grade = st.selectbox("Neoplasm Histologic Grade", ["Grade 1", "Grade 2", "Grade 3"])
#     tmb = st.number_input("TMB (nonsynonymous)", 0.0, 1000.0, 10.0)
#     stage = st.selectbox("Tumor Stage", ["0","1", "2", "3", "4"])

# with col2:
#     subtype_3gene = st.selectbox(
#         "3-Gene classifier subtype", 
#         ["ER+/HER2- LOW PROLIF", "ER+/HER2- HIGH PROLIF", "ER-/HER2-", "HER2+"]
#     )
#     pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
#     lymph_nodes = st.number_input("Lymph nodes examined positive", 0, 50, 1)
#     cluster = st.selectbox("Integrative Cluster", [str(i) for i in range(1, 11)])
#     hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
#     npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 3.5)
#     histologic_subtype = st.selectbox(
#         "Tumor Other Histologic Subtype",
#         ["Ductal", "Lobular", "Medullary", "Mucinous", "Tubular/Cribriform", "Mixed", "Other"]
#     )

# # -----------------------------
# # Prepare Input Data
# # -----------------------------
# input_dict = {
#     "Age at Diagnosis": [age],
#     "Type of Breast Surgery": [surgery],
#     "ER Status": [er_status],
#     "HER2 Status": [her2_status],
#     "Neoplasm Histologic Grade": [grade],
#     "TMB (nonsynonymous)": [tmb],
#     "Tumor Stage": [stage],
#     "3-Gene classifier subtype": [subtype_3gene],
#     "PR Status": [pr_status],
#     "Lymph nodes examined positive": [lymph_nodes],
#     "Integrative Cluster": [cluster],
#     "Hormone Therapy": [hormone_therapy],
#     "Nottingham prognostic index": [npi],
#     "Tumor Other Histologic Subtype": [histologic_subtype]
# }

# input_data = pd.DataFrame.from_dict(input_dict)

# # -----------------------------
# # Prediction using LabelEncoder
# # -----------------------------
# if st.button("🔮 Predict Molecular Subtype"):
#     try:
#         # Predict numeric label
#         pred_numeric = model.predict(input_data)[0]

#         # Convert numeric label back to string
#         pred_label = le.inverse_transform([pred_numeric])[0]

#         st.success(f"✅ Predicted Molecular Subtype: **{pred_label}**")
#     except Exception as e:
#         st.error(f"⚠ Prediction error: {e}")
#         st.write("🔎 Input preview:", input_data)
#=========================================================================================================
#========================================================================================================================

# import streamlit as st
# import joblib
# import pandas as pd
# import numpy as np
# import os

# # -----------------------------------------------------
# # Load Models
# # -----------------------------------------------------
# @st.cache_resource
# def load_models():
#     models = {
#         "Molecular Model": joblib.load("models/molecular_subtype_model.joblib"),
#         "Survival Model": joblib.load("models/survival_status_model.joblib"),
#         "Vital Status Model": joblib.load("models/vital_status_model.joblib")
#     }
#     return models

# models = load_models()

# # -----------------------------------------------------
# # Streamlit Page Setup
# # -----------------------------------------------------
# st.set_page_config(page_title="Breast Cancer Prediction Suite", layout="wide")
# st.title("🧬 Breast Cancer Multi-Model Prediction Suite")


# st.sidebar.header("⚙ Select Model")
# model_choice = st.sidebar.selectbox(
#     "Choose a model for prediction:",
#     list(models.keys())
# )
# model = models[model_choice]

# st.markdown(f"### Using {model_choice}")
# st.info("Provide patient information below to get predictions. All models use the same input features.")

# # -----------------------------------------------------
# # Input Form
# # -----------------------------------------------------
# st.subheader("🔢 Input Patient & Tumor Information")

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age at Diagnosis", min_value=0, max_value=120, value=50)
#     surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Lumpectomy", "Other"])
#     er_status = st.selectbox("ER Status", ["Positive", "Negative"])
#     her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
#     grade = st.selectbox("Neoplasm Histologic Grade", ["Grade 1", "Grade 2", "Grade 3", "Unknown"])
#     tmb = st.number_input("TMB (nonsynonymous)", min_value=0.0, max_value=1000.0, value=10.0)
#     stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"])

# with col2:
#     subtype = st.selectbox("3-Gene classifier subtype", ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like", "Unknown"])
#     pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
#     lymph_nodes = st.number_input("Lymph nodes examined positive", min_value=0, max_value=50, value=1)
#     cluster = st.selectbox("Integrative Cluster", [str(i) for i in range(1, 11)])
#     hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
#     npi = st.number_input("Nottingham prognostic index", min_value=0.0, max_value=10.0, value=3.5)
#     histologic_subtype = st.selectbox("Tumor Other Histologic Subtype", ["Ductal", "Lobular", "Medullary", "Mucinous", "Other"])

# # -----------------------------------------------------
# # Prepare Input Data
# # -----------------------------------------------------
# input_dict = {
#     "Age at Diagnosis": [age],
#     "Type of Breast Surgery": [surgery],
#     "ER Status": [er_status],
#     "HER2 Status": [her2_status],
#     "Neoplasm Histologic Grade": [grade],
#     "TMB (nonsynonymous)": [tmb],
#     "Tumor Stage": [stage],
#     "3-Gene classifier subtype": [subtype],
#     "PR Status": [pr_status],
#     "Lymph nodes examined positive": [lymph_nodes],
#     "Integrative Cluster": [cluster],
#     "Hormone Therapy": [hormone_therapy],
#     "Nottingham prognostic index": [npi],
#     "Tumor Other Histologic Subtype": [histologic_subtype]
# }

# input_data = pd.DataFrame.from_dict(input_dict)

# st.write("### Input Summary")
# st.dataframe(input_data)

# # -----------------------------------------------------
# # Encoding Maps (for categorical columns)
# # -----------------------------------------------------
# encoding_maps = {
#     "Type of Breast Surgery": {"Mastectomy": 0, "Lumpectomy": 1, "Other": 2},
#     "ER Status": {"Negative": 0, "Positive": 1},
#     "HER2 Status": {"Negative": 0, "Positive": 1},
#     "Neoplasm Histologic Grade": {"Unknown": 0, "Grade 1": 1, "Grade 2": 2, "Grade 3": 3},
#     "Tumor Stage": {"Unknown": 0, "Stage I": 1, "Stage II": 2, "Stage III": 3, "Stage IV": 4},
#     "3-Gene classifier subtype": {
#         "Unknown": 0, "Luminal A": 1, "Luminal B": 2,
#         "HER2-enriched": 3, "Basal-like": 4
#     },
#     "PR Status": {"Negative": 0, "Positive": 1},
#     "Integrative Cluster": {str(i): i for i in range(1, 11)},
#     "Hormone Therapy": {"No": 0, "Yes": 1},
#     "Tumor Other Histologic Subtype": {
#         "Other": 0, "Ductal": 1, "Lobular": 2, "Medullary": 3, "Mucinous": 4
#     }
# }

# # Apply encodings safely
# for col, mapping in encoding_maps.items():
#     if col in input_data.columns:
#         input_data[col] = input_data[col].map(mapping).fillna(0)

# # Convert to numeric safely
# input_data = input_data.apply(pd.to_numeric, errors="coerce").fillna(0)

# # -----------------------------------------------------
# # Prediction
# # -----------------------------------------------------
# st.markdown("---")

# if st.button("🔮 Predict"):
#     try:
#         prediction = model.predict(input_data)
#         if isinstance(prediction, (list, np.ndarray)):
#             prediction = prediction[0]
#         st.success(f"✅ Predicted Output: {prediction}")
#     except Exception as e:
#         st.error(f"⚠ Prediction error: {e}")
#         st.write("🧩 Debug info:", input_data.dtypes)
#         st.write("🔎 Input preview:", input_data)

# import streamlit as st
# import joblib
# import pandas as pd

# # -----------------------------------------------------
# # Load Models
# # -----------------------------------------------------
# @st.cache_resource
# def load_models():
#     models = {
#         "Molecular Model": joblib.load("models/molecular_subtype_model.joblib"),
#         "Survival Model": joblib.load("models/survival_status_model.joblib"),
#         "Vital Status Model": joblib.load("models/vital_status_model.joblib")
#     }
#     return models

# models = load_models()

# # -----------------------------------------------------
# # Streamlit App Interface
# # -----------------------------------------------------
# st.set_page_config(page_title="Breast Cancer Prediction Suite", layout="wide")
# st.title("🧬 Breast Cancer Multi-Model Prediction Suite")

# st.sidebar.header("⚙️ Select Model")
# model_choice = st.sidebar.selectbox(
#     "Choose a model for prediction:",
#     list(models.keys())
# )
# model = models[model_choice]

# st.markdown(f"### Using **{model_choice}**")

# st.info("Provide patient information below to get predictions. All models use the same input features.")

# # -----------------------------------------------------
# # Input Features
# # -----------------------------------------------------
# st.subheader("🔢 Input Patient & Tumor Information")

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age at Diagnosis", min_value=0, max_value=120, value=50)
#     surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Lumpectomy", "Other"])
#     er_status = st.selectbox("ER Status", ["Positive", "Negative"])
#     her2_status = st.selectbox("HER2 Status", ["Positive", "Negative"])
#     grade = st.selectbox("Neoplasm Histologic Grade", ["Grade 1", "Grade 2", "Grade 3", "Unknown"])
#     tmb = st.number_input("TMB (nonsynonymous)", min_value=0.0, max_value=1000.0, value=10.0)
#     stage = st.selectbox("Tumor Stage", ["Stage I", "Stage II", "Stage III", "Stage IV", "Unknown"])

# with col2:
#     subtype = st.selectbox("3-Gene classifier subtype", ["Luminal A", "Luminal B", "HER2-enriched", "Basal-like", "Unknown"])
#     pr_status = st.selectbox("PR Status", ["Positive", "Negative"])
#     lymph_nodes = st.number_input("Lymph nodes examined positive", min_value=0, max_value=50, value=1)
#     cluster = st.selectbox("Integrative Cluster", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
#     hormone_therapy = st.selectbox("Hormone Therapy", ["Yes", "No"])
#     npi = st.number_input("Nottingham prognostic index", min_value=0.0, max_value=10.0, value=3.5)
#     histologic_subtype = st.selectbox("Tumor Other Histologic Subtype", ["Ductal", "Lobular", "Medullary", "Mucinous", "Other"])

# # -----------------------------------------------------
# # Prepare Input DataFrame
# # -----------------------------------------------------
# input_dict = {
#     "Age at Diagnosis": [age],
#     "Type of Breast Surgery": [surgery],
#     "ER Status": [er_status],
#     "HER2 Status": [her2_status],
#     "Neoplasm Histologic Grade": [grade],
#     "TMB (nonsynonymous)": [tmb],
#     "Tumor Stage": [stage],
#     "3-Gene classifier subtype": [subtype],
#     "PR Status": [pr_status],
#     "Lymph nodes examined positive": [lymph_nodes],
#     "Integrative Cluster": [cluster],
#     "Hormone Therapy": [hormone_therapy],
#     "Nottingham prognostic index": [npi],
#     "Tumor Other Histologic Subtype": [histologic_subtype]
# }

# input_data = pd.DataFrame.from_dict(input_dict)

# st.write("### Input Summary")
# st.dataframe(input_data)

# # -----------------------------------------------------
# # Prediction
# # -----------------------------------------------------
# if st.button("🔮 Predict"):
#     try:
#         prediction = model.predict(input_data)
#         st.success(f"**Predicted Output:** {prediction[0]}")
#     except Exception as e:
#         st.error(f"Prediction error: {e}")