
import streamlit as st
import joblib
from pathlib import Path
import numpy as np
# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Employee Job Satisfaction Predictor",
    page_icon="📊",
    layout="centered"
)

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = e

if not model_loaded:
    st.warning(f"""
    Trained model file not found.

    Please make sure this file exists:
    {MODEL_PATH}
    """)

st.markdown("""
<style>

/* Slider active track */
.stSlider > div > div > div > div {
    background-color: #1f77b4;
}

/* Slider handle */
.stSlider [role="slider"] {
    background-color: #1f77b4;
    border: 2px solid #1f77b4;
}

/* Optional: button styling */
.stButton > button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    background-color: #155a8a;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Title and Description
# ---------------------------------------------------

st.title("📊 Employee Job Satisfaction Prediction")

st.markdown("""
This application predicts employee job satisfaction using a trained Random Forest machine learning pipeline developed through preprocessing, class imbalance handling, hyperparameter optimization, and explainable AI techniques.
""")

# ---------------------------------------------------
# User Inputs
# ---------------------------------------------------

st.header("Employee Information")

age = st.slider("Age", 18, 65, 30)

work_environment = st.slider("Work Environment", 1, 5, 3)

work_life_balance = st.slider("Work-Life Balance", 1, 5, 3)
workload = st.slider("Workload", 1, 5, 3)

stress_level = st.slider("Stress Level", 1, 5, 3)

sleep_quality = st.slider("Sleep Quality", 1, 5, 3)

years_at_company = st.slider("Years at Company", 0, 40, 5)
# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button("Predict Job Satisfaction"):

    if model_loaded:
        st.success("Trained model loaded successfully.")

        st.info("""
        The model expects 31 processed input features based on the original training pipeline.

        This starter Streamlit interface currently includes selected key features for demonstration.
        Future improvements will integrate the complete preprocessing pipeline and full feature engineering workflow used during model training.
        """)

    else:
        st.warning(f"""
        Trained model file not found.

        Please make sure this file exists:
        {MODEL_PATH}
        """)
# ---------------------------------------------------
# Footer
# ---------------------------------------------------

st.markdown("---")

st.markdown(
    "Developed by Therese Kabayanja | Machine Learning Engineer • Data Science • Software Development"
)