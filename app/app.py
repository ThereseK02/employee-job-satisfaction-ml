import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
import numpy as np

# ---------------------------------------------------
# Landing Page State
# ---------------------------------------------------

if "show_predictor" not in st.session_state:
    st.session_state.show_predictor = False

# ---------------------------------------------------
# Landing Page
# ---------------------------------------------------
if not st.session_state.show_predictor:
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f0e7d1, #e3d4b2); padding:3rem; border-radius:18px; text-align:center; box-shadow:0 12px 30px rgba(0,0,0,0.18); width:100%; max-width:950px; margin:auto;">
      <h2 style="color:#375a7f; margin-bottom:0.5rem;">
    Welcome to
</h2>

<h1 style="font-size:2.8rem; color:#0f2747; margin-bottom:0.75rem;">
    Employee Job Satisfaction Predictor
</h1>
            <h3 style="font-weight:500; color:#334155; margin-bottom:1.5rem;">
               Predict employee job satisfaction using machine learning.
            </h3>
            <p style="font-size:1.1rem; line-height:1.8; max-width:820px; margin:auto; color:#1f2937;">
                An interactive application that estimates employee job satisfaction
                using information about the workplace, workload, work-life balance, and employee wellbeing.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        div.stButton > button {
            background-color: #7cb342;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1.8rem;
            font-weight: 600;
            font-size: 1rem;
        }

        div.stButton > button:hover {
            background-color: #689f38;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("Start Prediction"):
        st.session_state.show_predictor = True
        st.rerun()

    st.stop()
# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
    st.success("Trained model loaded successfully.")
except FileNotFoundError as e:
    model_loaded = False
    st.error(f"Model file not found: {MODEL_PATH}")
except Exception as e:
    model_loaded = False
    st.error("Model file was found, but it could not be loaded.")
    st.exception(e)

st.divider()

st.markdown("""
<style>

/* Slider active track */
.stSlider > div > div > div > div {
    background-color: #7cb342;
}

/* Slider handle */
.stSlider [role="slider"] {
    background-color: #1f77b4;
    border: 2px solid #1f77b4;
}

/* Optional: button styling */
div.stButton > button:first-child {
    background-color: #7cb342;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.7rem 1.8rem;
    font-weight: 600;
    font-size: 1rem;
}

div.stButton > button:first-child:hover {
    background-color: #689f38;
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

st.subheader("Demographic Information")
gender = st.selectbox("Gender", ["Female", "Male", "Other"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
education_level = st.selectbox("Education Level",["High School", "Bachelor", "Master", "PhD"])

st.subheader("Job Information")
department = st.selectbox("Department", ["HR", "IT", "Finance", "Marketing", "Sales", "Legal", "Operations"])
employment_type = st.selectbox("Employment Type", ["Full-Time", "Part-Time"])
job_level = st.slider("Job Level", 1, 5, 2)
experience = st.slider("Experience", 0, 40, 5)
num_companies = st.slider("Number of Companies Worked", 0, 15, 2)
team_size = st.slider("Team Size", 1, 50, 8)
num_reports = st.slider("Number of Direct Reports", 0, 20, 1)
have_ot = st.selectbox("Works Overtime?", ["No", "Yes"])

st.subheader("Workplace Wellbeing")
work_environment = st.slider("Work Environment", 1, 5, 3)
work_life_balance = st.slider("Work-Life Balance", 1, 5, 3)
workload = st.slider("Workload", 1, 5, 3)
stress_level = st.slider("Stress Level", 1, 5, 3)
sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
physical_activity_hours = st.slider("Physical Activity Hours", 0.0, 20.0, 3.0, 0.5)

st.subheader("Commute Information")
commute_mode = st.selectbox("Commute Mode", ["Car", "Public Transport", "Motorbike", "Walk"])
commute_distance = st.slider("Commute Distance", 0, 100, 10)

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button("Predict Job Satisfaction"):

    if model_loaded:
        input_data = {
            "JobLevel": job_level,
            "Experience": experience,
            "WLB": work_life_balance,
            "WorkEnv": work_environment,
            "PhysicalActivityHours": physical_activity_hours,
            "Workload": workload,
            "Stress": stress_level,
            "SleepHours": sleep_hours,
            "CommuteDistance": commute_distance,
            "NumCompanies": num_companies,
            "TeamSize": team_size,
            "NumReports": num_reports,
            "EduLevel": {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}[education_level],
            "haveOT": 1 if have_ot == "Yes" else 0,

            "Gender_Male": 1 if gender == "Male" else 0,
            "Gender_Other": 1 if gender == "Other" else 0,

            "MaritalStatus_Married": 1 if marital_status == "Married" else 0,
            "MaritalStatus_Single": 1 if marital_status == "Single" else 0,

            "Dept_Finance": 1 if department == "Finance" else 0,
            "Dept_HR": 1 if department == "HR" else 0,
            "Dept_IT": 1 if department == "IT" else 0,
            "Dept_Legal": 1 if department == "Legal" else 0,
            "Dept_Marketing": 1 if department == "Marketing" else 0,
            "Dept_Operations": 1 if department == "Operations" else 0,
            "Dept_Sales": 1 if department == "Sales" else 0,

            "EmpType_Full-Time": 1 if employment_type == "Full-Time" else 0,
            "EmpType_Part-Time": 1 if employment_type == "Part-Time" else 0,

            "CommuteMode_Car": 1 if commute_mode == "Car" else 0,
            "CommuteMode_Motorbike": 1 if commute_mode == "Motorbike" else 0,
            "CommuteMode_Public Transport": 1 if commute_mode == "Public Transport" else 0,
            "CommuteMode_Walk": 1 if commute_mode == "Walk" else 0,
        }

        input_df = pd.DataFrame([input_data])
        input_df = input_df[model.feature_names_in_]

        prediction = model.predict(input_df)[0]

        st.success("Trained model loaded successfully.")

        st.info(f"Predicted Job Satisfaction Level: {prediction}/5")

        if prediction >= 4:
            st.success("High Job Satisfaction")
        elif prediction == 3:
            st.warning("Moderate Job Satisfaction")
        else:
            st.error("Low Job Satisfaction")

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
st.caption("Employee Job Satisfaction Prediction System")