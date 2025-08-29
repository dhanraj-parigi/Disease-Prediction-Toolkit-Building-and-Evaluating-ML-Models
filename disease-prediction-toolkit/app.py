import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("disease_model.pkl")

st.set_page_config(page_title="Disease Prediction Toolkit", layout="centered")

st.title("🩺 Disease Prediction Toolkit")
st.write("Enter patient details to predict the risk of heart disease.")

# --- Collect user input (aligned with dataset) ---
def user_input_features():
    age = st.slider("Age", 20, 80, 40)

    # Sex -> dataset expects numeric (1 = male, 0 = female)
    sex_map = {"Male": 1, "Female": 0}
    sex_label = st.selectbox("Sex", list(sex_map.keys()))
    sex = sex_map[sex_label]

    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.slider("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.slider("Max Heart Rate Achieved", 70, 210, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])

    # Thal -> dataset values: 1 = fixed defect, 2 = normal, 3 = reversible defect
    thal_map = {"Fixed Defect (1)": 1, "Normal (2)": 2, "Reversible Defect (3)": 3}
    thal_label = st.selectbox("Thalassemia", list(thal_map.keys()))
    thal = thal_map[thal_label]

    # Must match training dataset column names
    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Optional: show user’s input
st.write("### Input Data Preview", input_df)

# --- Prediction ---
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High risk of disease! (Probability: {prediction_proba:.2f})")
        else:
            st.success(f"✅ Low risk of disease. (Probability: {prediction_proba:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")
