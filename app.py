import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sidebar menu
st.sidebar.title("Prediction of Disease Outbreaks System")
menu = st.sidebar.radio("Select Prediction Type", [
                        "Diabetes Prediction", "Heart Disease Prediction", "Parkinson’s Prediction"])

# Preloading datasets
diabetes_data = pd.read_csv("diabetes.csv")
heart_data = pd.read_csv("heart.csv")
parkinsons_data = pd.read_csv("parkinsons.csv")

# Function to train and predict


def train_and_predict(user_input, target_column, data):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    user_input = np.array(user_input).reshape(1, -1)
    user_input = scaler.transform(user_input)
    prediction = model.predict(user_input)

    return "Yes, you have the disease." if prediction[0] == 1 else "No, you do not have the disease."


if menu == "Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    with st.form("diabetes_form"):
        pregnancies = st.number_input("Number of Pregnancies", value=0)
        glucose = st.number_input("Glucose Level", value=0)
        blood_pressure = st.number_input("Blood Pressure value", value=0)
        skin_thickness = st.number_input("Skin Thickness value", value=0)
        insulin = st.number_input("Insulin Level", value=0)
        bmi = st.number_input("BMI value", value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function value", value=0.0)
        age = st.number_input("Age of the Person", value=0)

        submit = st.form_submit_button("Diabetes Test Result")

        if submit:
            user_input = [pregnancies, glucose, blood_pressure,
                          skin_thickness, insulin, bmi, dpf, age]
            prediction = train_and_predict(
                user_input, "Outcome", diabetes_data)
            st.write(prediction)

elif menu == "Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")
    with st.form("heart_disease_form"):
        age = st.number_input("Age", value=0)
        sex = st.number_input("Sex", value=0)
        cp = st.number_input("Chest Pain types", value=0)
        trestbps = st.number_input("Resting Blood Pressure", value=0)
        chol = st.number_input("Serum Cholesterol in mg/dl", value=0)
        fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl", value=0)
        restecg = st.number_input(
            "Resting Electrocardiographic results", value=0)
        thalach = st.number_input("Maximum Heart Rate achieved", value=0)
        exang = st.number_input("Exercise Induced Angina", value=0)
        oldpeak = st.number_input("ST depression induced by exercise", value=0)
        slope = st.number_input(
            "Slope of the peak exercise ST segment", value=0)
        ca = st.number_input("Major vessels colored by fluoroscopy", value=0)
        thal = st.number_input(
            "Thal: 0 = normal; 1 = fixed defect; 2 = reversible defect", value=0)

        submit = st.form_submit_button("Heart Disease Test Result")

        if submit:
            user_input = [age, sex, cp, trestbps, chol, fbs,
                          restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction = train_and_predict(user_input, "target", heart_data)
            st.write(prediction)

elif menu == "Parkinson’s Prediction":
    st.title("Parkinson’s Disease Prediction using ML")
    with st.form("parkinson_form"):
        mdvp_fo = st.number_input("MDVP Fo (Hz)", value=0.0)
        mdvp_fhi = st.number_input("MDVP Fhi (Hz)", value=0.0)
        mdvp_flo = st.number_input("MDVP Flo (%)", value=0.0)
        mdvp_jitter = st.number_input("MDVP Jitter (Abs)", value=0.0)
        shimmer = st.number_input("Shimmer", value=0.0)
        hnr = st.number_input("HNR", value=0.0)
        rpde = st.number_input("RPDE", value=0.0)
        dfa = st.number_input("DFA", value=0.0)
        spread1 = st.number_input("Spread1", value=0.0)
        spread2 = st.number_input("Spread2", value=0.0)
        d2 = st.number_input("D2", value=0.0)
        ppe = st.number_input("PPE", value=0.0)

        submit = st.form_submit_button("Parkinson's Test Result")

        if submit:
            user_input = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter,
                          shimmer, hnr, rpde, dfa, spread1, spread2, d2, ppe]
            prediction = train_and_predict(user_input, "status", parkinsons_data)
            st.write("Prediction:", "Yes, you have the disease." if prediction == 1 else "No, you do not have the disease.")
