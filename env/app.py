import streamlit as st
import numpy as np
import pickle

with open("diabetes.pkl", "rb") as file:
    diabetes_model = pickle.load(file)

with open("heart.pkl", "rb") as file:
    heart_disease_model = pickle.load(file)

with open("parkinsons.pkl", "rb") as file:
    parkinsons_model = pickle.load(file)


def diabetes_prediction():
    st.title("Diabetes Prediction using ML")

    pregnancies = st.text_input("Number of Pregnancies", key="pregnancies")
    glucose = st.text_input("Glucose Level", key="glucose")
    blood_pressure = st.text_input("Blood Pressure", key="blood_pressure")
    skin_thickness = st.text_input("Skin Thickness", key="skin_thickness")
    insulin = st.text_input("Insulin Level", key="insulin")
    bmi = st.text_input("BMI", key="bmi")
    diabetes_pedigree = st.text_input(
        "Diabetes Pedigree Function", key="diabetes_pedigree")
    age = st.text_input("Age", key="age")

    if st.button("Diabetes Test Result", key="diabetes_test_button"):
        try:

            input_data = np.array([float(pregnancies), float(glucose), float(blood_pressure),
                                   float(skin_thickness), float(
                                       insulin), float(bmi),
                                   float(diabetes_pedigree), float(age)]).reshape(1, -1)

            prediction = diabetes_model.predict(input_data)

            if prediction[0] == 1:
                st.error("The model predicts that the person **has diabetes.**")
            else:
                st.success(
                    "The model predicts that the person **does not have diabetes.**")

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")


def heart_disease_prediction():
    st.title("Heart Disease Prediction using ML")

    age = st.text_input("Age", key="heart_age")
    sex = st.text_input("Sex (1 = Male, 0 = Female)", key="heart_sex")
    cp = st.text_input("Chest Pain Type (0-3)", key="heart_cp")
    trestbps = st.text_input("Resting Blood Pressure", key="heart_trestbps")
    chol = st.text_input("Cholesterol Level", key="heart_chol")
    fbs = st.text_input(
        "Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", key="heart_fbs")
    restecg = st.text_input("Resting ECG Result (0-2)", key="heart_restecg")
    thalach = st.text_input("Max Heart Rate Achieved", key="heart_thalach")
    exang = st.text_input(
        "Exercise-Induced Angina (1 = Yes, 0 = No)", key="heart_exang")
    oldpeak = st.text_input(
        "ST Depression Induced by Exercise", key="heart_oldpeak")
    slope = st.text_input(
        "Slope of Peak Exercise ST Segment (0-2)", key="heart_slope")
    ca = st.text_input("Number of Major Vessels (0-4)", key="heart_ca")
    thal = st.text_input("Thalassemia (0-3)", key="heart_thal")

    if st.button("Heart Disease Test Result", key="heart_test_button"):
        try:
            input_data = np.array([float(age), float(sex), float(cp), float(trestbps),
                                   float(chol), float(fbs), float(
                                       restecg), float(thalach),
                                   float(exang), float(oldpeak), float(slope), float(ca), float(thal)]).reshape(1, -1)

            prediction = heart_disease_model.predict(input_data)

            if prediction[0] == 1:
                st.error(
                    "The model predicts that the person **has heart disease.**")
            else:
                st.success(
                    "The model predicts that the person **does not have heart disease.**")

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")


def parkinsons_prediction():
    st.title("Parkinson's Disease Prediction using ML")

    mdvp_1 = st.text_input("MDVP (Hz)", key="mdvp_hz_1")
    mdvp_2 = st.text_input("MDVP (Hz)", key="mdvp_hz_2")
    mdvp_percent = st.text_input("MDVP (%)", key="mdvp_percent")
    mdvp_abs = st.text_input("MDVP (Abs)", key="mdvp_abs")
    mdvp_dB = st.text_input("MDVP (dB)", key="mdvp_db")
    jitter = st.text_input("Jitter", key="jitter")
    shimmer_1 = st.text_input("Shimmer", key="shimmer_1")
    shimmer_2 = st.text_input("Shimmer", key="shimmer_2")
    hnr = st.text_input("HNR", key="hnr")
    rpde = st.text_input("RPDE", key="rpde")
    dfa = st.text_input("DFA", key="dfa")
    spread1 = st.text_input("Spread1", key="spread1")
    spread2 = st.text_input("Spread2", key="spread2")
    ppe = st.text_input("PPE", key="ppe")

    if st.button("Parkinson's Test Result", key="parkinsons_test_button"):
        try:
            input_data = np.array([float(mdvp_1), float(mdvp_2), float(mdvp_percent), float(mdvp_abs),
                                   float(mdvp_dB), float(jitter), float(
                                       shimmer_1), float(shimmer_2),
                                   float(hnr), float(rpde), float(
                                       dfa), float(spread1),
                                   float(spread2), float(ppe)]).reshape(1, -1)

            prediction = parkinsons_model.predict(input_data)

            if prediction[0] == 1:
                st.error(
                    "The model predicts that the person **has Parkinson's disease.**")
            else:
                st.success(
                    "The model predicts that the person **does not have Parkinson's disease.**")

        except ValueError:
            st.warning("Please enter valid numeric values for all fields.")


def main():
    st.sidebar.title("Disease Prediction App")
    st.sidebar.markdown("Select a disease prediction model:")

    option = st.sidebar.radio(
        "", ["Diabetes Prediction", "Heart Disease Prediction", "Parkinson's Prediction"])

    if option == "Diabetes Prediction":
        diabetes_prediction()
    elif option == "Heart Disease Prediction":
        heart_disease_prediction()
    elif option == "Parkinson's Prediction":
        parkinsons_prediction()


if __name__ == "__main__":
    main()
