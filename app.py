from pycaret.classification import *
import streamlit as st
import pandas as pd

model = load_model('tuned_rf_diabetes')

def predict(model, input_df):
    predictions_df = predict_model(estimator = model, data = input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions

def run():
    pregnancies = st.text_input("Pregnancies")
    glucose = st.text_input("Glucose")
    blood_pressure = st.text_input("Blood Pressure")
    skin_thickness = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    dpf = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")
    output = ""

    input_dict = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age
    }

    input_df = pd.DataFrame([input_dict])

    if st.button("Predict"):
        output = predict(model= model, input_df=input_df)
        output = str(output)
    st.success(f"The output is {output}")

if __name__ == '__main__':
    run()

  

