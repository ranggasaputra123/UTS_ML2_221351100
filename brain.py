import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Brain Tumor Dataset.tflite")
interpreter.allocate_tensors()

# Load encoder dan scaler
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Streamlit UI
st.title("Prediksi Follow-Up Tumor Otak")
st.write("Masukkan data pasien untuk memprediksi apakah diperlukan follow-up.")

# Input fields sesuai nama fitur
age = st.number_input("Age")
tumor_type = st.selectbox("Tumor Type", [0, 1])
tumor_size = st.number_input("Tumor Size")
location = st.selectbox("Location", [1, 2, 3])
histology = st.selectbox("Histology", [0, 1])
stage = st.selectbox("Stage", [1, 2, 3])
radiation = st.selectbox("Radiation Treatment", [0, 1])
surgery = st.selectbox("Surgery Performed", [0, 1])
chemo = st.selectbox("Chemotherapy", [0, 1])
mri_result = st.selectbox("MRI Result", [0, 1])

# Gabungkan input
input_features = [age, tumor_type, tumor_size, location, histology, stage, radiation, surgery, chemo, mri_result]

if st.button("Prediksi"):
    # Preprocessing
    input_array = np.array([input_features])
    input_scaled = scaler.transform(input_array).astype(np.float32)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Decode label
    prediction_index = np.argmax(output_data)
    prediction_label = le.inverse_transform([prediction_index])[0]

    # Show result
    st.success(f"Model memprediksi: **{prediction_label}**")
