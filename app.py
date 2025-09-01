import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("ecg5000_lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Class mapping (ECG5000 dataset → 5 classes)
class_labels = {
    1: "Normal",
    2: "R-on-T Premature Ventricular Contraction",
    3: "Premature Ventricular Contraction",
    4: "Unclassifiable Beat",
    5: "Fusion Beat"
}

# Streamlit UI
st.title("ECG5000 Signal Classification")
st.write("Upload a CSV file containing **one ECG time-series row** to classify it into one of 5 classes.")

uploaded_file = st.file_uploader("📂 Upload ECG CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded CSV
        data = pd.read_csv(uploaded_file, header=None)
        st.success("File Uploaded Successfully!")

        # Check shape
        if data.shape[1] != 140:
            st.error(f"Expected 140 features, but got {data.shape[1]}. Please upload a correct ECG row.")
        else:
            # Scale & reshape input
            input_data = scaler.transform(data.values)
            input_data = input_data.reshape((input_data.shape[0], input_data.shape[1], 1))

            # Prediction
            prediction = model.predict(input_data)
            pred_class = np.argmax(prediction, axis=1)[0] + 1  
            confidence = float(np.max(prediction))

            # Show results
            st.subheader("📊 Prediction Result")
            st.write(f"**Predicted Class:** {class_labels[pred_class]} ({pred_class})")
            st.write(f"**Confidence Score:** {confidence:.4f}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

