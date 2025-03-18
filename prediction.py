import streamlit as st
from io import BytesIO
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

def transform_sh_data(data):
    data_numeric = data.select_dtypes(include=['number'])
    data_numeric = data_numeric.fillna(data_numeric.median())
    X = data_numeric.drop('Is_True', axis=1)
    return X

def transform_nh_data(data):
    df = data.fillna(method='ffill').fillna(method='bfill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df.drop(['Is_Good'], axis=1)
    return X

def load_model(model_type, dataset_type):
    return tf.keras.models.load_model(f'{dataset_type}_{model_type.lower()}_model.h5')

def prediction_page():
    st.title("Prediction")

    dataset_type = st.radio("Select Dataset Type", ("SH", "NH"), key="prediction_dataset_type")
    uploaded_file = st.file_uploader(f"Choose an Excel file for {dataset_type}", type=['xlsx'], key="prediction_file_uploader")

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Uploaded Data:")
            st.write(data)
            st.session_state[f'{dataset_type}_prediction_data'] = data

            # Perform data transformation
            if dataset_type == "SH":
                X = transform_sh_data(data)
            elif dataset_type == "NH":
                X = transform_nh_data(data)

            # Load the fitted scaler
            scaler_file_name = f'{dataset_type}_scaler.pkl'
            with open(scaler_file_name, 'rb') as f:
                scaler = pickle.load(f)
            X_scaled = scaler.transform(X)

            st.write("Transformed Data:")
            st.write(pd.DataFrame(X_scaled, columns=X.columns))

            model_type = st.selectbox("Select Model Type for Prediction", ["DNN", "CNN", "RNN"], key="prediction_model_type")

            if st.button("Predict"):
                model = load_model(model_type, dataset_type)
                
                if model_type in ["CNN", "RNN"]:
                    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], X_scaled.shape[1], 1))

                predictions = model.predict(X_scaled)
                data['Predictions'] = np.argmax(predictions, axis=1)
                st.write("Prediction Results:")
                st.write(data)

                towrite = BytesIO()
                data.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)

                st.download_button(
                    label="Download Predictions as Excel",
                    data=towrite,
                    file_name="predictions.xlsx",
                    mime="application/vnd.ms-excel"
                )

        except Exception as e:
            st.error("An error occurred. Please try again.")
            st.write("Error message:", e)
    else:
        st.write("Please upload a dataset first.")
