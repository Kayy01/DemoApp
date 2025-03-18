import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from io import BytesIO
import pickle 

# Define functions to create models
def create_dnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

def create_cnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model

def create_rnn_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(input_shape, 1)),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    return model


def transform_sh_data(data):
    data_numeric = data.select_dtypes(include=['number'])
    data_numeric = data_numeric.fillna(data_numeric.median())
    X = data_numeric.drop('Is_True', axis=1)
    y = data_numeric['Is_True']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

def transform_nh_data(data):
    df = data.fillna(method='ffill').fillna(method='bfill')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df.drop(['Is_Good'], axis=1)
    y = df['Is_Good']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler

# Update the train_model_page function
def train_model_page():
    st.title("Train Model")

    dataset_type = st.session_state['dataset_type']
    selected_data_key = f'selected_data_{dataset_type}'

    if selected_data_key in st.session_state:
        data_key = f"{dataset_type}_data"
        if st.session_state[selected_data_key] == 'augmented':
            data_key = f"{dataset_type}_augmented_data"
        
        data = st.session_state[data_key]

        if dataset_type == "SH":
            X_train, X_test, y_train, y_test, scaler = transform_sh_data(data)
        elif dataset_type == "NH":
            X_train, X_test, y_train, y_test, scaler = transform_nh_data(data)

        y_train_enc = tf.keras.utils.to_categorical(y_train)
        y_test_enc = tf.keras.utils.to_categorical(y_test)

        model_type = st.selectbox("Select Model Type", ["DNN", "CNN", "RNN"])

        if st.button("Train"):
            if model_type == "DNN":
                model = create_dnn_model(X_train.shape[1])
            elif model_type == "CNN":
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                model = create_cnn_model(X_train.shape[1])
            elif model_type == "RNN":
                X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                model = create_rnn_model(X_train.shape[1])

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            with st.spinner('Training the model...'):
                model.fit(X_train, y_train_enc, epochs=10, batch_size=32, validation_split=0.1)
                loss, accuracy = model.evaluate(X_test, y_test_enc)
            
            st.success('Model training completed!')
            st.write(f"Model Type: {model_type}")
            st.write(f"Test Loss: {loss}")
            st.write(f"Test Accuracy: {accuracy}")
            
            model_file_name = f'{dataset_type}_{model_type.lower()}_model.h5'
            model.save(model_file_name)
            st.write(f"The trained model has been saved as {model_file_name}")

            # Save the fitted scaler
            scaler_file_name = f'{dataset_type}_scaler.pkl'
            with open(scaler_file_name, 'wb') as f:
                pickle.dump(scaler, f)
            st.write(f"The scaler has been saved as {scaler_file_name}")
    else:
        st.write("Please upload and select a dataset first on the 'Upload Data' page.")

