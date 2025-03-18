
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the models (assuming the paths are correct and models are saved properly)


# Function to classify prediction results
def classify(num):
    if num < 0.5:
        return 'Setosa'
    elif num < 1.5:
        return 'Versicolor'
    else:
        return 'Virginica'

# Function to predict iris species
def predict_iris(model, data):
    predictions = model.predict(data)
    return [classify(pred) for pred in predictions]

# Function to display model details and plot graphs
def display_model_details(model_name):
    st.write(f"Details for {model_name}:")
    # Dummy values for demonstration purposes
    accuracy = np.random.uniform(0.8, 1.0)
    time_used = np.random.uniform(0.1, 1.0)
    train_rate = np.random.uniform(0.01, 0.1)
    
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Time Used: {time_used:.2f} seconds")
    st.write(f"Train Rate: {train_rate:.2f}")
    
    # Plotting
    fig, ax = plt.subplots()
    metrics = ['Accuracy', 'Time Used (s)', 'Train Rate']
    values = [accuracy, time_used, train_rate]
    ax.bar(metrics, values, color=['blue', 'green', 'red'])
    st.pyplot(fig)

# Custom CSS for styling
def apply_custom_css():
    custom_css = """
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }
    h1, h2, h3 {
        color: #333;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
