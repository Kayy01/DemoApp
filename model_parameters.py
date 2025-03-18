
import streamlit as st
from utils import display_model_details

def model_parameters_page():
    st.title("Model Parameters")
    st.write("Modify the model parameters and select the model to be used.")
    
    model_name = st.selectbox("Select Model", ["Logistic Regression", "Linear Model", "SVM"])
    st.session_state['model'] = model_name
    
    if model_name == "Logistic Regression":
        display_model_details("Logistic Regression")
    elif model_name == "Linear Model":
        display_model_details("Linear Model")
    elif model_name == "SVM":
        display_model_details("SVM")
