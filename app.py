import streamlit as st
from upload_data import upload_data_page
from model_parameters import model_parameters_page
from prediction import prediction_page
from utils import apply_custom_css
from train_model import train_model_page

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Model Parameters", "Train Model", "Prediction"], key="main_navigation")
    
    dataset_type = st.sidebar.radio("Select Dataset Type", ("SH", "NH"), key="dataset_type_selection")
    st.session_state['dataset_type'] = dataset_type
    
    apply_custom_css()
    
    if page == "Upload Data":
        upload_data_page()
    elif page == "Model Parameters":
        model_parameters_page()
    elif page == "Prediction":
        prediction_page()
    elif page == "Train Model":
        train_model_page()

if __name__ == "__main__":
    main()
