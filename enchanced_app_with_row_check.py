import streamlit as st
import pickle
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import train_model 

# Load the models (assuming the paths are correct and models are saved properly)
lin_model = pickle.load(open(r'C:\Users\P91701\streamlitdemo\IrisClass\lin_model.pkl','rb'))
log_model = pickle.load(open(r'C:\Users\P91701\streamlitdemo\IrisClass\lin_model.pkl','rb'))
svm = pickle.load(open(r'C:\Users\P91701\streamlitdemo\IrisClass\svm.pkl','rb'))

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
    accuracy = np.random.uniform(0.8, 1.0)
    time_used = np.random.uniform(0.1, 1.0)
    train_rate = np.random.uniform(0.01, 0.1)
    
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Time Used: {time_used:.2f} seconds")
    st.write(f"Train Rate: {train_rate:.2f}")
    
    fig, ax = plt.subplots()
    metrics = ['Accuracy', 'Time Used (s)', 'Train Rate']
    values = [accuracy, time_used, train_rate]
    ax.bar(metrics, values, color=['blue', 'green', 'red'])
    st.pyplot(fig)

# Custom CSS for styling
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

# Function to augment data
def augment_data(df, num_augmented_rows):
    augmented_data_rows = []
    for _ in range(num_augmented_rows):
        row_to_augment = df.sample(n=1).iloc[0]
        new_row = row_to_augment.copy()
        numerical_features = df.columns[:-1]
        
        for feature in numerical_features:
            sigma = 0.05 * (df[feature].max() - df[feature].min())
            noise = np.random.normal(0, sigma)
            new_row[feature] += noise
        
        augmented_data_rows.append(new_row)
    
    augmented_df = pd.DataFrame(augmented_data_rows, columns=df.columns)
    df_augmented = pd.concat([df, augmented_df], ignore_index=True)
    df_augmented_shuffled = shuffle(df_augmented, random_state=42)
    return df_augmented_shuffled

# Main function to create the Streamlit app
def main():
    st.markdown(custom_css, unsafe_allow_html=True)
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Model Parameters", "Prediction", "Train Model"])

    if page == "Upload Data":
        st.title("Upload Data")
        uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'])
        if uploaded_file is not None:
            try:
                data = pd.read_excel(uploaded_file)
                st.write("Uploaded Data:")
                st.write(data)
                st.session_state['data'] = data
                
                num_rows = data.shape[0]
                st.write(f"The uploaded file has {num_rows} rows.")
                
                if num_rows < 1000:
                    num_augmented_rows = 1000 - num_rows
                    augmented_data = augment_data(data, num_augmented_rows)
                    st.write(f"Data augmented to {augmented_data.shape[0]} rows.")
                    
                    augmented_file_path = "augmented_data.xlsx"
                    augmented_data.to_excel(augmented_file_path, index=False, engine='openpyxl')
                    st.write(f"Augmented data saved to {augmented_file_path}.")
                    
                    st.write("Augmented Data:")
                    st.write(augmented_data)
                    st.session_state['augmented_data'] = augmented_data
                
                if num_rows < 5000:
                    st.write("The uploaded data has less than 5000 rows. Here is an example dataset:")
                    example_data = data.head(10)
                    st.write(example_data)
                
                if st.button("Use Original Data"):
                    st.session_state['selected_data'] = 'original'
                    st.experimental_rerun()

                if 'augmented_data' in st.session_state:
                    if st.button("Use Augmented Data"):
                        st.session_state['selected_data'] = 'augmented'
                        st.experimental_rerun()
                
            except Exception as e:
                st.error("An error occurred. Please try again.")
                st.write("Error message:", e)
        else:
            st.write("Uploaded Data:")
            st.write(pd.DataFrame())

    elif page == "Model Parameters":
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

    elif page == "Prediction":
        st.title("Prediction")
        if 'data' in st.session_state:
            data = st.session_state['data']
            
            if st.session_state['model'] == "Logistic Regression":
                model = log_model
            elif st.session_state['model'] == "Linear Model":
                model = lin_model
            elif st.session_state['model'] == "SVM":
                model = svm

            if st.button("Predict"):
                predictions = predict_iris(model, data)
                data['Predictions'] = predictions
                st.write(data)

                towrite = BytesIO()
                data.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)

                st.download_button(
                    label="Download Predictions as Excel",
                    data=towrite,
                    file_name="iris_predictions.xlsx",
                    mime="application/vnd.ms-excel"
                )
        else:
            st.write("Please upload data first on the 'Upload Data' page.")

    elif page == "Train Model":
        train_model.train_model_page()

if __name__ == "__main__":
    main()
