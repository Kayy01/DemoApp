import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

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

def upload_data_page():
    st.title("Upload Data")
    dataset_type = st.radio("Select Dataset Type", ("SH", "NH"), key="upload_data_dataset_type")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx'], key="file_uploader")

    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.write("Uploaded Data:")
            st.write(data)
            st.session_state[f'{dataset_type}_data'] = data
            
            num_rows = data.shape[0]
            st.write(f"The uploaded file has {num_rows} rows.")
            
            if num_rows < 1000:
                num_augmented_rows = 1000 - num_rows
                augmented_data = augment_data(data, num_augmented_rows)
                st.write(f"Data augmented to {augmented_data.shape[0]} rows.")
                
                augmented_file_path = f"{dataset_type}_augmented_data.xlsx"
                augmented_data.to_excel(augmented_file_path, index=False, engine='openpyxl')
                st.write(f"Augmented data saved to {augmented_file_path}.")
                
                st.write("Augmented Data:")
                st.write(augmented_data)
                st.session_state[f'{dataset_type}_augmented_data'] = augmented_data
            
            if num_rows < 5000:
                st.write("The uploaded data has less than 5000 rows. Here is an example dataset:")
                example_data = data.head(10)
                st.write(example_data)

            if st.button("Use Original Data", key="use_original_data_button"):
                st.session_state[f'selected_data_{dataset_type}'] = 'original'
                st.experimental_rerun()

            if f'{dataset_type}_augmented_data' in st.session_state:
                if st.button("Use Augmented Data", key="use_augmented_data_button"):
                    st.session_state[f'selected_data_{dataset_type}'] = 'augmented'
                    st.experimental_rerun()
            
        except Exception as e:
            st.error("An error occurred. Please try again.")
            st.write("Error message:", e)
    else:
        st.write("Uploaded Data:")
        st.write(pd.DataFrame())
