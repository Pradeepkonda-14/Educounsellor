import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_data():
    try:
        # Get the directory where the app.py file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'JEE_Rank_2016_2024.csv')
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def prepare_data_for_ml(df):
    # Create a copy to avoid modifying the original dataframe
    df_ml = df.copy()
    
    # Convert categorical variables to numerical using Label Encoding
    le = LabelEncoder()
    categorical_columns = ['Institute', 'Academic_Program_Name', 'Quota', 'Gender', 'Seat_Type']
    
    for col in categorical_columns:
        df_ml[col] = le.fit_transform(df_ml[col])
    
    # Convert Closing_Rank to numeric
    df_ml['Closing_Rank'] = pd.to_numeric(df_ml['Closing_Rank'], errors='coerce')
    
    # Drop rows with missing values
    df_ml = df_ml.dropna()
    
    return df_ml

def train_model(df_ml):
    # Features for prediction
    X = df_ml[['Institute', 'Academic_Program_Name', 'Quota', 'Gender', 'Seat_Type']]
    y = df_ml['Closing_Rank']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def predict_colleges(model, df, input_rank, selected_quota, selected_gender, selected_seat_type):
    # Create a copy of the dataframe for predictions
    df_pred = df.copy()
    
    # Filter based on user selections
    if selected_quota != 'All':
        df_pred = df_pred[df_pred['Quota'] == selected_quota]
    if selected_gender != 'All':
        df_pred = df_pred[df_pred['Gender'] == selected_gender]
    if selected_seat_type != 'All':
        df_pred = df_pred[df_pred['Seat_Type'] == selected_seat_type]
    
    # Prepare features for prediction
    le = LabelEncoder()
    categorical_columns = ['Institute', 'Academic_Program_Name', 'Quota', 'Gender', 'Seat_Type']
    
    for col in categorical_columns:
        df_pred[col] = le.fit_transform(df_pred[col])
    
    # Make predictions
    X_pred = df_pred[categorical_columns]
    predicted_ranks = model.predict(X_pred)
    
    # Add predictions to the dataframe
    df_pred['Predicted_Rank'] = predicted_ranks
    
    # Filter colleges where predicted rank is higher than input rank
    recommended_colleges = df_pred[df_pred['Predicted_Rank'] >= input_rank]
    
    # Sort by predicted rank and get top 10
    recommended_colleges = recommended_colleges.sort_values(by='Predicted_Rank').head(10)
    
    return recommended_colleges

def main():
    st.title("IIT Seat Allocation Finder with ML Predictions")
    
    df = load_data()

    if df.empty:
        st.error("Failed to load data")
        return

    st.sidebar.header("Filters")
    
    # Add filters
    quotas = ['All'] + list(df['Quota'].unique())
    selected_quota = st.sidebar.selectbox('Select Quota', quotas)

    genders = ['All'] + list(df['Gender'].unique())
    selected_gender = st.sidebar.selectbox('Select Gender', genders)
    
    seat_types = ['All'] + list(df['Seat_Type'].unique())
    selected_seat_type = st.sidebar.selectbox('Select Seat Type', seat_types)

    input_rank = st.number_input("Enter your JEE Rank", min_value=1, max_value=100000, value=5000, step=1)

    if st.button("Find Colleges"):
        # Prepare data and train model
        df_ml = prepare_data_for_ml(df)
        model, X_test, y_test = train_model(df_ml)
        
        # Get predictions
        recommended_colleges = predict_colleges(model, df, input_rank, selected_quota, selected_gender, selected_seat_type)

        if not recommended_colleges.empty:
            st.success(f"Found {len(recommended_colleges)} colleges for rank {input_rank}")
            
            # Display the results
            st.subheader("Recommended Colleges")
            display_cols = ['Institute', 'Academic_Program_Name', 'Opening_Rank', 'Closing_Rank', 'Predicted_Rank', 'Quota', 'Gender', 'Seat_Type']
            st.dataframe(recommended_colleges[display_cols].style.highlight_min(subset=['Predicted_Rank'], color='lightgreen'))

            # Create visualizations
            st.subheader("Analysis")
            
            # 1. Predicted vs Actual Ranks
            st.write("Predicted vs Actual Closing Ranks")
            fig = px.scatter(recommended_colleges, x='Closing_Rank', y='Predicted_Rank', 
                           hover_data=['Institute', 'Academic_Program_Name'],
                           title='Predicted vs Actual Closing Ranks')
            st.plotly_chart(fig)
            
            # 2. Program Distribution
            st.write("Program Distribution")
            program_counts = recommended_colleges['Academic_Program_Name'].value_counts()
            st.bar_chart(program_counts)
            
            # 3. Seat Type Distribution
            st.write("Seat Type Distribution")
            seat_type_counts = recommended_colleges['Seat_Type'].value_counts()
            st.bar_chart(seat_type_counts)
        else:
            st.warning("No colleges found for the given criteria.")

if __name__ == "__main__":
    main()
