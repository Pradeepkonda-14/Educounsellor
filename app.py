import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def load_data():
    try:
        df = pd.read_csv('JEE_Rank_2016_2024.csv')  
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Define function for filtering colleges based on rank
def find_top_5_colleges(df, input_rank):
    df['Closing_Rank'] = pd.to_numeric(df['Closing_Rank'], errors='coerce')
    
    filtered_colleges = df[df['Closing_Rank'] >= input_rank]
    
    top_5_colleges = filtered_colleges.sort_values(by='Closing_Rank').head(10)
    
    return top_5_colleges

def main():
    st.title("IIT Seat Allocation Finder")
    
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
    
    # Add seat type filter
    seat_types = ['All'] + list(df['Seat_Type'].unique())
    selected_seat_type = st.sidebar.selectbox('Select Seat Type', seat_types)

    input_rank = st.number_input("Enter your JEE Rank", min_value=1, max_value=100000, value=5000, step=1)

    if st.button("Find Colleges"):
        recommended_colleges = find_top_5_colleges(df, input_rank)

        # Apply all filters
        if selected_quota != 'All':
            recommended_colleges = recommended_colleges[recommended_colleges['Quota'] == selected_quota]
        if selected_gender != 'All':
            recommended_colleges = recommended_colleges[recommended_colleges['Gender'] == selected_gender]
        if selected_seat_type != 'All':
            recommended_colleges = recommended_colleges[recommended_colleges['Seat_Type'] == selected_seat_type]

        if not recommended_colleges.empty:
            st.success(f"Found {len(recommended_colleges)} colleges for rank {input_rank}")
            
            # Display the results in a more organized way
            st.subheader("Recommended Colleges")
            display_cols = ['Institute', 'Academic_Program_Name', 'Opening_Rank', 'Closing_Rank', 'Quota', 'Gender', 'Seat_Type']
            st.dataframe(recommended_colleges[display_cols].style.highlight_min(subset=['Closing_Rank'], color='lightgreen'))

            # Create visualizations
            st.subheader("Analysis")
            
            # 1. Opening Ranks Comparison
            st.write("Opening Ranks Comparison")
            chart_data = recommended_colleges.set_index('Institute')['Opening_Rank']
            st.bar_chart(chart_data)
            
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
