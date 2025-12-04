import streamlit as st
import pandas as pd
from ai_powered_eda import explaratory_data_analysis_pipeline

st.title("AI-Powered EDA Dashboard")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview", df.head())

    st.write("Running EDA...")
    result = explaratory_data_analysis_pipeline(df)
    st.write(result)
