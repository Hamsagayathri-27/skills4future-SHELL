import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Final Project: Predicting Energy Efficiency")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Simple example: pick two columns
    if "X" in df.columns and "Y" in df.columns:
        X = df[["X"]]
        y = df["Y"]

        model = LinearRegression()
        model.fit(X, y)

        st.success("Model trained successfully!")

        # User input for prediction
        val = st.number_input("Enter value for X:")
        if val:
            prediction = model.predict(np.array([[val]]))
            st.write("Predicted Y:", prediction[0])
