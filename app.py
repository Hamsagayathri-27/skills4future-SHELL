import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

st.title("Final Project: Predicting Energy Efficiency")

# Upload dataset
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview:", df.head())

    # Ask user which columns to use
    st.write("### Select Features (X) and Target (Y)")
    all_columns = df.columns.tolist()
    x_col = st.selectbox("Select X (feature)", all_columns)
    y_col = st.selectbox("Select Y (target)", all_columns)

    if x_col and y_col:
        X = df[[x_col]]
        y = df[y_col]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        st.success("Model trained successfully!")

        # Show model score
        st.write(f"### Model Accuracy (R²): {model.score(X, y):.2f}")

        # Predict for new input
        new_val = st.number_input(f"Enter a value for {x_col} to predict {y_col}:")
        if new_val:
            prediction = model.predict([[new_val]])[0]
            st.write(f"### Predicted {y_col}: {prediction:.2f}")

        # Plot data + regression line
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="Data", color="blue")
        ax.plot(X, model.predict(X), color="red", label="Regression Line")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.legend()
        st.pyplot(fig)
