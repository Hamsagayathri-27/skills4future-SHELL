import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime

st.title("Final Project: Time & Multiple Feature Prediction")

# --- 1. Upload dataset ---
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Check if time column exists or create one
    time_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
    if time_cols:
        time_col = st.selectbox("Select Time Column", time_cols)
        df[time_col] = pd.to_datetime(df[time_col])
    else:
        st.info("No time column found. Creating 'Time' column automatically.")
        df['Time'] = pd.date_range(start=datetime.now(), periods=len(df), freq='T')
        time_col = 'Time'

    # Show numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("### Numeric Columns")
    st.dataframe(df[numeric_cols].head())

    # --- 2. Select features and target ---
    x_cols = st.multiselect("Select X (features)", numeric_cols, default=numeric_cols[:1])
    y_col = st.selectbox("Select Y (target)", numeric_cols)

    if x_cols and y_col:
        X = df[x_cols]
        y = df[y_col]

        # Handle missing values
        if X.isnull().any().any() or y.isnull().any():
            st.warning("Dataset contains missing values. Dropping missing rows.")
            data = pd.concat([X, y], axis=1).dropna()
            X = data[x_cols]
            y = data[y_col]

        # --- Train model ---
        model = LinearRegression()
        model.fit(X, y)
        st.success("Model trained successfully!")
        st.write(f"### Model Accuracy (R²): {model.score(X, y):.2f}")

        # --- Predict new input ---
        st.write("### Predict New Values")
        new_inputs = {}
        for col in x_cols:
            val = st.number_input(f"Enter value for {col}:", value=0.0, step=0.1)
            new_inputs[col] = val

        if st.button("Predict"):
            input_values = [new_inputs[col] for col in x_cols]
            prediction = model.predict([input_values])[0]
            st.write(f"### Predicted {y_col}: {prediction:.2f}")

            result_df = pd.DataFrame({**new_inputs, y_col: [prediction]})
            st.write("### Input Values with Prediction")
            st.dataframe(result_df)

        # --- Plot Time vs Features ---
        st.write("### Time vs Selected Features Chart")
        selected_plot_cols = st.multiselect("Select features to plot over Time", numeric_cols, default=[y_col])

        if selected_plot_cols:
            plot_df = df[[time_col] + selected_plot_cols].copy()
            plot_df = plot_df.dropna()
            plot_df_melted = plot_df.melt(id_vars=[time_col], value_vars=selected_plot_cols,
                                          var_name="Feature", value_name="Value")
            chart = alt.Chart(plot_df_melted).mark_line(point=True).encode(
                x=f'{time_col}:T',
                y='Value:Q',
                color='Feature:N'
            )
            st.altair_chart(chart, use_container_width=True)
