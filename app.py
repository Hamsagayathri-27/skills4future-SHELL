import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime, timedelta

st.title("Final Project: Time & Dynamic Prediction Dashboard")

# --- 1. Upload dataset ---
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # Check for time column or create one
    time_cols = df.select_dtypes(include=['datetime64', 'object']).columns.tolist()
    if time_cols:
        time_col = st.selectbox("Select Time Column", time_cols)
        df[time_col] = pd.to_datetime(df[time_col])
    else:
        st.info("No time column found. Creating 'Time' column automatically.")
        df['Time'] = pd.date_range(start=datetime.now(), periods=len(df), freq='T')
        time_col = 'Time'

    # Numeric columns
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

        # Train model
        model = LinearRegression()
        model.fit(X, y)
        st.success("Model trained successfully!")
        st.write(f"### Model Accuracy (R²): {model.score(X, y):.2f}")

        # --- 3. Dynamic prediction input ---
        st.write("### Predict New Values Dynamically")
        new_inputs = {}
        for col in x_cols:
            val = st.number_input(f"Enter value for {col}:", value=0.0, step=0.1)
            new_inputs[col] = val

        if st.button("Predict & Add to Chart"):
            input_values = [new_inputs[col] for col in x_cols]
            prediction = model.predict([input_values])[0]
            st.write(f"### Predicted {y_col}: {prediction:.2f}")

            # Append new prediction to dataframe with current timestamp
            new_row = {time_col: datetime.now(), **new_inputs, y_col: prediction}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # --- 4. Plot dynamic Altair chart ---
        selected_plot_cols = st.multiselect("Select features to plot over Time", numeric_cols + [y_col],
                                            default=[y_col])
        if selected_plot_cols:
            plot_df = df[[time_col] + selected_plot_cols].copy()
            plot_df = plot_df.dropna()
            plot_df_melted = plot_df.melt(id_vars=[time_col], value_vars=selected_plot_cols,
                                          var_name="Feature", value_name="Value")
            chart = alt.Chart(plot_df_melted).mark_line(point=True).encode(
                x=f'{time_col}:T',
                y='Value:Q',
                color='Feature:N'
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        # --- Optional: 1D regression plot ---
        if len(x_cols) == 1:
            fig, ax = plt.subplots()
            ax.scatter(X, y, label="Data", color="blue")
            ax.plot(X, model.predict(X), color="red", label="Regression Line")
            ax.set_xlabel(x_cols[0])
            ax.set_ylabel(y_col)
            ax.legend()
            st.pyplot(fig)
