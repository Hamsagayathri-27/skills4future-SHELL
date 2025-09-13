import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import altair as alt
from datetime import datetime

st.set_page_config(page_title="Energy Efficiency Dashboard", layout="wide")
st.title("Final Project: Dynamic Energy Efficiency Dashboard 🚀")

# --- 1. Upload dataset ---
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file:
    # Limit rows for large CSVs to avoid freezing
    df = pd.read_csv(uploaded_file, nrows=5000)
    st.write("### Data Preview")
    st.dataframe(df.head())

    # --- 2. Handle Time Column safely ---
    time_cols = df.columns.tolist()
    time_col = st.selectbox("Select Time Column", time_cols)

    if pd.api.types.is_numeric_dtype(df[time_col]):
        st.warning(f"'{time_col}' is numeric. Creating automatic Time column instead.")
        df['Time'] = pd.date_range(start=datetime.now(), periods=len(df), freq='T')
        time_col = 'Time'
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        if df[time_col].isna().all():
            st.warning(f"Cannot parse '{time_col}' as datetime. Creating automatic Time column.")
            df['Time'] = pd.date_range(start=datetime.now(), periods=len(df), freq='T')
            time_col = 'Time'

    # --- 3. Numeric columns ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.write("### Numeric Columns")
    st.dataframe(df[numeric_cols].head())

    # --- 4. Select features and target ---
    st.sidebar.header("Regression Options")
    x_cols = st.sidebar.multiselect("Select Features (X)", numeric_cols, default=numeric_cols[:1])
    y_col = st.sidebar.selectbox("Select Target (Y)", numeric_cols)

    if x_cols and y_col:
        X = df[x_cols]
        y = df[y_col]

        # Drop missing values
        if X.isnull().any().any() or y.isnull().any():
            st.warning("Dropping rows with missing values.")
            data = pd.concat([X, y], axis=1).dropna()
            X = data[x_cols]
            y = data[y_col]

        # --- 5. Train Model with caching ---
        @st.cache_data
        def train_model(X, y):
            model = LinearRegression()
            model.fit(X, y)
            return model

        model = train_model(X, y)
        st.success("Model trained successfully!")
        st.write(f"### Model Accuracy (R²): {model.score(X, y):.2f}")

        # --- 6. Dynamic Predictions ---
        st.sidebar.header("Predict New Values")
        new_inputs = {}
        for col in x_cols:
            val = st.sidebar.number_input(f"{col}:", value=0.0, step=0.1)
            new_inputs[col] = val

        if st.sidebar.button("Predict & Add"):
            input_values = [new_inputs[col] for col in x_cols]
            prediction = model.predict([input_values])[0]
            st.write(f"### Predicted {y_col}: {prediction:.2f}")

            # Append new row with timestamp
            new_row = {time_col: datetime.now(), **new_inputs, y_col: prediction}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        # --- 7. Time vs Features Chart (Last 1000 rows) ---
        st.subheader("Time vs Features Chart")
        selected_plot_cols = st.multiselect(
            "Select features to plot", numeric_cols + [y_col], default=[y_col]
        )

        if selected_plot_cols:
            # Keep only valid columns
            valid_cols = [col for col in selected_plot_cols if col in df.columns]
            if not valid_cols:
                st.warning("Selected columns not found in data. Please select valid columns.")
            else:
                plot_df = df[[time_col] + valid_cols].dropna().tail(1000)
                plot_df_melted = plot_df.melt(
                    id_vars=[time_col], value_vars=valid_cols,
                    var_name="Feature", value_name="Value"
                )
                chart = alt.Chart(plot_df_melted).mark_line(point=True).encode(
                    x=f'{time_col}:T',
                    y='Value:Q',
                    color='Feature:N',
                    tooltip=[time_col, 'Feature', 'Value']
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

        # --- 8. Optional 1D Regression Line ---
        if len(x_cols) == 1:
            st.subheader("1D Regression Line")
            fig, ax = plt.subplots()
            ax.scatter(X, y, label="Data", color="blue")
            ax.plot(X, model.predict(X), color="red", label="Regression Line", linewidth=2)

            # Highlight predicted point
            if 'prediction' in locals():
                ax.scatter([new_inputs[x_cols[0]]], [prediction], color='green', s=100, label='Prediction')

            ax.set_xlabel(x_cols[0])
            ax.set_ylabel(y_col)
            ax.legend()
            st.pyplot(fig)

        # --- 9. Optional: Normalize features toggle ---
        if st.checkbox("Normalize features for chart visualization"):
            norm_df = df[valid_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            norm_df[time_col] = df[time_col]
            norm_melted = norm_df.melt(id_vars=[time_col], value_vars=valid_cols,
                                       var_name="Feature", value_name="Normalized Value")
            norm_chart = alt.Chart(norm_melted).mark_line(point=True).encode(
                x=f'{time_col}:T',
                y='Normalized Value:Q',
                color='Feature:N',
                tooltip=[time_col, 'Feature', 'Normalized Value']
            ).interactive()
            st.subheader("Normalized Feature Chart")
            st.altair_chart(norm_chart, use_container_width=True)
