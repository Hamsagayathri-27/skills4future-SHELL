# 💡 Skills4Future-SHELL
Internship projects completed during the Shell Skills4Future Program.
This repo includes data analysis, machine learning, and visualization projects, such as Renewable Energy Dataset Analysis.

---

## 🌍 Renewable Energy Dataset Analysis

This project is part of the Shell Skills4Future Internship.
It involves analyzing renewable energy consumption data across multiple countries and years using Python, Pandas, Matplotlib, and Scikit-learn.

---

## ⚙️ Tools & Libraries Used
- Python
- Pandas
- Matplotlib / Seaborn (for visualization)
- Scikit-learn (for ML models)
- Jupyter Notebook
- Joblib (for model deployment)

---

## 📊 Dataset

Dataset link → https://www.kaggle.com/datasets/ayushchandramaurya/renewable-energy

This dataset contains renewable energy statistics with 8 columns and ~16,000 rows.

Column Descriptions:
- LOCATION: Country/region code (e.g., AUS, OECD)
- INDICATOR: Type of indicator (e.g., RENEWABLE)
- SUBJECT: Sub-category/type of renewable energy (TOT = Total)
- MEASURE: Unit (KTOE, % of primary energy supply)
- FREQUENCY: Data frequency (A = Annual)
- TIME: Year of record (1960–2015)
- Value: Numerical value of renewable energy consumption/production
- Flag Codes: Notes/flags (NaN if not applicable)

---

## 📑 Example Records

LOCATION | INDICATOR | SUBJECT | MEASURE | FREQUENCY | TIME | Value | Flag Codes
---|---|---|---|---|---|---|---
AUS | RENEWABLE | TOT | KTOE | A | 1960 | 4436.932 | NaN
AUS | RENEWABLE | TOT | KTOE | A | 1961 | 4490.510 | NaN
OECD | RENEWABLE | TOT | PC_PRYENRGSUPPLY | A | 2015 | 9.640 | NaN

---

# 📌 Project Work

## ✅ Week 1 – Exploratory Data Analysis (EDA) 
- Imported and cleaned dataset.
- Explored dataset structure (shape, columns, missing values).
- Analyzed renewable energy supply trends by country and year.
- Visualized growth patterns globally and regionally.
- Compared adoption among countries (India, USA, China).
- Identified top years of renewable energy usage for selected countries.
---

## ✅ Week 2 – Machine Learning Model Implementation
- Preprocessed dataset (handled NaN, selected features: LOCATION, TIME, Value).
- Applied Linear Regression to predict renewable energy usage.
- Trained/Tested the model using train_test_split.
- Evaluated performance with MAE and R² Score.
- Visualized predictions vs actual values for India, USA, China, Australia.

ML Output Example:
- Model trained on renewable energy data of India.
- Predicted growth trend for upcoming years.
- Accuracy (R² Score): ~0.92

Insight: Forecasts renewable energy adoption to assist policymakers.

---

## ✅ Week 3 – ML Deployment & Predictions
- Implemented deployment workflow for the trained ML model using joblib.

Steps Performed:
1. Dataset Loading & Preprocessing
   - Dropped unnecessary columns and handled missing target values.
   - One-hot encoded categorical columns (e.g., LOCATION).
   - Standardized numeric features with StandardScaler.
2. Model Training
   - Trained a Random Forest Regressor (200 trees, max depth 10).
   - Evaluated with MSE and R² Score.
   - Determined feature importance.
3. Model Saving
   - Saved model and scaler using joblib: renewable_energy_model.pkl & renewable_energy_scaler.pkl
4. Flexible Prediction Workflow
   - Batch prediction from CSV → outputs predicted_results.csv
   - Manual input mode → dynamically enter any feature values
   - Predictions scaled, computed, and displayed automatically
5. Visualization of Predictions
   - Plotted predicted values for batch CSV or manual input

Manual Input Example:
Enter values for features to predict:
LOCATION_OECD: 0
TIME: 2015
...
Prediction Completed:
LOCATION_OECD  TIME  Predicted_Value
0 0 2015 5980.23

Feature Importance Example:
Feature | Importance
---|---
TIME | 0.42
LOCATION_OECD | 0.30
... | ...

---

## 🔍 Insights You Can Explore
- Compare renewable energy usage across countries & regions
- Analyze year-wise trends (1960–2015)
- Study energy transition by units (KTOE vs % supply)
- Identify countries with low renewable energy adoption
- OECD vs non-OECD renewable energy performance
- Forecast future renewable energy adoption with ML models

---

## 🚀 Potential Applications
- Policy analysis for sustainable energy growth
- Tracking climate change goals (Paris Agreement)
- Comparative research (developed vs developing countries)
- Forecasting renewable energy demand

---

## 🌱 Features Implemented
- India’s renewable energy growth trend
- Comparison of India, China, USA
- Top 10 years of India’s renewable energy usage
- Global renewable energy usage in the latest year
- Growth rate calculation (CAGR)
- Forecasting using ML (Linear Regression & Random Forest)
- Model deployment with joblib
- Manual & batch input prediction workflow
- Feature importance visualization
- Prediction plotting
---

## 📌 How to Run
- Notebook: Open renewable_energy.ipynb in Jupyter → Run all cells
- Script: python renewable_energy_ml_deployment.py
