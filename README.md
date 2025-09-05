# skills4future-SHELL
💡 Internship projects completed during the Shell Skills4Future Program.  
This repo includes **data analysis, machine learning, and visualization projects** such as **Renewable Energy Dataset Analysis**.

---

## 🌍 Renewable Energy Dataset Analysis  

This project is part of the **Shell Skills4Future Internship**.  
It involves analyzing renewable energy consumption data across multiple countries and years using **Python, Pandas, Matplotlib, and Scikit-learn**.  

---

## ⚙️ Tools & Libraries Used  
- Python  
- Pandas  
- Matplotlib / Seaborn (for visualization)  
- Scikit-learn (for ML models)  
- Jupyter Notebook  

---

## 📊 Dataset  

📌 Dataset link → [Kaggle - Renewable Energy Dataset](https://www.kaggle.com/datasets/ayushchandramaurya/renewable-energy)  

This dataset contains **renewable energy statistics** with 8 columns and ~16,000 rows.  

| Column Name   | Description |
|---------------|-------------|
| **LOCATION**  | Country or region code (e.g., AUS for Australia, OECD for Organization for Economic Co-operation and Development). |
| **INDICATOR** | Type of indicator measured (e.g., RENEWABLE). |
| **SUBJECT**   | Sub-category or type of renewable energy (e.g., TOT = Total). |
| **MEASURE**   | Unit of measurement (e.g., KTOE = Kilotonne of Oil Equivalent, PC_PRYENRGSUPPLY = % of Primary Energy Supply). |
| **FREQUENCY** | Data frequency (e.g., A = Annual). |
| **TIME**      | Year of the record (1960–2015). |
| **Value**     | Numerical value of renewable energy consumption/production. |
| **Flag Codes**| Notes/flags for data quality (NaN if not applicable). |

---

## 📑 Example Records  

| LOCATION | INDICATOR | SUBJECT | MEASURE | FREQUENCY | TIME | Value    | Flag Codes |
|----------|-----------|---------|---------|-----------|------|----------|------------|
| AUS      | RENEWABLE | TOT     | KTOE    | A         | 1960 | 4436.932 | NaN        |
| AUS      | RENEWABLE | TOT     | KTOE    | A         | 1961 | 4490.510 | NaN        |
| AUS      | RENEWABLE | TOT     | KTOE    | A         | 1962 | 4407.097 | NaN        |
| AUS      | RENEWABLE | TOT     | KTOE    | A         | 1963 | 4628.738 | NaN        |
| OECD     | RENEWABLE | TOT     | PC_PRYENRGSUPPLY | A | 2015 | 9.640    | NaN        |

---

# 📌 Project Work  

## ✅ Week 1 – Exploratory Data Analysis (EDA) & Visualization  

🔹 Imported and cleaned the dataset.  
🔹 Explored dataset structure (shape, columns, missing values).  
🔹 Analyzed renewable energy supply trends by **country and year**.  
🔹 Visualized renewable energy growth patterns globally and regionally.  
🔹 Compared renewable energy adoption among countries (India, USA, China).  
🔹 Identified **top years of renewable energy usage** for selected countries.  

📊 **Sample Visualizations:**  
- Line plot of renewable energy growth for India (1960–2015).  
- Bar chart comparing renewable energy usage across multiple countries.  
- Pie chart of renewable energy contribution by country in 2015.  

---

## ✅ Week 2 – Machine Learning Model Implementation  

In Week 2, we applied **ML models** to analyze and predict renewable energy trends.  

### 🔹 Steps Performed:  
1. **Preprocessed the dataset** (handled NaN, selected features: LOCATION, TIME, Value).  
2. **Applied Linear Regression** to predict renewable energy usage over years.  
3. **Trained/Tested the model** using `train_test_split`.  
4. **Evaluated model performance** using **Mean Absolute Error (MAE), R² Score**.  
5. **Visualized predictions vs actual values** for countries (India, USA, China, Australia).  

### 🔹 Example ML Output:  
- Model trained on renewable energy data of India.  
- Predicted growth trend for upcoming years.  
- Accuracy (R² Score): ~0.92 (highly reliable).  

📈 The ML model helps in **forecasting renewable energy adoption** and can assist policymakers in planning sustainable energy growth.  

---

# 🔍 Insights You Can Explore  
- 🌍 Compare renewable energy usage across countries and regions.  
- 📈 Analyze year-wise trends (1960–2015).  
- ⚡ Study energy transition by units (KTOE vs % supply).  
- 📉 Identify countries with low renewable energy adoption.  
- 🏆 OECD vs non-OECD renewable energy performance.  
- 🔮 Forecast future renewable energy adoption with ML models.  

---

# 🚀 Potential Applications  
- Policy analysis for sustainable energy growth.  
- Tracking climate change goals (Paris Agreement targets).  
- Comparative research (developed vs developing countries).  
- Forecasting renewable energy demand.  

---

# 🌱 Features Implemented  
✔ India’s renewable energy growth trend  
✔ Comparison of India, China, USA  
✔ Top 10 years of India’s renewable energy usage  
✔ Global renewable energy usage in the latest year  
✔ Growth rate calculation (CAGR)  
✔ Forecasting using ML (Linear Regression)  

---
