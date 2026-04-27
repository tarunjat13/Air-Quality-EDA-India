# 🌫️ Air Quality Analysis & AQI Prediction in India

## 📌 Project Overview
This project focuses on Exploratory Data Analysis (EDA) and Machine Learning to analyze air pollution trends across major Indian cities and predict the Air Quality Index (AQI).

The dataset contains real-world air quality data from 26 Indian cities between 2015 and 2020.

---

## 🎯 Objectives
- Analyze AQI trends across cities and years  
- Identify the most polluted cities  
- Study seasonal pollution patterns  
- Understand pollutant correlations  
- Build a Linear Regression model to predict AQI  

---

## 📂 Dataset Information
- **Source:** CPCB India (via Kaggle)  
- **File:** `city_day.csv`  
- **Records:** 29,531 (24,850 after cleaning)  
- **Features:** 16 columns  

### Key Features:
- PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3  
- City, Date, AQI, AQI_Bucket  

---

## 🧹 Data Cleaning
- Dropped rows with missing AQI values  
- Filled missing values using mean  
- Converted Date column to datetime  
- Extracted Year and Month  

---

## 📊 Exploratory Data Analysis

### 🔴 Key Insights:
- Most polluted cities: Ahmedabad, Delhi, Patna  
- Cleaner cities: Bengaluru, Chennai  
- Highest AQI in winter (Nov–Jan)  
- Lowest AQI during monsoon (Jun–Aug)  
- PM2.5 and PM10 are major contributors  

---

## 📈 Visualizations
- Top polluted cities  
- Year-wise AQI trend  
- Monthly AQI trend  
- AQI category distribution  
- Correlation heatmap  
- City-wise AQI comparison  

---

## 🤖 Machine Learning Model

### Model:
Linear Regression  

### Features Used:
PM2.5, PM10, NO, NO2, NOx, NH3, CO, SO2, O3  

### Performance:
- R² Score: 0.81  
- MAE: 31.22  

---

## 🧠 Key Findings
- PM2.5 has highest impact on AQI  
- Northern cities show higher pollution  
- AQI dropped in 2020 due to COVID lockdown  
- Pollution strongly linked to human activity  

---

## 🚀 Future Scope
- Advanced models (Random Forest, XGBoost, LSTM)  
- Real-time AQI prediction  
- GIS mapping  
- Health impact analysis  
- Interactive dashboards  

---

## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## 📚 References
- CPCB India  
- Kaggle Dataset  
- Pandas, Scikit-learn, Matplotlib, Seaborn Docs

---

## Visualizations

### 1. Top 10 Most Polluted Cities
![Top Cities](v4.png)

> Ahmedabad ranks highest with Avg AQI 452, followed by Delhi (259) and Patna (240).

---

### 2. Yearly AQI Trend (2015–2020)
![Yearly Trend](v1.png)

> AQI steadily decreased from 2015 to 2020. Sharp drop in 2020 due to COVID-19 lockdowns.

---

### 3. Monthly AQI Trend
![Monthly Trend](v2.png)

> AQI peaks in November–January (winter) and drops lowest in July (monsoon season).

---

### 4. AQI Category Distribution
![AQI Distribution](v3.png)

> Moderate and Satisfactory are most common. Poor + Very Poor + Severe = 26% of all days.

---

### 5. Correlation Heatmap
![Heatmap](v5.png)

> PM2.5 (0.65) and CO (0.68) show strongest correlation with AQI.

---

### 6. City-wise AQI Box Plot
![Box Plot](v8.png)

> Ahmedabad shows extreme outliers up to AQI 2000. Delhi and Patna show consistently high spread.

---

### 7. Feature Importance (ML Model)
![Feature Importance](v6.png)

> CO has the highest regression coefficient (11.7), followed by PM2.5 (1.12) and SO2 (0.68).

---

### 8. Actual vs Predicted AQI
![Actual vs Predicted](v7.png)

> Model R² = 0.809. Points cluster tightly along the perfect prediction line confirming strong accuracy.
