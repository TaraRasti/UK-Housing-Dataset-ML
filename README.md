# UK-Housing-Dataset-ML
🏡 UK Housing Dataset 2024 — Predicting House Prices with Machine Learning
## 🧠 UK Housing Price Prediction – End-to-End Machine Learning Pipeline

This notebook presents an end-to-end machine learning pipeline using real UK house price data to predict property prices based on various features such as location, property type, tenure, and transaction date.

---

### 📌 Key Features of This Project

#### 🔍 Exploratory Data Analysis (EDA)
- Identify missing values and outliers
- Plot distributions, time trends, and correlations
- Explore categorical variables with boxplots

#### 🧹 Data Preprocessing
- One-hot encoding of categorical variables
- Log-transformation of skewed target (`price`)
- Missing value handling and feature scaling

#### 🤖 Modeling
- Train a deep neural network (TensorFlow/Keras) for regression
- Evaluate performance using **MAE**, **RMSE**, and **R²** metrics
- Address challenges like exploding loss and unstable predictions

#### 🔁 Post-processing
- Reverse log-transformed predictions to original price scale
- Visualize predicted vs actual prices
- Handle prediction `NaN`s and outlier issues

## 📈 How I Improved Model Accuracy from 15% to 90%

Initially, the model achieved a low **R² score of ~0.15**, indicating it explained only 15% of the variance in house prices. Through a series of targeted improvements, I was able to significantly increase the accuracy — reaching an R² of over **0.90**.

### Dataset: `pp-monthly-update-new-version.csv`

- Source: HM Land Registry Price Paid Data (England & Wales) as of Month‑Year
- Contains real residential sales data since 1995: prices, postcodes, property types, tenure, and dates
- Ideal for ML pipeline: rich categorical, temporal, and numerical features
- Licensed under the Open Government Licence (OGL) — perfect for open-source projects

  
