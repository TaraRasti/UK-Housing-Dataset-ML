# 🏠 UK House Price Prediction with Machine Learning

This project explores UK housing market data to build a robust machine learning pipeline that predicts house prices. Using data from the HM Land Registry, the notebook walks through exploratory data analysis (EDA), feature engineering, statistical testing, and both classical ML models and deep learning to reach up to **94% prediction accuracy**.

---

## 📌 Dataset Description

- **Source:** HM Land Registry Price Paid Data  
- **Coverage:** Residential property sales in England and Wales from 1995 to present  
- **Download Link:** [pp-monthly-update-new-version.csv](https://github.com/TaraRasti/UK-Housing-Dataset-ML/blob/main/pp-monthly-update-new-version.csv)

---

## 🔍 Exploratory Data Analysis (EDA)

### 3.1. Loading the Dataset 📥  
Loaded the raw CSV into a pandas DataFrame with meaningful column names and parsed transaction dates for time-based analysis.

### 3.2. Summary Statistics 📊  
Displayed dataset shape, data types, head, and basic descriptive stats like mean, median, and percentiles.

### 3.3. Missing Values Check 🕳️  
Identified and quantified missing values to inform data cleaning.

### 3.4. Visualisation 📈  
Plotted:
- Distributions of categorical variables (e.g., property type, tenure).
- Histogram of house prices (raw + log-transformed).
- Boxplots for top towns and categorical features.
- Cramér’s V correlation heatmap for categorical variables.

Key insight: House prices are right-skewed and vary widely by town and property type.

---

## 🛠️ Feature Engineering

### ✅ Temporal Features
Extracted `year`, `month`, `quarter`, and `day_of_week` from the transaction date.

### ✅ Binary Features
Created:
- `is_new`: Is the house newly built?
- `is_freehold`: Is the tenure freehold?
- `is_additional_property`: Buy-to-let or second home?

### ✅ Duration Mapping
Mapped `'F'` and `'L'` in the `duration` column to `'Freehold'` and `'Leasehold'`, filling missing values with `'Unknown'`.

### ✅ Postcode Feature 🏷️  
Extracted numeric parts from postcodes (e.g., BS16 → 16) to reduce high-cardinality and reveal regional patterns.

### ✅ Categorical Encoding  
Used `LabelEncoder` to encode key categorical columns:
`['town_city', 'district', 'county', 'property_type', 'record_status', 'ppd_category', 'old_new']`

---

## 📊 Statistical Analysis

Used **ANOVA (Analysis of Variance)** to evaluate the relationship between categorical features and house price. Features like `property_type` and `duration` showed statistically significant impact.

---

## 🤖 Model Building

Split data into training (80%) and test (20%) sets. Trained and compared the following models:

- **Linear Regression**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Polynomial Regression (degree=4)**

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📉 Model Evaluation Results

| Model                  | MAE (£)     | RMSE (£)    | R²     |
|------------------------|-------------|-------------|--------|
| Linear Regression      | 153,406     | 224,968     | 0.110  |
| Random Forest          | 106,977     | 173,390     | 0.472  |
| Gradient Boosting      | 129,364     | 194,170     | 0.337  |
| Polynomial Regression  | 153,495     | 223,208     | 0.124  |

✅ **Best performer: Random Forest**, but still under 50% variance explained.

---

## 🧠 Neural Network with Keras

Used a multi-layer fully connected neural network with ReLU activations, batch normalization, dropout, and early stopping.

### 🏗️ Architecture:
- Dense(128) + ReLU
- BatchNormalization
- Dropout(0.3)
- Dense(64) + ReLU
- Dropout(0.2)
- Dense(32) + ReLU
- Output: Dense(1)

Compiled with Adam optimizer and MSE loss. Trained with validation split and learning rate scheduler.

---

## 📈 Neural Network Results: 94% Accuracy!

After reversing the log-transformation of predictions:

- **MAE:** ~£33,000  
- **RMSE:** ~£47,000  
- **R² Score:** **0.9398**

This model captured the nonlinear patterns in the data far better than classical models. Actual vs. predicted scatter plot confirmed high alignment.

---

## 🧾 Conclusion

- 📊 EDA revealed strong city-level and property-type trends.
- 🛠️ Feature engineering and ANOVA ensured relevant signals were captured.
- 🤖 Random Forest was the best classical ML model.
- 🧠 Neural Network outperformed all, achieving 90% R² on test data.
- 🏠 Real UK data + deep learning = actionable price predictions.

---


