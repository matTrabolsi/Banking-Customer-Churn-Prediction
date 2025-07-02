
# 🏦 Bank Customer Churn Prediction – Detailed Report

This project focuses on predicting customer churn for a bank using machine learning models. The goal is to classify whether a customer is likely to exit the bank based on their profile and behavior.

---

## 📁 Dataset Overview

The dataset used (`Churn_Modelling.csv`) contains 10,000 rows and the following features:

| Column Name       | Description                                  |
|-------------------|----------------------------------------------|
| CreditScore       | Customer’s credit score                      |
| Geography         | Country (France, Germany, Spain)             |
| Gender            | Male or Female                               |
| Age               | Customer’s age                               |
| Tenure            | Years as a customer                          |
| Balance           | Bank balance                                 |
| NumOfProducts     | Number of products the customer holds        |
| HasCrCard         | Has credit card (1 = Yes, 0 = No)            |
| IsActiveMember    | Is an active member (1 = Yes, 0 = No)        |
| EstimatedSalary   | Estimated annual salary                      |
| Exited            | Target: 1 if customer left, 0 otherwise      |

---

## 🧹 Data Cleaning and Preprocessing

### 🔍 1. Missing Values Handling

- Rows with missing values in the target column `Exited` were removed:
  ```python
  df.dropna(subset='Exited', inplace=True)
  ```

- For numerical features like `CreditScore`, `Balance`, and `EstimatedSalary`, **KNN Imputer** was used:
  ```python
  from sklearn.impute import KNNImputer
  imputer = KNNImputer(n_neighbors=5)
  df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
  ```

- To validate the effectiveness of **KNN imputation**, it was compared with **mean imputation** using **Mean Squared Error (MSE)**:
  ```python
  from sklearn.metrics import mean_squared_error
  mse_knn = mean_squared_error(original, knn_imputed)
  mse_mean = mean_squared_error(original, mean_imputed)
  ```
  ✅ KNN imputation showed lower MSE and preserved feature relationships better.

---

### 📉 2. Outlier Detection and Handling

To filter out unrealistic salary values based on economic standards, outliers were removed using **minimum wage thresholds per country**:

```python
country_min_salary = {
    'France': 17981,
    'Germany': 18360,
    'Spain': 12021
}

df = df[
    (df['Geography'] == 'France') & (df['EstimatedSalary'] >= 17981) |
    (df['Geography'] == 'Germany') & (df['EstimatedSalary'] >= 18360) |
    (df['Geography'] == 'Spain') & (df['EstimatedSalary'] >= 12021)
]
```

This ensured that only customers with plausible salaries remained in the dataset.

---

## 🧠 Models Trained

| Model                     | Accuracy  | Notes                                 |
|---------------------------|-----------|---------------------------------------|
| Logistic Regression       | ~79%      | Baseline linear classifier            |
| Decision Tree             | ~84%      | Performs well but prone to overfit    |
| Random Forest             | ~86%      | Robust ensemble, handles imbalance    |
| K-Nearest Neighbors (KNN) | ~83%      | Distance-based, sensitive to scaling  |
| Support Vector Machine    | ~84%      | Good on clean data                    |
| Gradient Boosting         | ~87%      | Handles complex patterns              |
| XGBoost                   | ~88%      | Best performance across all models    |

> Accuracy values may vary slightly based on train-test splits.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix
- ROC Curve plots

Visual diagnostics include:
- Heatmaps of confusion matrices
- ROC curve comparison across models

