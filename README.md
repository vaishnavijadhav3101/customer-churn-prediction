# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn based on demographic, service, contract, and billing information.

## 📌 Project Overview

Customer churn occurs when a customer stops using a company's products or services.

The objective of this project is to build a binary classification model that can identify customers who are more likely to churn. Such predictions can help businesses identify customers who may require targeted retention strategies.

This project covers the complete machine learning workflow:

- Data understanding and cleaning
- Exploratory Data Analysis (EDA)
- Feature preparation
- Data preprocessing
- Model training
- Model comparison
- Cross-validation
- Model evaluation
- Churn probability prediction
- Model serialization

---

## 🎯 Business Problem

Telecom companies can lose recurring revenue when customers discontinue their services.

A churn prediction system can help identify customers who are potentially at higher risk of leaving, allowing the business to consider appropriate retention actions.

### Machine Learning Objective

Predict whether a customer will:

- `0` → No Churn
- `1` → Churn

---

## 📊 Dataset

The dataset contains **7,043 customers and 21 columns**.

The features include:

- Customer demographics
- Phone services
- Internet services
- Additional services
- Contract information
- Billing information
- Payment method

### Target Variable

**Churn**

- `No` → Customer did not churn
- `Yes` → Customer churned

### Target Distribution

| Class | Customers | Percentage |
|---|---:|---:|
| No Churn | 5,174 | 73.46% |
| Churn | 1,869 | 26.54% |

The target variable is moderately imbalanced. Therefore, model evaluation was performed using multiple metrics rather than relying on accuracy alone.

---

## 🧹 Data Cleaning

The following data-cleaning steps were performed:

1. Inspected the dataset structure and data types.
2. Checked for missing values.
3. Identified blank values in `TotalCharges`.
4. Converted blank values to missing values.
5. Converted `TotalCharges` to a numerical data type.
6. Filled missing `TotalCharges` values with `0` for zero-tenure customers.
7. Checked for duplicate records.
8. Removed `customerID` before model training because it is a unique identifier and does not provide useful predictive information.

---

## 📈 Exploratory Data Analysis

Several relationships between customer characteristics and churn were explored.

### Key observations

- Month-to-month customers had a substantially higher observed churn rate than customers with one-year or two-year contracts.
- Customers who churned had lower average tenure than customers who did not churn.
- Churned customers had higher average monthly charges.
- Churn rates differed across internet service categories, with fiber-optic customers showing a higher observed churn rate in this dataset.
- Churned customers had lower average total charges, which is consistent with their lower average tenure.

These findings represent associations observed in the dataset and do not establish causal relationships.

---

## ⚙️ Data Preprocessing

### Numerical Features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

Numerical features were standardized using `StandardScaler`.

### Categorical Features

Categorical variables were transformed using `OneHotEncoder`.

```python
OneHotEncoder(handle_unknown="ignore")
