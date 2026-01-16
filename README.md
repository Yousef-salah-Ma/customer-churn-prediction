# Customer Churn Prediction

Machine learning project to predict **customer churn** using **XGBoost**, **CatBoost**, and **LightGBM**.

---

## Project Overview
The goal of this project is to predict whether a customer will leave the service (churn) based on their account, demographic, and service usage data. Early prediction of churn helps companies improve customer retention and reduce revenue loss.

---

## Dataset Description
The dataset contains **synthetic customer data** with the following columns:

| Feature | Description |
|---------|-------------|
| customer_id | Unique identifier for each customer |
| signup_date | Date when the customer signed up |
| age | Customer age |
| gender | Customer gender (Male/Female) |
| annual_income | Customer annual income in USD |
| education | Education level (high_school, college, bachelor, master) |
| marital_status | Marital status (single, married, widowed) |
| dependents | Number of dependents |
| tenure | Number of months the customer has been with the service |
| contract | Type of contract (one_year, two_year) |
| payment_method | Payment method (bank_transfer, credit_card, electronic_check) |
| paperless_billing | Whether the customer uses paperless billing (Yes/No) |
| senior_citizen | 1 if senior, 0 otherwise |
| monthlycharges | Monthly charges in USD |
| totalcharges | Total charges accumulated |
| num_services | Number of services subscribed |
| has_phone_service | 1 if has phone service, 0 otherwise |
| has_internet_service | 1 if has internet service, 0 otherwise |
| has_online_security | 1 if online security is enabled |
| has_online_backup | 1 if online backup is enabled |
| has_device_protection | 1 if device protection is enabled |
| has_tech_support | 1 if tech support is enabled |
| has_streaming_tv | 1 if streaming TV service is enabled |
| has_streaming_movies | 1 if streaming movies service is enabled |
| customer_satisfaction | Customer satisfaction score |
| num_complaints | Number of complaints made |
| num_service_calls | Number of service calls |
| late_payments | Number of late payments |
| avg_monthly_gb | Average monthly GB usage |
| days_since_last_interaction | Days since last customer interaction |
| credit_score | Customer credit score |
| churn | Target variable: 0 = No churn, 1 = Churn |
| signup_year | Year of signup |
| signup_month_sin | Sine transform of signup month |
| signup_month_cos | Cosine transform of signup month |

---

## Preprocessing
Data preprocessing is handled using a **pipeline** with the following steps:

1. **Handling Missing Values**
   - Numeric columns with low missingness (`customer_satisfaction`, `num_complaints`) filled with **mean**.
   - Numeric columns with high missingness (`avg_monthly_gb`, `credit_score`, `annual_income`) filled with **median**.

2. **Scaling**
   - All numeric features are scaled using **RobustScaler** to reduce the effect of outliers.

3. **Encoding**
   - Categorical variables (`gender`, `education`, `marital_status`, `contract`, `payment_method`, `paperless_billing`) are transformed using **OneHotEncoder**.

4. **Balancing**
   - The dataset is imbalanced. **SMOTE** is used to oversample the minority class (`churn = 1`).

---

## Models Used
Three tree-based classifiers were evaluated:

| Model | Description |
|-------|-------------|
| XGBoostClassifier | Gradient boosting algorithm optimized for speed and performance |
| CatBoostClassifier | Gradient boosting algorithm that handles categorical features natively |
| LGBMClassifier | Gradient boosting algorithm optimized for efficiency and low memory usage |

**Evaluation Metric:** F1-score (to balance precision and recall due to class imbalance).  
**Cross-validation:** 5-fold cross-validation used for robust evaluation.

---

## Results
| Model | F1-score (Test) | F1-score (Train) |
|-------|----------------|----------------|
| XGBoost | ~90.08% | ~90.11% |
| CatBoost | ~90.09% | ~90.13% |
| LightGBM | ~90.09% | ~90.13% |

All three models performed similarly, with XGBoost being slightly faster in training.

---
## How to Use
1. Clone this repository:
   ```bash
   git clone https://github.com/Yousef-salah-Ma/customer-churn-prediction.git
2.Install dependencies:
      ```bash
      pip install -r requirements.txt
Load the pre-trained model:

import joblib
model = joblib.load("model.pkl")
Predict on new data:

y_pred = model.predict(X_new)
Notes
All preprocessing (imputation, scaling, encoding) and balancing (SMOTE) are included in the pipeline, so the model can directly accept raw input.

Focused on reproducibility and easy deployment.

Can be extended to include hyperparameter tuning, feature engineering, or deep learning models if dataset grows larger.
