## American Express Customer Churn Prediction
End-to-end machine learning project with interpretability, advanced models, and actionable business insights.

# Project Overview

This project builds a complete churn prediction system for American Express credit card customers.
It uses machine learning and deep learning models to identify customers at high risk of leaving (attrition) and generate insights that help the business take targeted retention actions.

# The project follows a full data science lifecycle:
✔ Data Cleaning
✔ Feature Engineering
✔ Preprocessing Pipeline
✔ Baseline Models
✔ Advanced Models (XGBoost, LightGBM, Random Forest)
✔ Deep Learning (PyTorch Tabular NN)
✔ SHAP Interpretability
✔ Final Recommendations & Next Steps

#  Repository Structure
# notebooks/
 Amex_churn_Notebook.ipynb
 # data/
 BankChurners.csv   (not included due to size/licensing)
 # models/
 preprocessor.joblib
 logistic_regression.joblib
 random_forest.joblib
 xgboost_model.joblib
 lightgbm_model.joblib
 pytorch_model.pth
#  README.md

# 1. Data Cleaning

-Removed irrelevant columns (e.g., CLIENTNUM)
-Created target variable churn
-Removed known leakage features:
-Naive Bayes classifier output columns
-Ensured no missing/invalid values remain

# 2. Feature Engineering
Added meaningful features to improve predictive power:

# Engagement Score
-Combines spending frequency, amount, and activity.
-Derived categorical features such as card tier.
-Encoded categorical variables using OneHotEncoder.

# 3. Preprocessing Pipeline

-Implemented a production-grade preprocessing workflow using scikit-learn pipelines:
-Numeric: Median Imputation + StandardScaler
-Categorical: Most Frequent Imputation + One-Hot Encoding
-Saved pipeline for deployment (preprocessor.joblib)

# 4. Models Trained
-Baseline Models
-Logistic Regression
-Random Forest
-Advanced Models
-XGBoost
-LightGBM
-Deep Learning
-PyTorch Tabular Neural Network
-Early stopping
-BatchNorm + Dropout
-BCEWithLogitsLoss for stability

# 5. Model Performance (ROC-AUC)
Model	ROC-AUC
-Logistic Regression	~0.91
-Random Forest	~0.98
-XGBoost	~0.99
-LightGBM	~0.99
-PyTorch NN	~0.96–0.97

# Winner: XGBoost / LightGBM (Best generalization & stability)

#  6. Interpretability — SHAP Analysis

-Used SHAP to understand why customers churn.
-Top 3 Churn Drivers
-Low transaction count / amount (lower engagement)
-High inactivity (Months_Inactive_12_mon)
-High service contact frequency (Contacts_Count_12_mon)
-These factors directly correlate with customer dissatisfaction or reduced card usage.

# 7. Key Insights

-High-value customers with low recent usage are a major churn risk.
-Customers contacting support frequently show dissatisfaction patterns.
-Engagement (spending frequency + amount) is the strongest retention driver.

# 8. Business Recommendations

# Preventive Actions
-Personalized offers for low-engagement users
-Rewards for next few transactions
-EMI bonuses, partner discounts, fee waivers

# Service Improvements
-Priority support for high-contact customers
-Fast-tracked dispute resolution
-Dedicated relationship managers

# Win-Back Campaigns
-Target customers inactive for 2+ months
-Use category-specific offers based on past spending

#  9. Next Steps & Enhancements

-Model Calibration: Improve probability accuracy
-Cost-Sensitive Learning: Reduce false negatives
-Uplift Modeling: Predict who will respond to interventions
-Deployment: Flask/FastAPI + Docker
-Monitoring: Track drift, refresh model quarterly
-A/B Testing: Evaluate retention strategies

# 10. Technologies Used
-Python
-NumPy, Pandas
-Scikit-learn
-XGBoost, LightGBM
-PyTorch
-SHAP
-Matplotlib, Seaborn

#  11. Author

Burhanuddin Farigh
Data Analyst | ML Enthusiast 
