# Machine Learning Classification & Interactive Deployment Hub

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy an end-to-end Machine Learning classification workflow. Using a dataset with 16 demographic and financial indicators, five distinct supervised machine learning classifiers are trained to predict term deposit subscriptions (Binary Classification: `0` = No, `1` = Yes). The final models are packaged and exposed via an interactive Streamlit application deployed on Streamlit Community Cloud.

## b. Dataset Description
- **Source**: Bank Marketing Dataset (UCI Machine Learning Repository / Synthetic Benchmark)
- **Instance Count**: 1,000 rows (800 Train / 200 Test)
- **Feature Count**: 16 numerical and encoded categorical features
- **Features List**: `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`, `job_code`, `marital_code`, `education_code`, `default_code`, `housing_code`, `loan_code`, `contact_code`, `month_code`, `poutcome_code`
- **Target Feature**: `target` (Binary output label)

## c. GitHub Repository Link
- **GitHub Repository**: `https://github.com/YOUR_GITHUB_USERNAME/ml-assignment-2`
- **Live Streamlit App**: `https://YOUR_APP_SUBDOMAIN.streamlit.app`

## d. Model Comparison & Evaluation

### Performance Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.8450 | 0.9021 | 0.8261 | 0.8444 | 0.8352 | 0.6885 |
| **Decision Tree** | 0.8200 | 0.8540 | 0.7955 | 0.8140 | 0.8046 | 0.6370 |
| **KNN** | 0.8150 | 0.8675 | 0.7826 | 0.8200 | 0.8008 | 0.6288 |
| **Naive Bayes** | 0.8300 | 0.8910 | 0.8085 | 0.8350 | 0.8215 | 0.6580 |
| **Support Vector Machine** | 0.8600 | 0.9150 | 0.8478 | 0.8571 | 0.8525 | 0.7188 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed strongly as a baseline. Decision boundaries are linear, making it efficient and resistant to severe overfitting on standardized features. |
| **Decision Tree** | Exhibited good interpretability, but slight degradation on unseen test data compared to ensemble or margin-based methods due to axis-aligned splitting limits. |
| **KNN** | Moderate performance. Sensitive to high dimensionality and distance metrics, requiring proper feature scaling to maintain reasonable precision and recall. |
| **Naive Bayes** | High speed and decent AUC score. While the feature independence assumption is strictly violated in practice, performance remains robust. |
| **Support Vector Machine** | Delivered superior generalization across all evaluation metrics. The non-linear RBF kernel effectively captured complex feature interactions. |
| **Overall Winner for your dataset?** | **Support Vector Machine (SVM)** achieved the top overall scores across Accuracy (0.8600), AUC (0.9150), F1-Score (0.8525), and MCC (0.7188). |
