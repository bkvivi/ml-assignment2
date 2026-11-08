import numpy as np
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Set seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Generate synthetic dataset meeting >12 features and >500 instances requirement
data = {
    'age': np.random.randint(18, 70, n_samples),
    'balance': np.random.randint(-1000, 50000, n_samples),
    'day': np.random.randint(1, 31, n_samples),
    'duration': np.random.randint(10, 1000, n_samples),
    'campaign': np.random.randint(1, 10, n_samples),
    'pdays': np.random.randint(-1, 500, n_samples),
    'previous': np.random.randint(0, 10, n_samples),
    'job_code': np.random.randint(0, 12, n_samples),
    'marital_code': np.random.randint(0, 3, n_samples),
    'education_code': np.random.randint(0, 4, n_samples),
    'default_code': np.random.randint(0, 2, n_samples),
    'housing_code': np.random.randint(0, 2, n_samples),
    'loan_code': np.random.randint(0, 2, n_samples),
    'contact_code': np.random.randint(0, 3, n_samples),
    'month_code': np.random.randint(0, 12, n_samples),
    'poutcome_code': np.random.randint(0, 4, n_samples),
}

df = pd.DataFrame(data)

# Create realistic binary target variable based on duration and balance
logits = (df['duration'] * 0.005) + (df['balance'] * 0.00005) - 2.5
probs = 1 / (1 + np.exp(-logits))
df['target'] = (probs > np.random.rand(n_samples)).astype(int)

# Split into Train / Test sets
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save test dataset (CSV format) required for Streamlit evaluation
test_df = X_test.copy()
test_df['target'] = y_test
test_df.to_csv('test_data.csv', index=False)
print("Saved test_data.csv successfully.")

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')

# Train and evaluation of models
models = {
    'logistic_regression': LogisticRegression(random_state=42),
    'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'naive_bayes': GaussianNB(),
    'svm': SVC(probability=True, random_state=42)
}

metrics_summary = []

for name, model in models.items():
    if name == 'decision_tree':
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Export trained artifacts
    joblib.dump(model, f'models/{name}.pkl')
    
    # Calculate evaluation metrics
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    mcc = matthews_corrcoef(y_test, preds)
    
    metrics_summary.append({
        'Model': name, 'Accuracy': acc, 'AUC': auc, 
        'Precision': prec, 'Recall': rec, 'F1': f1, 'MCC': mcc
    })

print("Models successfully trained and exported to models/")
print(pd.DataFrame(metrics_summary))
