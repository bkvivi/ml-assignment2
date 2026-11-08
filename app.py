import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="ML Classifier Evaluation Hub", layout="wide")

st.title("📊 Machine Learning Classification Model Evaluator")
st.markdown("BITS Pilani WILP - Machine Learning Assignment 2")

# Sidebar
st.sidebar.header("1. Upload Test Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV Test Data", type=["csv"])

st.sidebar.header("2. Choose Model")
model_choice = st.sidebar.selectbox(
    "Select Model for Evaluation",
    (
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Support Vector Machine"
    )
)

model_file_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl"
}

if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.subheader("📋 Dataset Preview")
    st.dataframe(df_test.head(5))

    if 'target' not in df_test.columns:
        st.error("Uploaded CSV must contain a 'target' column!")
    else:
        X_test = df_test.drop(columns=['target'])
        y_test = df_test['target']

        try:
            scaler = joblib.load('models/scaler.pkl')
            model_path = os.path.join('models', model_file_map[model_choice])
            model = joblib.load(model_path)

            if model_choice == "Decision Tree":
                preds = model.predict(X_test)
                probs = model.predict_proba(X_test)[:, 1]
            else:
                X_test_scaled = scaler.transform(X_test)
                preds = model.predict(X_test_scaled)
                probs = model.predict_proba(X_test_scaled)[:, 1]

            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)

            st.markdown("---")
            st.subheader(f"📈 Evaluation Metrics: **{model_choice}**")

            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")

            st.markdown("---")
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("🎯 Confusion Matrix")
                cm = confusion_matrix(y_test, preds)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_xlabel('Predicted Label')
                ax.set_ylabel('True Label')
                st.pyplot(fig)

            with col_right:
                st.subheader("📝 Classification Report")
                report_dict = classification_report(y_test, preds, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format(precision=4))

        except Exception as e:
            st.error(f"Error executing prediction pipeline: {e}")
else:
    st.info("👆 Please upload the `test_data.csv` file from the sidebar to evaluate the models.")
