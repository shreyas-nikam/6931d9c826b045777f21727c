
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lime
import lime.lime_tabular
import shap

def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    data = {
        "credit_score": np.random.randint(300, 850, num_samples),
        "income": np.random.randint(30000, 150000, num_samples),
        "loan_amount": np.random.randint(5000, 100000, num_samples),
        "loan_term": np.random.choice([12, 24, 36, 60], num_samples),
        "employment_duration": np.random.randint(0, 20, num_samples),
        "debt_to_income_ratio": np.random.uniform(0.1, 0.5, num_samples),
        "num_credit_lines": np.random.randint(1, 10, num_samples),
        "delinquency_2yrs": np.random.randint(0, 5, num_samples),
        "rejection_reason": np.random.choice(["High DTI", "Low Income", "Poor Credit", "Other", ""], num_samples)
    }
    df = pd.DataFrame(data)

    # Simulate a "loan_approved" target variable
    # Approval is more likely with higher credit score, income, employment duration, lower DTI
    df["loan_approved"] = (
        (df["credit_score"] > 650).astype(int) +
        (df["income"] > 60000).astype(int) +
        (df["employment_duration"] > 3).astype(int) +
        (df["debt_to_income_ratio"] < 0.3).astype(int)
    ) >= 2
    df["loan_approved"] = df["loan_approved"].astype(int)

    # Ensure some rejections align with actual "loan_approved" == 0
    df.loc[df["loan_approved"] == 0, "rejection_reason"] = np.random.choice(["High DTI", "Low Income", "Poor Credit"], size=(df["loan_approved"] == 0).sum())

    return df

def train_credit_model(df):
    features = ["credit_score", "income", "loan_amount", "loan_term", "employment_duration", "debt_to_income_ratio", "num_credit_lines", "delinquency_2yrs"]
    X = df[features]
    y = df["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, features, X_train, accuracy

def identify_critical_applications(df, model, features, threshold=0.5):
    X = df[features]
    predictions = model.predict_proba(X)[:, 1] # Probability of approval
    
    # Identify rejected applications (loan_approved == 0) that are critical for explanation
    # For simplicity, let's consider all rejected applications as critical for this scenario.
    # In a real scenario, "critical" might mean close to the decision boundary, or with specific rejection reasons.
    rejected_applications = df[df["loan_approved"] == 0].copy()
    rejected_applications["predicted_probability"] = model.predict_proba(rejected_applications[features])[:, 1]
    
    return rejected_applications.sort_values(by="predicted_probability", ascending=True)

def explain_with_lime(model, X_train, case_data, features):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=features,
        class_names=["Rejected", "Approved"],
        mode="classification"
    )
    explanation = explainer.explain_instance(
        data_row=case_data.values,
        predict_fn=model.predict_proba,
        num_features=len(features)
    )
    return explanation.as_list()

def explain_with_shap(model, X_train, case_data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(case_data)
    return explainer, shap_values

def generate_human_readable_explanation(application_data, lime_explanation=None, shap_explanation=None, threshold=0.5):
    explanation_text = f"\n**Loan Application Analysis for Applicant ID: {application_data.name}**\n"
    explanation_text += f"\n**Decision:** {