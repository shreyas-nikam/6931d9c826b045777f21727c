
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
    df.loc[df["loan_approved"] == 0, "rejection_reason"] = np.random.choice(
        ["High DTI", "Low Income", "Poor Credit"], size=(df["loan_approved"] == 0).sum())

    return df


def train_credit_model(df):
    features = ["credit_score", "income", "loan_amount", "loan_term",
                "employment_duration", "debt_to_income_ratio", "num_credit_lines", "delinquency_2yrs"]
    X = df[features]
    y = df["loan_approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, features, X_train, accuracy


def identify_critical_applications(df, model, features, threshold=0.5):
    X = df[features]
    predictions = model.predict_proba(X)[:, 1]  # Probability of approval

    # Identify rejected applications (loan_approved == 0) that are critical for explanation
    # For simplicity, let's consider all rejected applications as critical for this scenario.
    # In a real scenario, "critical" might mean close to the decision boundary, or with specific rejection reasons.
    rejected_applications = df[df["loan_approved"] == 0].copy()
    rejected_applications["predicted_probability"] = model.predict_proba(
        rejected_applications[features])[:, 1]

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
    # Ensure case_data is 2D array (1 sample, n features)
    if hasattr(case_data, 'values'):
        case_data_array = case_data.values.reshape(1, -1)
    else:
        case_data_array = np.array(case_data).reshape(1, -1)
    shap_values = explainer.shap_values(case_data_array)
    return explainer, shap_values


def generate_human_readable_explanation(application_data, lime_explanation=None, shap_explanation=None, threshold=0.5):
    explanation_text = f"\n**Loan Application Analysis for Applicant ID: {application_data.name}**\n"
    # Use actual predicted probability if available
    prediction_prob = application_data.get("predicted_probability", 0.0)

    if prediction_prob < threshold:  # Assuming a lower probability indicates rejection
        explanation_text += "\n**Decision:** Declined\n"
        explanation_text += f"\nBased on our model, the probability of approval for this application was low ($P(\\text{{Approved}}) = {prediction_prob:.4f}$). The primary factors contributing to this decision include:\n"
    else:
        explanation_text += "\n**Decision:** Approved\n"
        explanation_text += f"\nBased on our model, the probability of approval for this application was high ($P(\\text{{Approved}}) = {prediction_prob:.4f}$). Key factors supporting this decision include:\n"

    # Incorporate LIME insights
    if lime_explanation:
        explanation_text += "\n**LIME Insights (Local Feature Impact):**\n"
        # For simplicity, filter for top contributing features towards rejection/approval
        sorted_lime = sorted(
            lime_explanation, key=lambda x: x[1])
        top_positive = [f"{feat} ({weight:.2f})" for feat,
                        weight in sorted_lime if weight > 0][:3]
        top_negative = [f"{feat} ({weight:.2f})" for feat,
                        weight in sorted_lime if weight < 0][:3]

        if top_negative and prediction_prob < threshold:
            explanation_text += "- Features that significantly *decreased* the likelihood of approval:\n"
            for item in top_negative:
                explanation_text += f"  - {item}\n"
        elif top_positive and prediction_prob >= threshold:
            explanation_text += "- Features that significantly *increased* the likelihood of approval:\n"
            for item in top_positive:
                explanation_text += f"  - {item}\n"

        # Add a general statement about current feature values
        explanation_text += "\n*Current application values that influenced the decision:*\n"
        for feature in application_data.index:
            if feature not in ["loan_approved", "rejection_reason", "predicted_probability"]:
                explanation_text += f"- **{feature.replace('_', ' ').title()}:** {application_data[feature]}\n"

    # Incorporate SHAP insights
    if shap_explanation is not None and not shap_explanation.empty:
        explanation_text += "\n**SHAP Insights (Feature Contribution Relative to Average):**\n"
        # Filter for top N features by absolute SHAP value
        shap_explanation["Abs_SHAP"] = shap_explanation["SHAP Value"].abs()
        sorted_shap = shap_explanation.sort_values(
            by="Abs_SHAP", ascending=False).head(3)

        for _, row in sorted_shap.iterrows():
            direction = "increased" if row["SHAP Value"] > 0 else "decreased"
            explanation_text += f"- **{row['Feature']}:** This feature {direction} the likelihood of approval by approximately `{row['SHAP Value']:.2f}` points relative to the average application.\n"

    # Default rejection reasons if no specific XAI insights are prominent or for general context
    if prediction_prob < threshold and application_data.get("rejection_reason", "") != "":
        explanation_text += f"\n**Additional Context:** The system also flagged this application with the rejection reason: '{application_data['rejection_reason']}'.\n"

    explanation_text += "\n"
    return explanation_text


def create_adverse_action_notice(application_data, final_explanation):
    notice = f"**Adverse Action Notice for Loan Application (Applicant ID: {application_data.name})**\n\n"
    notice += "Dear Applicant,\n\n"
    notice += "We regret to inform you that your application for a loan has been declined.\n\n"
    notice += "This decision was based on information obtained from your application. The primary factors contributing to this decision were:\n\n"
    # Extract key reasons from the final_explanation. This is a simplified extraction.
    # In a real system, you'd parse `final_explanation` to get the top negative factors directly.
    # For this demo, we'll just present the relevant part of the explanation.

    # Extracting a summarized reason for adverse action from the full explanation
    reason_lines = []
    if "The primary factors contributing to this decision include:" in final_explanation:
        start_index = final_explanation.find("The primary factors contributing to this decision include:") + len(
            "The primary factors contributing to this decision include:")
        end_index_lime = final_explanation.find(
            "\n**SHAP Insights", start_index)
        end_index_shap = final_explanation.find(
            "\n**Additional Context", start_index)
        end_index = min(idx for idx in [end_index_lime, end_index_shap] if idx != -1) if any(
            idx != -1 for idx in [end_index_lime, end_index_shap]) else len(final_explanation)

        extracted_reasons = final_explanation[start_index:end_index].strip()
        if extracted_reasons:
            notice += extracted_reasons
        else:
            notice += "- Specific reasons derived from our credit model analysis.\n"
    else:
        notice += "- Specific reasons derived from our credit model analysis.\n"

    if application_data.get("rejection_reason") and application_data.get("rejection_reason", "") != "":
        notice += f"\n- Furthermore, the application was flagged for: {application_data['rejection_reason']}.\n"

    notice += "\n\nYou have the right under federal law to receive the following information:\n"
    notice += "- The name and address of the credit bureau or other information source we used.\n"
    notice += "- Your right to a free copy of your credit report from the credit bureau.\n"
    notice += "- Your right to dispute inaccurate or incomplete information in your credit report.\n\n"
    notice += "If you believe any information used in this decision was inaccurate, you have the right to request more specific reasons for the denial.\n\n"
    notice += "If you have any questions or wish to discuss this decision, please contact us at:\n"
    notice += "**Financial Institution Name**\n"
    notice += "Credit Department\n"
    notice += "Phone: 1-800-XXX-XXXX\n"
    notice += "Email: credit@financialinstitution.com\n\n"
    notice += "We appreciate your interest and encourage you to reapply in the future if your financial circumstances improve.\n\n"
    notice += "Sincerely,\n\n"
    notice += "Credit Review Team\n"
    notice += "Financial Institution\n"

    return notice
