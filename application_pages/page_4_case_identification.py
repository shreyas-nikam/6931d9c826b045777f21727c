
import streamlit as st
import pandas as pd
from utils import identify_critical_applications

def main():
    st.markdown(
        """
        # Step 3: Case Identification - Pinpointing Critical Decisions

        With our model trained, the immediate next step in addressing a regulatory inquiry is to identify the specific cases
        that require detailed explanation. This often involves focusing on rejected applications, especially those where the
        decision might be ambiguous or has a significant impact on the applicant.

        **What you're trying to achieve:** Identify and select individual loan applications that were rejected by the model,
        and which will be the focus of our explainability efforts using LIME and SHAP. This step is about narrowing down the scope to critical instances.

        **How this page helps:** You can adjust a threshold to filter applications and then manually select specific rejected cases for deeper analysis.
        This mimics the real-world process of an analyst focusing on particular cases flagged for review.
        """
    )

    if "credit_model" not in st.session_state or st.session_state["credit_model"] is None:
        st.warning("Please go to 'Model Training' to train a model first.")
        return
    if "loan_data" not in st.session_state or st.session_state["loan_data"] is None:
        st.warning("Please go to 'Data Inspection' to load the loan data first.")
        return

    st.subheader("Identify Critical Rejected Applications")

    model = st.session_state["credit_model"]
    loan_data = st.session_state["loan_data"]
    features = st.session_state["model_features"]

    st.markdown(
        r"""
        **Decision Threshold:** The model predicts a probability of approval. By adjusting this threshold, you can define what constitutes a "rejection" for review.
        For instance, if the predicted probability of approval $P(	ext{Approved})$ falls below this threshold, the application is considered for rejection.
        """
    )
    
    prediction_threshold = st.slider(
        "Select Probability Threshold for Rejection (P(Approved) < threshold)",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        key="rejection_threshold"
    )

    rejected_cases = identify_critical_applications(loan_data, model, features, threshold=prediction_threshold)

    if not rejected_cases.empty:
        st.subheader(f"Rejected Applications (P(Approved) < {prediction_threshold:.2f})")
        st.markdown("Select individual cases below for detailed XAI analysis.")

        # Display rejected cases for selection
        selected_indices = []
        for i, row in rejected_cases.head(10).iterrows(): # Show top 10 rejected for selection
            checkbox_label = f"Applicant ID: {i} - P(Approved): {row[