
import streamlit as st
import pandas as pd
from utils import explain_with_lime

def main():
    st.markdown(
        """
        # Step 4: LIME Explanation - Understanding Local Decisions

        As a Quant Analyst, when faced with a specific rejected loan application, one of your primary tools for explanation is LIME (Local Interpretable Model-agnostic Explanations).
        LIME helps you understand *why* a black-box model made a particular prediction for a single instance by approximating it with an interpretable local model.

        **What you're trying to achieve:** Generate a localized explanation for a chosen rejected loan application, identifying the key features that drove its individual rejection.
        This insight is crucial for providing specific feedback to applicants or defending a decision to regulators.

        **How this page helps:** You will select a specific rejected case and trigger a LIME analysis. The output will highlight the features that were most influential
        in the model's decision for that particular applicant, providing a concrete, case-by-case rationale.
        """
    )

    if "selected_rejected_cases" not in st.session_state or not st.session_state["selected_rejected_cases"]:
        st.warning("Please go to 'Case Identification' to select at least one rejected application first.")
        return
    if "credit_model" not in st.session_state or st.session_state["credit_model"] is None:
        st.warning("Please go to 'Model Training' to train a model first.")
        return
    if "X_train" not in st.session_state or st.session_state["X_train"] is None:
        st.warning("Training data (X_train) not found in session state. Please retrain the model.")
        return

    st.subheader("Apply LIME to a Selected Case")

    # Allow selecting one case for LIME explanation from the previously selected rejected cases
    case_options = {f"Applicant ID: {idx}": idx for idx in st.session_state["selected_rejected_cases"]}
    selected_case_id_str = st.selectbox(
        "Select an application for LIME explanation:",
        options=list(case_options.keys()),
        key="lime_case_selection"
    )
    selected_case_id = case_options[selected_case_id_str]

    application_data = st.session_state["loan_data"].loc[selected_case_id]
    model = st.session_state["credit_model"]
    features = st.session_state["model_features"]
    X_train = st.session_state["X_train"]

    st.markdown(f"**Analyzing Applicant ID:** `{selected_case_id}`")
    st.dataframe(application_data.to_frame().T)

    if st.button("Generate LIME Explanation", key="generate_lime_button"):
        with st.spinner("Generating LIME explanation..."):            
            lime_explanation_list = explain_with_lime(model, X_train, application_data[features], features)
            st.session_state[f"lime_explanation_{selected_case_id}"] = lime_explanation_list
            st.success("LIME explanation generated!")

    if f"lime_explanation_{selected_case_id}" in st.session_state:
        st.subheader(f"LIME Explanation for Applicant ID: {selected_case_id}")
        st.markdown(
            r"""
            **Interpreting LIME Output:** The list below shows features that are locally important for this specific decision.
            Each tuple `(feature, weight)` indicates how much that feature (and its value) contributes to the model's prediction for this instance.
            A positive weight suggests it pushes towards "Approved", while a negative weight pushes towards "Rejected".
            """
        )
        for feature, weight in st.session_state[f"lime_explanation_{selected_case_id}"]:
            st.write(f"- `{feature}`: `{weight:.4f}`")

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:** LIME works by creating a local, interpretable model (like a linear model)
            around the prediction of a single instance. It perturbs the instance's features and observes how the black-box model's prediction changes.
            This allows us to attribute importance to features for *that specific decision*, which is critical for satisfying inquiries about individual cases.
            """
        )

