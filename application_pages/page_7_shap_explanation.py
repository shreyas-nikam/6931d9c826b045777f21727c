
import streamlit as st
import pandas as pd
from utils import explain_with_shap
import shap # Import shap for explainer objects

def main():
    st.markdown(
        """
        # Step 6: SHAP Explanation - Deeper Dive into Feature Impact

        Beyond LIME's local explanations, SHAP (SHapley Additive exPlanations) provides a unified framework to explain the output of any machine learning model.
        As a Quant Analyst, SHAP values are incredibly powerful for understanding not just individual predictions, but also global model behavior by consistently attributing
        each feature's contribution to a prediction.

        **What you're trying to achieve:** Apply SHAP analysis to another selected rejected loan application to gain a more robust and theoretically sound explanation
        of individual feature contributions. SHAP provides a rigorous method for attributing model output.

        **How this page helps:** You will select a different rejected case and initiate a SHAP analysis. This will generate SHAP values that precisely quantify
        how much each feature contributes to pushing the model's output from the base value to the final predicted value for that instance.
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

    st.subheader("Apply SHAP to a Selected Case")

    # Allow selecting one case for SHAP explanation
    case_options = {f"Applicant ID: {idx}": idx for idx in st.session_state["selected_rejected_cases"]}
    selected_case_id_str = st.selectbox(
        "Select an application for SHAP explanation (ideally a different one from LIME):",
        options=list(case_options.keys()),
        key="shap_case_selection"
    )
    selected_case_id = case_options[selected_case_id_str]

    application_data = st.session_state["loan_data"].loc[selected_case_id]
    model = st.session_state["credit_model"]
    features = st.session_state["model_features"]
    X_train = st.session_state["X_train"]

    st.markdown(f"**Analyzing Applicant ID:** `{selected_case_id}`")
    st.dataframe(application_data.to_frame().T)

    if st.button("Generate SHAP Explanation", key="generate_shap_button"):
        with st.spinner("Generating SHAP explanation... This might take a moment for larger models."):
            # SHAP explainer needs the training data or a background dataset
            explainer, shap_values = explain_with_shap(model, X_train, application_data[features])
            
            # For classification, shap_values is a list of arrays (one per class). We need the explanation for the "rejected" class (class 0).
            # Or, if we are explaining the probability of approval, we use class 1.
            # Let's assume we want to explain the probability of approval (class 1)
            st.session_state[f"shap_explainer_{selected_case_id}"] = explainer
            st.session_state[f"shap_values_{selected_case_id}"] = shap_values[1] if isinstance(shap_values, list) else shap_values # Take values for the positive class (approval)
            st.session_state[f"shap_base_value_{selected_case_id}"] = explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value # Base value for positive class
            st.success("SHAP explanation generated!")

    if f"shap_explainer_{selected_case_id}" in st.session_state and f"shap_values_{selected_case_id}" in st.session_state:
        st.subheader(f"SHAP Explanation for Applicant ID: {selected_case_id}")
        st.markdown(
            r"""
            **Interpreting SHAP Values:** SHAP values represent the contribution of each feature to the prediction for a specific instance, relative to the average prediction.
            A positive SHAP value means the feature increases the prediction (e.g., towards approval), while a negative value decreases it (e.g., towards rejection).
            The sum of SHAP values plus the `base value` (average prediction) equals the model's raw output for this instance.
            """
        )
        shap_values_for_case = st.session_state[f"shap_values_{selected_case_id}"]
        base_value = st.session_state[f"shap_base_value_{selected_case_id}"]

        st.markdown(f"**Base Value (Average Model Output):** `{base_value:.4f}`")
        st.markdown(f"**Model's Predicted Probability for this Case (raw output before sigmoid):** `{base_value + shap_values_for_case.sum():.4f}`")

        shap_df = pd.DataFrame({
            "Feature": features,
            "SHAP Value": shap_values_for_case
        }).sort_values(by="SHAP Value", ascending=False)

        st.dataframe(shap_df)

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:** SHAP is based on game theory and Shapley values, ensuring fair distribution of prediction credit among features.
            This means the sum of feature contributions (SHAP values) plus a base value (the average model output for the dataset) reconstructs the actual prediction.
            This robust attribution method provides a consistent and theoretically sound way to explain individual predictions, critical for rigorous regulatory compliance.
            """
        )

