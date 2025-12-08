
import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from utils import explain_with_shap


def main():
    st.markdown(
        """
        # Step 5: SHAP Explanation & Visualization - Deeper Dive into Feature Impact

        Beyond LIME's local explanations, SHAP (SHapley Additive exPlanations) provides a unified framework to explain the output of any machine learning model.
        As a Quant Analyst, SHAP values are incredibly powerful for understanding not just individual predictions, but also global model behavior by consistently attributing
        each feature's contribution to a prediction.

        **What you're trying to achieve:** Apply SHAP analysis to a selected rejected loan application to gain a robust and theoretically sound explanation
        of individual feature contributions, with interactive visualizations including force plots and dependence plots.

        **How this page helps:** You will select a rejected case, initiate a SHAP analysis, and see both numerical SHAP values and powerful visualizations
        that reveal feature impacts and interactions.
        """
    )

    if "selected_rejected_cases" not in st.session_state or not st.session_state["selected_rejected_cases"]:
        st.warning(
            "Please go to 'Case Identification' to select at least one rejected application first.")
        return
    if "credit_model" not in st.session_state or st.session_state["credit_model"] is None:
        st.warning("Please go to 'Model Training' to train a model first.")
        return
    if "X_train" not in st.session_state or st.session_state["X_train"] is None:
        st.warning(
            "Training data (X_train) not found in session state. Please retrain the model.")
        return

    st.subheader("Apply SHAP to a Selected Case")

    # Allow selecting one case for SHAP explanation
    case_options = {
        f"Applicant ID: {idx}": idx for idx in st.session_state["selected_rejected_cases"]}
    selected_case_id_str = st.selectbox(
        "Select an application for SHAP explanation:",
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
            explainer, shap_values = explain_with_shap(
                model, X_train, application_data[features])

            # For binary classification with TreeExplainer, shap_values can be:
            # - A list of 2 arrays (old format): [array(n_samples, n_features), array(n_samples, n_features)]
            # - A single array (new format): array(n_samples, n_features, n_classes)
            st.session_state[f"shap_explainer_{selected_case_id}"] = explainer

            # Extract SHAP values for class 1 (approval) and ensure it's 1D
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Old format: list of arrays, one per class
                shap_vals = shap_values[1]
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
                    shap_vals = shap_vals[0]  # Take first sample
                expected_val = explainer.expected_value[1] if isinstance(explainer.expected_value, (
                    list, np.ndarray)) and len(explainer.expected_value) > 1 else explainer.expected_value
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # New format: (n_samples, n_features, n_classes)
                # Extract class 1 (approval) for first sample
                # [sample_0, all_features, class_1]
                shap_vals = shap_values[0, :, 1]
                expected_val = explainer.expected_value[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                # Single class output: (n_samples, n_features)
                shap_vals = shap_values[0]  # Take first sample
                expected_val = explainer.expected_value[0] if isinstance(
                    explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                # Fallback
                shap_vals = shap_values
                expected_val = explainer.expected_value

            st.session_state[f"shap_values_{selected_case_id}"] = shap_vals
            st.session_state[f"shap_base_value_{selected_case_id}"] = expected_val
            st.success("SHAP explanation generated!")

    if (f"shap_explainer_{selected_case_id}" in st.session_state and
        f"shap_values_{selected_case_id}" in st.session_state and
            f"shap_base_value_{selected_case_id}" in st.session_state):
        
        st.subheader(f"SHAP Explanation for Applicant ID: {selected_case_id}")
        
        # Display numerical explanation
        st.markdown("#### SHAP Values")
        st.markdown(
            r"""
            **Interpreting SHAP Values:** SHAP values represent the contribution of each feature to the prediction for a specific instance, relative to the average prediction.
            A positive SHAP value means the feature increases the prediction (e.g., towards approval), while a negative value decreases it (e.g., towards rejection).
            The sum of SHAP values plus the `base value` (average prediction) equals the model's raw output for this instance.
            """
        )
        shap_values_for_case = st.session_state[f"shap_values_{selected_case_id}"]
        base_value = st.session_state[f"shap_base_value_{selected_case_id}"]

        # Ensure shap_values is 1D (flatten if necessary)
        if isinstance(shap_values_for_case, np.ndarray) and shap_values_for_case.ndim > 1:
            shap_values_for_case = shap_values_for_case.flatten()

        # Ensure the length matches the number of features
        if len(shap_values_for_case) != len(features):
            st.error(
                f"Mismatch: {len(features)} features but {len(shap_values_for_case)} SHAP values. Please regenerate the SHAP explanation.")
            return

        st.markdown(
            f"**Base Value (Average Model Output):** `{base_value:.4f}`")
        st.markdown(
            f"**Model's Predicted Probability for this Case (raw output before sigmoid):** `{base_value + shap_values_for_case.sum():.4f}`")

        shap_df = pd.DataFrame({
            "Feature": features,
            "SHAP Value": shap_values_for_case
        }).sort_values(by="SHAP Value", ascending=False)

        st.dataframe(shap_df)

        # Visualizations
        st.markdown("---")
        st.markdown("#### SHAP Force Plot (Waterfall Visualization)")
        st.markdown(
            r"""
            The **Waterfall Plot** visualizes how features contribute to pushing the model's output from the `base value` (average prediction)
            to the actual prediction for this specific instance. Features pushing the prediction higher (towards approval) are shown in red/pink,
            and those pushing it lower (towards rejection) are in blue.
            """
        )
        
        explainer = st.session_state[f"shap_explainer_{selected_case_id}"]
        application_data_features = application_data[features]

        fig, ax = plt.subplots(figsize=(12, 2))
        shap.waterfall_plot(shap.Explanation(values=shap_values_for_case,
                                             base_values=base_value,
                                             data=application_data_features.values,
                                             feature_names=features),
                            max_display=len(features), show=False)
        plt.title(f"SHAP Waterfall Plot for Applicant ID {selected_case_id}")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### SHAP Dependence Plot (Feature Interaction)")
        st.markdown(
            r"""
            The **Dependence Plot** shows the relationship between a single feature and the predicted outcome, often revealing interactions with other features.
            This helps in understanding how changes in a feature's value influence the model's prediction for individual instances, and if this influence is modulated by another feature.
            """
        )

        features_for_dependence = features
        selected_feature = st.selectbox(
            "Select a feature for Dependence Plot:",
            options=features_for_dependence,
            index=features_for_dependence.index("credit_score") if "credit_score" in features_for_dependence else 0,
            key="shap_dependence_feature_selection"
        )

        interaction_features = [
            f for f in features_for_dependence if f != selected_feature]
        selected_interaction_feature = st.selectbox(
            "Select an interaction feature (optional):",
            options=["None"] + interaction_features,
            index=0,
            key="shap_interaction_feature_selection"
        )

        if selected_interaction_feature == "None":
            interaction_index = None
        else:
            interaction_index = selected_interaction_feature

        # For dependence plot, we need to use the full dataset SHAP values, not just a single instance
        # Generate SHAP values for the entire training set if not already computed
        if "shap_values_all" not in st.session_state:
            with st.spinner("Computing SHAP values for all training samples for dependence plot..."):
                # For efficiency, use a sample of training data
                X_sample = X_train.sample(min(100, len(X_train)), random_state=42)
                explainer_all = shap.TreeExplainer(model)
                shap_values_all = explainer_all.shap_values(X_sample)

                # Extract for class 1 (approval)
                if isinstance(shap_values_all, list) and len(shap_values_all) == 2:
                    shap_values_all_class1 = shap_values_all[1]
                elif isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3:
                    shap_values_all_class1 = shap_values_all[:, :, 1]
                else:
                    shap_values_all_class1 = shap_values_all

                st.session_state["shap_values_all"] = shap_values_all_class1
                st.session_state["X_sample_for_shap"] = X_sample

        fig_dp, ax_dp = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            ind=selected_feature,
            shap_values=st.session_state["shap_values_all"],
            features=st.session_state["X_sample_for_shap"].values,
            feature_names=features,
            interaction_index=interaction_index,
            show=False,  # Important for Streamlit
            ax=ax_dp
        )
        st.pyplot(fig_dp)
        plt.close(fig_dp)

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:** SHAP is based on game theory and Shapley values, ensuring fair distribution of prediction credit among features.
            This means the sum of feature contributions (SHAP values) plus a base value (the average model output for the dataset) reconstructs the actual prediction.
            This robust attribution method provides a consistent and theoretically sound way to explain individual predictions, critical for rigorous regulatory compliance.
            
            The visualizations transform complex SHAP values into intuitive insights. The waterfall plot shows the cumulative effect of each feature's SHAP value,
            starting from the base value to the final output. The dependence plot reveals non-linear effects and interactions by displaying how the SHAP value
            for a specific feature changes as the value of that feature changes, with coloring based on another feature to identify conditional relationships.
            """
        )
