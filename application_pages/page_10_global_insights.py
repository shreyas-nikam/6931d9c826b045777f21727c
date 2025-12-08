
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt


def main():
    st.markdown(
        """
        # Step 7: Global Insights - Understanding Overall Feature Impact

        While individual explanations (LIME and SHAP for specific cases) are crucial for justifying single decisions,
        as a Quant Analyst, you also need to understand the *overall* behavior of the credit scoring model.
        Global interpretability, often achieved with SHAP summary plots, helps in identifying the most influential features across the entire dataset,
        which is vital for model validation, development, and high-level regulatory reporting.

        **What you're trying to achieve:** Gain insights into which features are most important for the model's predictions across all loan applications,
        and how these features generally influence the outcome (e.g., positive or negative correlation with approval).
        This informs broader policy decisions and model improvements.

        **How this page helps:** This page generates a SHAP summary plot, providing a birds-eye view of feature importance.
        It helps you answer questions like, "Which are the top 5 factors driving loan approvals or rejections in general?"
        """
    )

    if "credit_model" not in st.session_state or st.session_state["credit_model"] is None:
        st.warning("Please go to 'Model Training' to train a model first.")
        return
    if "X_train" not in st.session_state or st.session_state["X_train"] is None:
        st.warning(
            "Training data (X_train) not found in session state. Please retrain the model.")
        return

    st.subheader("Global Feature Importance with SHAP Summary Plot")

    model = st.session_state["credit_model"]
    X_train = st.session_state["X_train"]
    features = st.session_state["model_features"]

    st.markdown(
        r"""
        To generate global insights, we compute SHAP values for a significant portion of the training data.
        This provides a comprehensive view of how each feature influences the model's output overall.

        **SHAP Summary Plot:**
        - Each point on the plot is a Shapley value for a feature and an instance.
        - The position on the x-axis shows the SHAP value (impact on model output).
        - Color indicates the feature value (e.g., red for high, blue for low).
        - Vertical dispersion shows interaction effects.
        """
    )

    if st.button("Generate Global SHAP Summary Plot", key="generate_global_shap_button"):
        with st.spinner("Calculating global SHAP values and generating plot... This may take a while for large datasets."):
            # For global explanation, we often use a sample of the training data
            # Using X_train directly or a subset of it
            explainer = shap.TreeExplainer(model)
            shap_values_global = explainer.shap_values(X_train)

            # Assuming binary classification, we care about the positive class (index 1)
            if isinstance(shap_values_global, list):
                # For the "Approved" class
                shap_values_for_plot = shap_values_global[1]
            else:
                shap_values_for_plot = shap_values_global

            st.session_state["global_shap_values"] = shap_values_for_plot
            # Store data used for these SHAP values
            st.session_state["global_shap_data"] = X_train
            st.success("Global SHAP values calculated!")

    if "global_shap_values" in st.session_state and "global_shap_data" in st.session_state:
        st.subheader("Overall Feature Impact")
        
        # Call SHAP summary plot with size control
        shap.summary_plot(
            st.session_state["global_shap_values"], 
            st.session_state["global_shap_data"],
            feature_names=features, 
            show=False, 
            plot_type="dot",
            plot_size=(10, 6)  # Slightly taller to accommodate labels
        )
        
        # Get the figure and adjust spacing to prevent cutoff
        fig = plt.gcf()
        plt.subplots_adjust(bottom=0.2, top=0.88, left=0.15, right=0.95)  # More bottom space for xlabel
        
        # Add explanatory text as a legend - positioned to not overlap
        plt.text(
            0.02, 0.98,
            "Legend: Each dot = one instance | X-axis = SHAP value | Color = Feature value (red=high, blue=low)",
            transform=fig.transFigure,
            fontsize=6.5, 
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        st.pyplot(fig, use_container_width=False)
        plt.close('all')

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:** The SHAP summary plot condenses the insights from thousands of individual SHAP explanations into a single visualization.
            It shows not only which features are most important globally, but also the direction of their impact (positive or negative) and their distribution of impact values.
            This global view is indispensable for model development teams to ensure feature importance aligns with domain expertise, for risk management to assess overall model fairness,
            and for regulators who require a high-level understanding of model drivers.
            """
        )
