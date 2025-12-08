
import streamlit as st
import pandas as pd
from utils import train_credit_model


def main():
    st.markdown(
        """
        # Step 2: Model Training - Building Our Credit Scoring Engine

        Now that you've reviewed the data, the next critical step for any Quant Analyst facing a regulatory inquiry
        is to demonstrate the model that powers the credit decisions. On this page, you will train a credit scoring model
        using the historical data. This model will then be used to predict loan approval probabilities, which we will later explain.

        **What you're trying to achieve:** Establish a robust and fair credit scoring model that accurately predicts loan outcomes.
        This trained model is the subject of our explainability exercise, allowing us to generate explanations for its decisions.

        **How this page helps:** You can initiate the training process and see immediate feedback on the model's performance.
        This interaction simulates the practical step of deploying or validating a model within a financial institution.
        """
    )

    if "loan_data" not in st.session_state or st.session_state["loan_data"] is None:
        st.warning("Please go to 'Data Inspection' to load the loan data first.")
        return

    st.subheader("Train Credit Scoring Model")

    st.markdown("**Model Type:** Random Forest Classifier")

    if st.button("Train Model", key="train_model_button"):
        with st.spinner("Training Random Forest Classifier... This may take a moment."):
            model, features, X_train, accuracy = train_credit_model(
                st.session_state["loan_data"])
            st.session_state["credit_model"] = model
            st.session_state["model_features"] = features
            # Store for LIME/SHAP explainers
            st.session_state["X_train"] = X_train
            st.session_state["model_accuracy"] = accuracy
            st.success("Random Forest Classifier trained successfully!")
            st.write(f"Model Accuracy: {accuracy:.2f}")

    if "credit_model" in st.session_state and st.session_state["credit_model"] is not None:
        st.subheader("Trained Model Summary")
        st.markdown("**Model Type:** Random Forest Classifier")
        st.markdown(
            f"**Accuracy on Test Set:** `{st.session_state['model_accuracy']:.2f}`")
        st.markdown(
            f"**Features Used:** `{', '.join(st.session_state['model_features'])}`")

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:**
            
            The **Random Forest Classifier** is an ensemble learning method that combines multiple decision trees to make predictions.
            Each tree votes on the final prediction, and the majority vote determines the outcome. This approach offers several advantages:
            
            - **Robustness:** Less prone to overfitting compared to single decision trees
            - **Feature Importance:** Can naturally provide feature importance scores
            - **Non-linear Relationships:** Handles complex, non-linear patterns in data
            - **Tree-based Structure:** Well-suited for SHAP explanations due to its interpretable tree architecture
            
            By training this model on historical loan data, we've created a credit scoring engine that can predict loan approval outcomes.
            The next step is to identify critical rejected applications that require detailed explanation using XAI techniques.
            
            **Next Steps:** Navigate to "Case Identification" to select specific rejected applications for in-depth analysis.
            """
        )
