
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

    model_options = ["Random Forest Classifier"] # In a real scenario, more options would be available
    selected_model = st.selectbox(
        label="Select Model Type",
        options=model_options,
        index=model_options.index("Random Forest Classifier"),
        key="model_selection"
    )

    if st.button("Train Model", key="train_model_button"):
        with st.spinner(f"Training {selected_model}... This may take a moment."):
            model, features, X_train, accuracy = train_credit_model(st.session_state["loan_data"])
            st.session_state["credit_model"] = model
            st.session_state["model_features"] = features
            st.session_state["X_train"] = X_train # Store for LIME/SHAP explainers
            st.session_state["model_accuracy"] = accuracy
            st.success(f"{selected_model} trained successfully!")
            st.write(f"Model Accuracy: {accuracy:.2f}")

    if "credit_model" in st.session_state and st.session_state["credit_model"] is not None:
        st.subheader("Trained Model Summary")
        st.markdown(f"**Model Type:** {selected_model}")
        st.markdown(f"**Accuracy on Test Set:** `{st.session_state[