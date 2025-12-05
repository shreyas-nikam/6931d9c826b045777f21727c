
import streamlit as st
import pandas as pd
from utils import generate_synthetic_data


def main():
    st.markdown(
        """
        # Step 1: Data Review - Understanding Our Loan Portfolio

        As a Quantitative Analyst, your first step in any inquiry is to understand the data that underpins our decisions.
        This page allows you to inspect the historical loan application data, which serves as the foundation for our credit scoring model.
        Familiarizing yourself with the features and their distributions is crucial before delving into model explanations.

        **What you're trying to achieve:** Gain an initial understanding of the loan application dataset, its structure, and the types of applicants we process.
        This helps you to contextualize subsequent model explanations within the real-world characteristics of our customer base.

        **How this page helps:** You can either generate a synthetic dataset to simulate real-world data or upload your own (for future enhancements).
        Inspecting the raw data and summary statistics gives you immediate insights into the scale and nature of the loan applications.
        """
    )

    if "loan_data" not in st.session_state:
        st.session_state["loan_data"] = None

    st.subheader("Load Loan Application Data")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate Synthetic Data"):
            st.session_state["loan_data"] = generate_synthetic_data()
            st.success("Synthetic data generated successfully!")
    # with col2:
    #     # Placeholder for file uploader - not implementing actual file upload for this exercise
    #     st.info("File upload feature will be enabled in future versions for real data.")
        # uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        # if uploaded_file is not None:
        #     st.session_state["loan_data"] = pd.read_csv(uploaded_file)
        #     st.success("Data uploaded successfully!")

    if st.session_state["loan_data"] is not None:
        st.subheader("Raw Data Preview")
        st.dataframe(st.session_state["loan_data"].head())

        st.subheader("Data Description")
        st.write(st.session_state["loan_data"].describe())

        st.subheader("Loan Approval Distribution")
        approval_counts = st.session_state["loan_data"]["loan_approved"].value_counts(
        ).rename(index={0: "Rejected", 1: "Approved"})
        st.bar_chart(approval_counts)

        st.markdown(
            """
            **How the underlying concept or AI method supports this action:** This initial data inspection is fundamental to any data science workflow.
            Understanding the raw data helps in identifying potential biases, feature distributions, and the overall quality of information available.
            While not an AI method itself, it directly informs the feature engineering and model selection processes that follow.
            """
        )
    else:
        st.info("Please generate or upload data to proceed.")
