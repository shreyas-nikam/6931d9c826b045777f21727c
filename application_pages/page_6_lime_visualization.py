
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    st.markdown(
        """
        # Step 5: LIME Visualization - Picturing Local Impact

        After generating a LIME explanation, visualizing the results is crucial for effective communication.
        As a Quant Analyst, transforming raw feature importance scores into clear plots helps stakeholders
        quickly grasp *why* a specific loan application was rejected or approved.

        **What you're trying to achieve:** Visually represent the LIME explanation for a selected rejected loan application,
        making the individual feature contributions intuitive and easy to interpret.
        This aids in clarifying complex model behavior for a single case.

        **How this page helps:** This page takes the LIME output and generates an importance plot, showing which features
        pushed the decision towards approval or rejection for the selected applicant. This visual evidence strengthens your explanation.
        """
    )

    if "selected_rejected_cases" not in st.session_state or not st.session_state["selected_rejected_cases"]:
        st.warning("Please go to 'Case Identification' to select at least one rejected application first.")
        return

    st.subheader("Visualize LIME Explanation for a Case")

    # Allow selecting one case for LIME visualization
    case_options = {f"Applicant ID: {idx}": idx for idx in st.session_state["selected_rejected_cases"]}
    if not case_options:
        st.warning("No rejected cases with LIME explanations found. Please generate LIME explanations first.")
        return

    selected_case_id_str = st.selectbox(
        "Select an application to visualize its LIME explanation:",
        options=list(case_options.keys()),
        key="lime_viz_case_selection"
    )
    selected_case_id = case_options[selected_case_id_str]

    if f"lime_explanation_{selected_case_id}" not in st.session_state:
        st.warning(f"LIME explanation for Applicant ID {selected_case_id} not found. Please generate it on the 'LIME Explanation' page first.")
        return

    lime_explanation_list = st.session_state[f"lime_explanation_{selected_case_id}"]

    st.markdown(f"**Visualizing LIME for Applicant ID:** `{selected_case_id}`")

    # Prepare data for plotting
    features = [exp[0] for exp in lime_explanation_list]
    weights = [exp[1] for exp in lime_explanation_list]

    # Create a DataFrame for easier plotting
    lime_df = pd.DataFrame({
        "Feature": features,
        "Weight": weights
    })
    lime_df = lime_df.sort_values(by="Weight", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["red" if w < 0 else "green" for w in lime_df["Weight"]]
    ax.barh(lime_df["Feature"], lime_df["Weight"], color=colors)
    ax.set_xlabel("Contribution to Prediction")
    ax.set_ylabel("Feature")
    ax.set_title(f"LIME Feature Contributions for Applicant ID {selected_case_id}")
    ax.axvline(0, color="grey", linestyle="--")
    st.pyplot(fig)
    plt.close(fig) # Close the plot to prevent display issues

    st.markdown(
        """
        **How the underlying concept or AI method supports this action:** This visualization leverages basic bar chart principles to show directed feature importance.
        Positive bars indicate features pushing towards the "approved" class, while negative bars push towards "rejected".
        This direct visual mapping allows non-technical stakeholders, like regulators or loan officers, to quickly understand the primary factors for a specific decision,
        fulfilling the need for transparent communication.
        """
    )

