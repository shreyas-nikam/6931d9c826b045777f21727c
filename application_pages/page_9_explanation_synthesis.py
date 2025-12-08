
import streamlit as st
import pandas as pd
from utils import generate_human_readable_explanation, create_adverse_action_notice

def main():
    st.markdown(
        """
        # Step 6: Explanation Synthesis - Crafting Human-Readable Justifications

        The ultimate goal of explainable AI in a financial context is to translate technical insights into clear, actionable, and human-readable explanations.
        As a Quant Analyst, you must be able to synthesize the LIME and SHAP findings into coherent narratives for regulators, internal stakeholders, and even the applicants themselves.

        **What you're trying to achieve:** Develop a comprehensive, plain-language explanation for a selected rejected loan application, drawing upon the insights gained from LIME and SHAP analysis.
        This involves translating feature contributions into business logic and potential recommendations.

        **How this page helps:** You will select a rejected case, review its LIME and SHAP explanations (if available), and then generate a synthesized explanation.
        You can also add custom comments to refine the narrative, mimicking the analyst's role in crafting precise communications.
        """
    )

    if "selected_rejected_cases" not in st.session_state or not st.session_state["selected_rejected_cases"]:
        st.warning("Please go to 'Case Identification' to select at least one rejected application first.")
        return
    if "loan_data" not in st.session_state or st.session_state["loan_data"] is None:
        st.warning("Please go to 'Data Inspection' to load the loan data first.")
        return

    st.subheader("Synthesize Explanations for a Rejected Case")

    case_options = {f"Applicant ID: {idx}": idx for idx in st.session_state["selected_rejected_cases"]}
    if not case_options:
        st.warning("No rejected cases selected for explanation synthesis.")
        return

    selected_case_id_str = st.selectbox(
        "Select a rejected application for explanation synthesis:",
        options=list(case_options.keys()),
        key="synthesis_case_selection"
    )
    selected_case_id = case_options[selected_case_id_str]

    application_data = st.session_state["loan_data"].loc[selected_case_id]

    st.markdown(f"**Application Details for Applicant ID:** `{selected_case_id}`")
    st.dataframe(application_data.to_frame().T)

    lime_exp = st.session_state.get(f"lime_explanation_{selected_case_id}", None)
    shap_exp = None
    if f"shap_values_{selected_case_id}" in st.session_state:
        shap_exp = pd.DataFrame({
            "Feature": st.session_state["model_features"],
            "SHAP Value": st.session_state[f"shap_values_{selected_case_id}"]
        }).sort_values(by="SHAP Value", ascending=False)

    st.markdown("---")
    st.subheader("Review Explanations (LIME & SHAP)")
    
    if lime_exp:
        st.markdown("#### LIME Insights:")
        for feature, weight in lime_exp:
            st.write(f"- `{feature}`: `{weight:.4f}`")
    else:
        st.info("No LIME explanation found for this case. Please generate it on the 'LIME Explanation & Visualization' page.")
    
    if shap_exp is not None:
        st.markdown("#### SHAP Insights:")
        st.dataframe(shap_exp)
    else:
        st.info("No SHAP explanation found for this case. Please generate it on the 'SHAP Explanation & Visualization' page.")

    st.markdown("---")
    st.subheader("Craft Your Human-Readable Explanation")

    custom_comments = st.text_area(
        "Add specific comments or refine the explanation for stakeholders (e.g., policy implications, next steps for applicant):",
        height=150,
        key="custom_explanation_comments"
    )

    if st.button("Generate Final Explanation & Adverse Action Notice", key="generate_final_explanation_button"):
        with st.spinner("Generating synthesized explanation..."):            
            final_explanation = generate_human_readable_explanation(application_data, lime_explanation=lime_exp, shap_explanation=shap_exp)
            if custom_comments:
                final_explanation += f"\n\n**Analyst's Additional Comments:**\n{custom_comments}"

            st.session_state[f"final_explanation_{selected_case_id}"] = final_explanation
            
            # Generate Adverse Action Notice
            adverse_notice = create_adverse_action_notice(application_data, final_explanation)
            st.session_state[f"adverse_action_notice_{selected_case_id}"] = adverse_notice
            
            st.success("Explanation and Adverse Action Notice generated!")

    if f"final_explanation_{selected_case_id}" in st.session_state:
        st.subheader(f"Synthesized Explanation for Applicant ID: {selected_case_id}")
        st.markdown(st.session_state[f"final_explanation_{selected_case_id}"])

        st.subheader(f"Adverse Action Notice for Applicant ID: {selected_case_id}")
        st.markdown(st.session_state[f"adverse_action_notice_{selected_case_id}"])
        
        st.download_button(
            label="Download Explanation as Text",
            data=st.session_state[f"final_explanation_{selected_case_id}"],
            file_name=f"explanation_applicant_{selected_case_id}.txt",
            mime="text/plain",
            key="download_explanation"
        )
        st.download_button(
            label="Download Adverse Action Notice as Text",
            data=st.session_state[f"adverse_action_notice_{selected_case_id}"],
            file_name=f"adverse_action_notice_{selected_case_id}.txt",
            mime="text/plain",
            key="download_adverse_notice"
        )

    st.markdown(
        """
        **How the underlying concept or AI method supports this action:** This step is where the theoretical aspects of XAI (LIME and SHAP) are translated into practical, enforceable business outcomes.
        By combining quantified feature impacts with qualitative reasoning (your comments), you create a robust justification.
        This directly supports regulatory compliance by ensuring that all decisions, especially adverse ones, are transparent, understandable, and defensible, building trust and fairness in AI systems.
        """
    )
