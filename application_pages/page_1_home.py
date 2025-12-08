
import streamlit as st


def main():

    st.markdown("""
In this lab, you step into the shoes of a **Quantitative Analyst** at a leading financial institution.
Your institution is facing increased regulatory scrutiny regarding its automated credit decision-making process.
Specifically, there's a demand for greater transparency, especially concerning rejected loan applications.

Your mission is to leverage **Explainable AI (XAI)** techniques to justify loan application outcomes.
Navigate through each page to complete the workflow and test your knowledge with the final quiz!
""")

    st.markdown(
        """
        # Navigating Regulatory Scrutiny: Explaining Credit Decisions with AI

        Welcome, **Quantitative Analyst**! In today's financial landscape, transparency in AI-driven decisions isn't just a nice-to-have—it's a regulatory requirement. Your institution faces mounting pressure to explain automated credit decisions, particularly loan rejections, in clear, defensible terms.

        ## The Challenge

        Financial institutions leveraging AI for credit scoring face a critical dilemma:
        - **Black-box models** provide accuracy but lack transparency
        - **Regulatory bodies** demand clear explanations for adverse actions
        - **Customers** deserve understandable reasons for rejected applications
        - **Auditors** require defensible, documented decision-making processes

        ## Your Mission

        As a Quantitative Analyst, you'll navigate this complex landscape using **Explainable AI (XAI)** techniques. This application guides you through a complete workflow—from data inspection to generating regulatory-compliant adverse action notices.

        ## The Workflow

        Follow these systematic steps to master explainable credit scoring:

        ### 1. **Data Inspection**
        Review historical loan application data to understand the features and distributions that drive credit decisions. Familiarize yourself with applicant profiles before diving into model explanations.

        ### 2. **Model Training**
        Build a Random Forest credit scoring model that serves as your black-box decision engine. This classifier will determine loan approvals and rejections based on applicant characteristics.

        ### 3. **Case Identification**
        Identify critical rejected loan applications that require detailed explanation. Filter cases based on probability thresholds to focus on borderline or high-risk rejections.

        ### 4. **LIME Explanation & Visualization**
        Apply **LIME (Local Interpretable Model-agnostic Explanations)** to understand why a specific application was rejected. LIME provides local, instance-level explanations by approximating the model's behavior around a single prediction. Visualize these insights through intuitive bar charts showing feature contributions.

        ### 5. **SHAP Explanation & Visualization**
        Use **SHAP (SHapley Additive exPlanations)** for theoretically grounded, consistent local explanations. Generate SHAP force plots (waterfall visualizations) and dependence plots to reveal feature interactions and impacts. These visualizations help you understand not just what features matter, but how they interact.

        ### 6. **Explanation Synthesis**
        Combine LIME and SHAP insights into human-readable explanations. Draft formal **Adverse Action Notices** that meet regulatory requirements while remaining understandable to customers.

        ### 7. **Global Insights**
        Analyze overall model behavior using SHAP summary plots. Understand which features are most influential across all predictions, identifying potential biases or systemic patterns.

        ## Why This Matters

        **Regulatory Compliance:** Laws like the Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA) mandate specific explanations for adverse credit actions.

        **Customer Trust:** Clear explanations build confidence in your institution's fairness and decision-making processes.

        **Risk Management:** Understanding model behavior helps identify potential biases, errors, or discriminatory patterns before they become liability issues.

        **Auditability:** Document your decision-making process to withstand regulatory scrutiny and internal audits.

        ## Key Techniques You'll Master

        - **Local Explainability:** Understanding individual predictions using LIME and SHAP
        - **Global Explainability:** Grasping overall model behavior and feature importance
        - **Model-Agnostic Methods:** Techniques that work with any black-box model
        - **Regulatory Compliance:** Generating adverse action notices that meet legal requirements
        - **Visualization:** Presenting complex model insights through clear, intuitive graphics

        ## Getting Started

        Use the navigation menu in the sidebar to begin your journey. Start with **Data Inspection** to familiarize yourself with the loan portfolio, then progress through each step sequentially.

        By the end of this workflow, you'll have transformed opaque AI predictions into transparent, defensible, and compliant explanations—building trust in AI systems within the highly regulated financial industry.

        ---

        💡 **Pro Tip:** Work through the pages in order. Each step builds on the previous one, creating a comprehensive picture of your model's decision-making process.
        """
    )
