
import streamlit as st


def main():
    st.markdown(
        """
        # Interactive Quiz: Test Your XAI Knowledge

        Congratulations on completing the Explainable Credit Scoring workflow! 
        Now it's time to test your understanding of the concepts you've learned.
        This quiz covers key concepts from LIME, SHAP, and explainable AI principles.
        """
    )

    # Initialize session state for quiz
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}

    # Define quiz questions
    questions = [
        {
            "id": "q1",
            "question": "What does LIME stand for?",
            "options": [
                "Linear Interpretation Model Explanation",
                "Local Interpretable Model-agnostic Explanations",
                "Logical Inference for Machine Explanations",
                "Learning Interpretable Models Easily"
            ],
            "correct": 1,
            "explanation": "LIME stands for Local Interpretable Model-agnostic Explanations. It works by approximating the complex model locally around a specific prediction with an interpretable model (like linear regression)."
        },
        {
            "id": "q2",
            "question": "What is the key theoretical foundation of SHAP values?",
            "options": [
                "Linear Algebra",
                "Bayesian Statistics",
                "Game Theory (Shapley Values)",
                "Neural Networks"
            ],
            "correct": 2,
            "explanation": "SHAP is based on Shapley values from cooperative game theory. This ensures a fair distribution of the prediction 'credit' among features, guaranteeing consistency and local accuracy."
        },
        {
            "id": "q3",
            "question": "In a SHAP summary plot, what does the color of each dot represent?",
            "options": [
                "The SHAP value magnitude",
                "The feature value (high or low)",
                "The prediction probability",
                "The feature importance rank"
            ],
            "correct": 1,
            "explanation": "In a SHAP summary plot, the color represents the feature value. Red indicates high feature values and blue indicates low feature values. This helps you understand how feature values influence the model's predictions."
        },
        {
            "id": "q4",
            "question": "Why is explainable AI particularly important in credit scoring?",
            "options": [
                "It makes models run faster",
                "It improves model accuracy",
                "It meets regulatory compliance and ensures fairness",
                "It reduces the amount of training data needed"
            ],
            "correct": 2,
            "explanation": "Explainable AI is crucial in credit scoring for regulatory compliance (e.g., adverse action notices), ensuring fairness and preventing discrimination, and building trust with customers by providing transparent explanations for decisions."
        },
        {
            "id": "q5",
            "question": "What is the main difference between LIME and SHAP explanations?",
            "options": [
                "LIME is faster but less theoretically grounded than SHAP",
                "SHAP only works with tree-based models",
                "LIME provides global explanations while SHAP provides local explanations",
                "There is no difference, they are the same method"
            ],
            "correct": 0,
            "explanation": "LIME is generally faster and model-agnostic but uses sampling/approximation, making it less theoretically rigorous. SHAP is based on solid game-theoretic foundations (Shapley values), ensuring consistency and accuracy, but can be more computationally expensive."
        }
    ]

    st.markdown("---")
    st.subheader("Answer the following questions:")

    # Display questions
    for i, q in enumerate(questions, 1):
        st.markdown(f"### Question {i}")
        st.markdown(f"**{q['question']}**")
        
        # Radio buttons for options
        answer = st.radio(
            f"Select your answer:",
            options=q['options'],
            key=f"question_{q['id']}",
            index=None
        )
        
        # Store answer
        if answer:
            st.session_state["quiz_answers"][q['id']] = q['options'].index(answer)
        
        st.markdown("---")

    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Submit Quiz", type="primary", use_container_width=True):
            # Check if all questions are answered
            if len(st.session_state["quiz_answers"]) < len(questions):
                st.warning("Please answer all questions before submitting!")
            else:
                st.session_state["quiz_submitted"] = True
                st.rerun()

    # Show results if submitted
    if st.session_state["quiz_submitted"]:
        st.markdown("---")
        st.markdown("## 📊 Quiz Results")
        
        # Calculate score
        correct_count = 0
        for q in questions:
            user_answer = st.session_state["quiz_answers"].get(q['id'])
            if user_answer == q['correct']:
                correct_count += 1
        
        score_percentage = (correct_count / len(questions)) * 100
        
        # Display score with color coding
        if score_percentage >= 80:
            st.success(f"### 🎉 Excellent! You scored {correct_count}/{len(questions)} ({score_percentage:.0f}%)")
            st.balloons()
        elif score_percentage >= 60:
            st.info(f"### 👍 Good job! You scored {correct_count}/{len(questions)} ({score_percentage:.0f}%)")
        else:
            st.warning(f"### 📚 Keep learning! You scored {correct_count}/{len(questions)} ({score_percentage:.0f}%)")
        
        st.markdown("---")
        st.markdown("### Detailed Results:")
        
        # Show detailed results for each question
        for i, q in enumerate(questions, 1):
            user_answer = st.session_state["quiz_answers"].get(q['id'])
            is_correct = user_answer == q['correct']
            
            with st.expander(f"Question {i}: {'✅ Correct' if is_correct else '❌ Incorrect'}"):
                st.markdown(f"**{q['question']}**")
                st.markdown(f"**Your answer:** {q['options'][user_answer]}")
                st.markdown(f"**Correct answer:** {q['options'][q['correct']]}")
                
                if not is_correct:
                    st.markdown("**Explanation:**")
                    st.info(q['explanation'])
                else:
                    st.markdown("**Explanation:**")
                    st.success(q['explanation'])
        
        st.markdown("---")
        
        # Reset button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("Retake Quiz", use_container_width=True):
                st.session_state["quiz_submitted"] = False
                st.session_state["quiz_answers"] = {}
                st.rerun()

        st.markdown("---")
        st.markdown("""
        ### 🎓 Congratulations!
        
        You've completed the QuLab: Explainable Credit Scoring application. You now have hands-on experience with:
        - Understanding credit scoring data
        - Training predictive models
        - Applying LIME and SHAP for local explanations
        - Creating regulatory-compliant adverse action notices
        - Analyzing global model behavior
        
        These skills are essential for building transparent, fair, and compliant AI systems in financial services.
        """)

