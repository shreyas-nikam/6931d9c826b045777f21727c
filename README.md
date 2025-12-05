# QuLab: Explainable Credit Scoring for Regulatory Compliance

![Streamlit App Logo](https://www.quantuniversity.com/assets/img/logo5.jpg) <!-- This image is referenced in app.py -->

## Project Title and Description

**Project Title:** QuLab: Explainable Credit Scoring for Regulatory Compliance

**Description:**
This Streamlit application, "QuLab: Explainable Credit Scoring," empowers Quantitative Analysts to navigate the complex landscape of regulatory scrutiny in AI-driven credit decisions. The project simulates a real-world scenario where a financial institution must provide transparent and defensible explanations for its automated loan approval or rejection processes.

Leveraging state-of-the-art Explainable AI (XAI) techniques like **LIME (Local Interpretable Model-agnostic Explanations)** and **SHAP (SHapley Additive exPlanations)**, the application provides an end-to-end workflow: from data inspection and model training to identifying critical cases, generating both local and global explanations, and finally synthesizing these insights into human-readable justifications and adverse action notices.

The goal is to demonstrate how XAI can build trust, ensure fairness, and meet compliance requirements in highly regulated sectors by transforming opaque "black-box" model outputs into clear, actionable, and auditable insights.

## Features

This application offers a comprehensive suite of features designed to provide a holistic view of credit model interpretability:

*   **Data Simulation & Inspection:** Generate a synthetic loan application dataset and perform initial data review, including raw data preview, descriptive statistics, and target variable distribution.
*   **Credit Model Training:** Train a `RandomForestClassifier` on the simulated data to serve as the black-box credit scoring model. Provides model accuracy on a test set.
*   **Critical Case Identification:** Identify and select specific rejected loan applications based on a configurable probability threshold, pinpointing instances that require detailed explanation.
*   **LIME Local Explanations:** Generate feature importance scores for individual loan applications using LIME, detailing local reasons behind a specific decision.
*   **LIME Visualization:** Visualize LIME explanations through intuitive bar charts, showing feature contributions to a particular prediction.
*   **SHAP Local Explanations:** Apply SHAP analysis to provide theoretically sound and consistent local explanations for individual predictions.
*   **SHAP Visualization:** Illustrate SHAP explanations with interactive Force Plots (via Waterfall plots) and Dependence Plots, revealing feature impacts and interactions.
*   **Explanation Synthesis:** Combine LIME and SHAP insights into a human-readable explanation, allowing for custom analyst comments and generating formal Adverse Action Notices for rejected applicants.
*   **Global Model Insights:** Utilize SHAP Summary Plots to understand overall feature importance and how features collectively influence the model's predictions across the entire dataset.
*   **Interactive Streamlit Interface:** A user-friendly, step-by-step navigation guides the user through the XAI workflow.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/quolab-explainable-credit-scoring.git
    cd quolab-explainable-credit-scoring
    ```
    *(Note: Replace `your-username/quolab-explainable-credit-scoring` with the actual repository URL if this project is hosted.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies:**
    The project relies on several Python libraries. Install them using the `requirements.txt` file:
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    (Content of `requirements.txt`):
    ```
    streamlit
    pandas
    numpy
    scikit-learn
    lime
    shap
    matplotlib
    seaborn
    ```

## Usage

Once the dependencies are installed, you can run the Streamlit application.

1.  **Navigate to the project root directory** (if you're not already there):
    ```bash
    cd quolab-explainable-credit-scoring
    ```

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  Your web browser should automatically open to the Streamlit application (usually `http://localhost:8501`).
    *   Follow the navigation in the sidebar, starting from "Home" and proceeding through the numbered steps ("Data Inspection", "Model Training", etc.) to explore the full XAI workflow.

## Project Structure

The project is organized into logical components to ensure modularity and ease of maintenance:

```
quolab-explainable-credit-scoring/
├── app.py                      # Main Streamlit application entry point and navigation
├── utils.py                    # Utility functions for data generation, model training, LIME, SHAP, and explanation synthesis
├── requirements.txt            # List of Python dependencies
└── application_pages/          # Directory containing individual Streamlit pages
    ├── page_1_home.py          # Welcome page and project overview
    ├── page_2_data_inspection.py # Data loading (synthetic) and initial data exploration
    ├── page_3_model_training.py  # Page for training the credit scoring model
    ├── page_4_case_identification.py # Page for identifying and selecting critical rejected cases
    ├── page_5_lime_explanation.py    # Page to generate LIME explanations for selected cases
    ├── page_6_lime_visualization.py  # Page to visualize LIME explanations
    ├── page_7_shap_explanation.py    # Page to generate SHAP explanations for selected cases
    ├── page_8_shap_visualization.py  # Page to visualize SHAP explanations (Force & Dependence plots)
    ├── page_9_explanation_synthesis.py # Page to synthesize human-readable explanations and generate adverse action notices
    └── page_10_global_insights.py    # Page to derive global model insights using SHAP summary plots
```

## Technology Stack

The application is built using the following core technologies and libraries:

*   **Python 3.8+**: The primary programming language.
*   **Streamlit**: For creating the interactive web application user interface.
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations, especially with synthetic data generation.
*   **Scikit-learn**: For machine learning model implementation (e.g., `RandomForestClassifier`) and model evaluation.
*   **LIME (Local Interpretable Model-agnostic Explanations)**: An XAI library for local interpretability.
*   **SHAP (SHapley Additive exPlanations)**: An XAI library for robust local and global interpretability.
*   **Matplotlib**: For static plotting and visualization (used by LIME and SHAP visualizations).
*   **Seaborn**: For enhancing data visualizations.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  **Fork** the repository.
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `bugfix/your-bug-fix`.
3.  **Make your changes** and ensure the code adheres to the existing style.
4.  **Commit your changes** with clear and concise messages: `git commit -m "feat: Add new feature X"` or `fix: Resolve bug Y`.
5.  **Push your branch** to your forked repository: `git push origin feature/your-feature-name`.
6.  **Open a Pull Request** against the `main` branch of this repository, describing your changes in detail.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file containing the MIT license text should be created in the root directory of your project.)*

## Contact

For questions, feedback, or collaborations, please reach out to:

*   **Project Maintainer:** Your Name / Quant University
*   **Email:** your.email@example.com / info@quantuniversity.com
*   **Website:** [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **GitHub Issues:** [https://github.com/your-username/quolab-explainable-credit-scoring/issues](https://github.com/your-username/quolab-explainable-credit-scoring/issues)

---
Developed for educational and demonstration purposes.
