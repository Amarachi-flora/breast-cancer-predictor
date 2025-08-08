<p align = "right">
  <img src="breast_cancer_logo.png" width="180" alt = "Breast Cancer Awareness">
</p>

# 🧬 Breast Cancer Diagnosis Predictor

An interactive web application built with **Python**, **Streamlit**, and **machine learning** to predict whether a breast tumor is **benign (non-cancerous)** or **malignant (cancerous)** based on diagnostic features. The model is trained on the well-known **Wisconsin Breast Cancer Dataset** and provides easy-to-understand visual explanations to support awareness and early detection.


##  About the App

- ** Goal**: Support early breast cancer diagnosis using data-driven machine learning predictions.
- ** Dataset**: [Wisconsin Breast Cancer Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- ** Tools Used**: Python, Streamlit, Plotly, Scikit-learn
- **🚫 Disclaimer**: This tool is for educational purposes **only** and not a substitute for medical advice or diagnosis.


##  How It Works

1. **Input Tumor Measurements**
   - You’ll input 30 tumor features (e.g., radius, texture, area).
   - Or use the example dataset for quick testing.

2. **Prediction Output**
   - The app returns:
     - Whether the tumor is **benign** (🟢) or **malignant** (🔴)
     - A **confidence score** (e.g., 96%)

3. **Visual Summary**
   - A **bar chart** displays the top 10 most influential tumor features for that prediction.

4. **Insights & Recommendations**
   - The app highlights features that influenced the prediction.
   - Suggestions are given to guide further inquiry with a medical professional.


##  Model Details

- **Model Type**: Random Forest Classifier
- **Accuracy**: 97%+
- **Other Models Evaluated**: SVM, Logistic Regression, Decision Tree
- **Hyperparameter Tuning**: GridSearchCV
- **Evaluation Metrics**: Accuracy, F1 Score, Confusion Matrix


##  Features

- ✔️ Manual or example-based input
- ✔️ Predict tumor diagnosis (Benign / Malignant)
- ✔️ Visual explanation of top features
- ✔️ Confidence score with each result
- ✔️ Sidebar FAQs and health disclaimers
- ✔️ Educational and user-friendly layout


##  Run the App Locally

1. **Clone the Repository**

```bash
git clone https://github.com/Amarachi-flora/breast-cancer-predictor.git
cd breast-cancer-predictor


2. Install Dependencies
pip install -r requirements.txt

3. Launch the App
streamlit run app.py

# Useful Info
Feature	        | Description
# ML Model	    |Random Forest Classifier (Best Accuracy)
# Frameworks	|Streamlit, Plotly, Scikit-learn
# Dataset	    |Wisconsin Breast Cancer (30 diagnostic features)
# Author	    |Amarachi Florence Onyedinma-Nwamaghiro

🚨 Disclaimer

- This application is intended only for educational and awareness purposes.
- It should not be used to make real medical decisions.
- Always consult a qualified healthcare provider.

🌐 Live Demo
👉 Click here to view the app live:
   (Streamlit link will be added after deployment)

# Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- GridSearchCV

# How You Can Contribute
⭐ Star this repo to support the project
🍴 Fork it to experiment with your own changes
🛠 Submit issues or pull requests for improvements

# Contact and Credits
- Thank you for exploring this project
- If you have questions, feedback, or want to collaborate:

🔗 Connect with me on LinkedIn

# License
This project is licensed under the MIT License.
© 2025 Amarachi Florence Onyedinma-Nwamaghiro. Feel free to use or modify with attribution.


