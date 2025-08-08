<p align="right">
  <img src="breast_cancer_logo.png" width="180" alt="Breast Cancer Awareness">
</p>

# 🧬 Breast Cancer Diagnosis Predictor

An interactive web application built with **Python**, **Streamlit**, and **machine learning** to predict whether a breast tumor is **benign (non-cancerous)** or **malignant (cancerous)** based on diagnostic features. The model is trained on the well-known [Wisconsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and provides easy-to-understand visual explanations to support awareness and early detection.

---

##  About the App

-  **Goal**: Support early breast cancer diagnosis using data-driven ML predictions.

-  **Dataset**: [Wisconsin Breast Cancer Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

-  **Tools Used**: Python, Streamlit, Plotly, Scikit-learn

- 🚫 **Disclaimer**: This tool is for educational purposes **only** and not a substitute for medical advice.

---

## ⚙️ How It Works

1. **Input Tumor Measurements**  
   - Enter 30 diagnostic features or use sample input.

2. **Prediction Output**  
   - App returns:
     - 🟢 **Benign** or 🔴 **Malignant**
     - ✅ Confidence Score (e.g., 96%)

3. **Visual Summary**  
   - Bar chart of top 10 most important features.

4. **Insights**  
   - Key features influencing prediction are highlighted.
   - Suggestions to seek medical attention are included.

---

##  Model Details

- **Model**: Random Forest Classifier  
- **Accuracy**: 97%+  
- **Other Models Tried**: SVM, Logistic Regression, Decision Tree  
- **Tuning**: GridSearchCV  
- **Metrics**: Accuracy, F1 Score, Confusion Matrix

---

##  Features

- ✔️ Manual or sample-based input
- ✔️ Predict tumor diagnosis (Benign / Malignant)
- ✔️ Feature importance visualization
- ✔️ Confidence score with results
- ✔️ Sidebar FAQs and disclaimers
- ✔️ Clean, educational UI

---

##  Run the App Locally
  
```bash
# 1. Clone the repository
git clone https://github.com/Amarachi-flora/breast-cancer-predictor.git
cd breast-cancer-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py


- Project Overview

| Feature    | Description                              |
| ---------- | ---------------------------------------- |
| ML Model   | Random Forest Classifier (best accuracy) |
| Frameworks | Streamlit, Plotly, Scikit-learn          |
| Dataset    | Wisconsin Breast Cancer (30 features)    |
| Author     | Amarachi Florence Onyedinma-Nwamaghiro   |
  
  
# Technologies Used

    - Python

    - Pandas, NumPy

    - Scikit-learn

    - Streamlit

    - Plotly

    - GridSearchCV



 🤝 How You Can Contribute

    ⭐ Star this repo to show support

    🍴 Fork it and try your own improvements

    🛠️ Submit issues or pull requests


## 🌐 Live Demo

👉 [Click here to try the app](https://breast-cancer-predictor-7.streamlit.app/)

## 🙋🏽‍♀️ Contact

**Amarachi Florence Onyedinma-Nwamaghiro**  
🔗 [Connect on LinkedIn](https://www.linkedin.com/in/amarachi-florence/)


    - License

This project is licensed under the MIT License.
© 2025 Amarachi Florence Onyedinma-Nwamaghiro — free to use or modify with attribution.