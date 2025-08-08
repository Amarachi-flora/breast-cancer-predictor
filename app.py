#  ZION TECH HUB - Breast Cancer Detection Project 

#  File: app.py (Streamlit App)
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px

#  Load model AND feature names
model = joblib.load("breast_cancer_ml_project/models/best_rf_model.pkl")
feature_names = pd.read_csv("breast_cancer.csv").drop(columns=['id', 'diagnosis'], errors='ignore').columns.tolist()

# Streamlit page config
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# Styling
st.markdown("""
<style>
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    color: #3f51b5;
}
.stButton>button {
    background-color: #6200ea;
    color: white;
}
.stNumberInput>div>input {
    background-color: #f3e5f5;
}
</style>
""", unsafe_allow_html=True)

# Sidebar with Info
with st.sidebar:
    st.title("📘 Project Info")
    st.markdown("""
    **Breast Cancer Detection App**

    **Objective:** Help predict whether a tumor is benign or malignant using patient data.

    **Purpose:** Assist early diagnosis and awareness.

    *Note:* This is for educational purposes and not a medical diagnostic tool.

    **Built With:**
    - Random Forest Classifier
    - Streamlit
    - Python & Machine Learning 

    ❓ **Help / FAQ**

    **Q: What values should I enter?**
    - Use average values from tumor measurements or real test data if available.

    **Q: What does the prediction mean?**
    - 🟢 Benign: Not cancer.
    - 🔴 Malignant: Cancer detected. Please consult a medical professional.

    **Q: Can I trust this result?**
    - This model is well-trained but should not replace professional diagnosis.

    **Tip:** Click "Use Example Data" to auto-fill with sample measurements now.

    **Sample Data Source:** The example data used in this app is based on the **Wisconsin Breast Cancer Dataset** (UCI Machine Learning Repository).
    """)

# Header image and title
col1, col2 = st.columns([3, 1])
with col2:
    st.image("breast_cancer_awareness.png", use_container_width=True)

st.title(" 🧬 Breast Cancer Diagnosis Predictor")

st.markdown("## What is Breast Cancer?")
st.markdown("""
Breast cancer is a disease in which cells in the breast grow and divide uncontrollably, often forming a tumor. These tumors may be benign (non-cancerous) or malignant (cancerous). Early detection through screening and diagnostic tests improves the chances of successful treatment.

👉 Enter tumor features manually or click **Use Example Data** to see how the model works.
""" )

# Corresponding example values
example_values = [
    14.2, 20.1, 92.0, 660.0, 0.09,
    0.13, 0.10, 0.07, 0.18, 0.06,
    0.5, 1.2, 3.0, 40.0, 0.006,
    0.02, 0.02, 0.01, 0.02, 0.003,
    17.5, 27.5, 110.0, 950.0, 0.15,
    0.25, 0.20, 0.13, 0.28, 0.09
]

# Show example data
example_df = pd.DataFrame({"Feature": feature_names, "Example Value": example_values})
with st.expander(" Example Values for Testing"):
    st.markdown("The example data used in this app is based on the **Wisconsin Breast Cancer Dataset** (UCI Machine Learning Repository).")
    st.dataframe(example_df)

# Input section
st.header("  Enter Tumor Measurements")
user_input = []
columns = st.columns(3)

for i, name in enumerate(feature_names):
    col = columns[i % 3]
    val = col.number_input(name, min_value=0.0, value=example_values[i], key=name)
    user_input.append(val)

X_input = pd.DataFrame([user_input], columns=feature_names)

# Prediction
if st.button("🔍 Predict Diagnosis"):
    prediction = model.predict(X_input)[0]
    confidence = model.predict_proba(X_input)[0][prediction]
    result = "🟢 Benign (Not Cancer)" if prediction == 0 else "🔴 Malignant (Cancer Detected)"
    st.success(f"**Diagnosis:** {result}")
    st.info(f"**Confidence:** {confidence:.2%}")

    # Chart
    top10 = pd.DataFrame(user_input, index=feature_names, columns=["Value"]).sort_values("Value", ascending=False).head(10)
    fig = px.bar(top10, x="Value", y=top10.index, orientation="h", title="Top 10 Features")
    st.plotly_chart(fig, use_container_width=True)

    # Findings
    first = top10.index[0]
    second = top10.index[1]
    st.markdown(f"""
    ###  Findings
    - After analyzing the input features, the model identified **{first}** and **{second}** as having the most significant influence on the prediction.
    - These high-impact features might indicate critical tumor characteristics based on their values.

    ###  Recommendation
    - For healthcare professionals or researchers, extra attention should be given to **{first}** and **{second}** during diagnosis.
    - Combining these features with clinical observations could enhance diagnostic confidence and improve early intervention.

    ###  Conclusion
    - Visualizing the top influencing factors helps patients and healthcare providers understand the basis for the diagnosis.
    - It also supports data-driven decision-making, making it easier to explain and trust the prediction outcomes.
    - As technology advances, such AI-powered tools can assist in delivering quicker and more informed medical insights.
    """)
