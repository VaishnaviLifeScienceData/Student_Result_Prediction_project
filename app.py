import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Academic Performance Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1f2937, #111827);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    .result-box {
        padding: 24px;
        border-radius: 14px;
        background: linear-gradient(135deg, #064e3b, #065f46);
        color: #ecfdf5;
        border: 1px solid #10b981;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model loader with caching
@st.cache_resource
def load_model(path: str = "model.pkl"):
    if not os.path.exists(path):
        st.error(f"Error: `{path}` was not found in the root directory.")
        st.stop()
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Header Section
st.title("🎓 Student Outcome Classifier")
st.caption("Predict performance categories using your trained KNN model.")
st.write("---")

# Input Form
with st.container():
    st.subheader("Enter Subject Marks (0 - 100)")

    col1, col2 = st.columns(2)
    with col1:
        hindi = st.slider("Hindi", 0, 100, 65)
        science = st.slider("Science", 0, 100, 75)
        history = st.slider("History", 0, 100, 70)

    with col2:
        english = st.slider("English", 0, 100, 70)
        maths = st.slider("Maths", 0, 100, 80)
        geography = st.slider("Geography", 0, 100, 68)

    # Calculate total automatically
    total_score = hindi + english + science + maths + history + geography
    avg_score = total_score / 6.0

    st.markdown(
        f"""
        <div class="metric-card">
            <div style="display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <span style="color: #9ca3af; font-size: 0.9rem;">Total Marks</span>
                    <h3 style="color: #60a5fa; margin: 4px 0;">{total_score} / 600</h3>
                </div>
                <div>
                    <span style="color: #9ca3af; font-size: 0.9rem;">Average Score</span>
                    <h3 style="color: #34d399; margin: 4px 0;">{avg_score:.2f}%</h3>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Align input features with the model's exact metadata
    # Note: 'Geograpgy' keeps the key expected by the serialized classifier
    input_data = pd.DataFrame(
        [
            {
                "Hindi": hindi,
                "English": english,
                "Science": science,
                "Maths": maths,
                "History": history,
                "Geograpgy": geography,
                "Total": total_score,
            }
        ]
    )

    # Action Button
    predict_btn = st.button("Run Prediction 🚀", use_container_width=True, type="primary")

# Prediction & Animation
if predict_btn:
    with st.spinner("Analyzing performance vectors..."):
        time.sleep(0.6)  # Short delay for visual effect
        prediction = model.predict(input_data)[0]

    # Visual effects
    st.balloons()

    st.markdown(
        f"""
        <div class="result-box">
            🎯 Predicted Outcome: <span style="color: #a7f3d0;">Class {prediction}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Probability estimation if supported by the fitted instance
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_data)[0]
        st.write("")
        st.write("**Confidence Breakdown:**")
        prob_df = pd.DataFrame({"Class": model.classes_, "Probability": probs})
        st.bar_chart(prob_df.set_index("Class"))
