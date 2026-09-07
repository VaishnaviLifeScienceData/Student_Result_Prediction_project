import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="EduPredict | Academic Classifier",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .hero-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
    }
    
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: #c7d2fe;
        max-width: 600px;
        margin: 0 auto;
    }

    .card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .stat-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .pass-card {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.9), rgba(6, 95, 70, 0.9));
        border: 1px solid #059669;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #ecfdf5;
        animation: fadeIn 0.5s ease-in;
    }

    .fail-card {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.9), rgba(153, 27, 27, 0.9));
        border: 1px solid #dc2626;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: #fef2f2;
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model
@st.cache_resource
def load_model(path: str = "model.pkl"):
    if not os.path.exists(path):
        st.error(f"Error: `{path}` was not found in the root directory.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()

# Hero Header
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🎓 Academic Success Evaluator</div>
        <div class="hero-subtitle">
            Enter individual subject marks to analyze performance trends and predict student qualification status instantly.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Main Grid Layout
col_left, col_right = st.columns([7, 5], gap="large")

with col_left:
    st.markdown("### 📝 Enter Subject Scores")
    st.caption("Adjust the sliders corresponding to marks obtained (0 – 100)")
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        hindi = st.slider("Hindi", 0, 100, 65)
        science = st.slider("Science", 0, 100, 70)
        history = st.slider("History", 0, 100, 60)
        
    with sub_col2:
        english = st.slider("English", 0, 100, 75)
        maths = st.slider("Mathematics", 0, 100, 80)
        geography = st.slider("Geography", 0, 100, 65)

    total_score = hindi + english + science + maths + history + geography
    percentage = (total_score / 600) * 100

    st.write("")
    predict_clicked = st.button("Evaluate Result 🚀", type="primary", use_container_width=True)

with col_right:
    st.markdown("### 📊 Scorecard Summary")
    
    # Live summary metrics
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Score", f"{total_score} / 600")
    with metric_col2:
        st.metric("Percentage", f"{percentage:.1f}%")

    st.progress(min(max(total_score / 600.0, 0.0), 1.0))
    st.write("")

    # Align exactly with model training column names
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

    if predict_clicked:
        with st.spinner("Classifying results..."):
            time.sleep(0.5)
            raw_pred = model.predict(input_data)[0]

        # 1 indicates Pass, 0 indicates Fail
        if raw_pred == 1:
            st.balloons()
            st.markdown(
                """
                <div class="pass-card">
                    <h1 style="margin: 0; font-size: 2.5rem;">🎉 PASS</h1>
                    <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.9;">
                        The student meets all academic benchmark standards.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.snow()
            st.markdown(
                """
                <div class="fail-card">
                    <h1 style="margin: 0; font-size: 2.5rem;">⚠️ FAIL</h1>
                    <p style="margin-top: 0.5rem; font-size: 1.05rem; opacity: 0.9;">
                        The student falls below the passing qualification threshold.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Show confidence breakdown if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_data)[0]
            st.write("")
            st.write("**Model Confidence:**")
            
            # Map probabilities: 0 -> Fail, 1 -> Pass
            fail_prob = probs[0] if len(probs) > 0 else 0.0
            pass_prob = probs[1] if len(probs) > 1 else 0.0
            
            prob_df = pd.DataFrame(
                {"Probability": [fail_prob, pass_prob]},
                index=["Fail", "Pass"]
            )
            st.bar_chart(prob_df, y="Probability")
    else:
        st.info("👈 Set the subject marks and click **Evaluate Result** to trigger evaluation.")
