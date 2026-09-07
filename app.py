import os
import pickle
import time
import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="EduPredict Pro | Student Performance AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Gradient Hero Header */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        padding: 2.2rem 2rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 12px 30px -10px rgba(49, 46, 129, 0.45);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.35rem;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 650px;
        margin: 0 auto;
    }

    /* Glassmorphism Cards for Subject Sliders */
    .subject-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid #1f2937;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.85rem;
        transition: all 0.2s ease-in-out;
    }
    .subject-card:hover {
        border-color: #6366f1;
        box-shadow: 0 4px 20px rgba(99, 102, 241, 0.12);
        transform: translateY(-2px);
    }
    .subject-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    .subject-name {
        font-weight: 700;
        font-size: 1.05rem;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .score-badge {
        background: #1e1b4b;
        color: #a5b4fc;
        padding: 0.2rem 0.65rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* Modern Slider Track Styling */
    div[data-baseweb="slider"] {
        margin-top: 0.2rem;
    }
    div[data-baseweb="slider"] > div > div {
        background-color: #6366f1 !important;
    }

    /* Result Outcome Cards */
    .status-pass {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.95), rgba(6, 95, 70, 0.95));
        border: 2px solid #10b981;
        border-radius: 18px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        color: #ecfdf5;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.25);
        animation: pulseEffect 0.6s ease-out;
    }
    .status-fail {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.95), rgba(153, 27, 27, 0.95));
        border: 2px solid #ef4444;
        border-radius: 18px;
        padding: 2.2rem 1.5rem;
        text-align: center;
        color: #fef2f2;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.25);
        animation: pulseEffect 0.6s ease-out;
    }
    @keyframes pulseEffect {
        0% { transform: scale(0.96); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Cached Model Loading
@st.cache_resource
def load_classifier(path: str = "model.pkl"):
    if not os.path.exists(path):
        st.error(f"Error: `{path}` was not found in the root directory.")
        st.stop()
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_classifier()

# Top Hero Header
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">🎓 Student Academic Evaluation Portal</div>
        <div class="hero-desc">
            Use the interactive mark controllers below to assess performance metrics and predict the final pass/fail qualification in real time.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Two-Column Dashboard Layout
col_left, col_right = st.columns([7, 5], gap="large")

# Left Column: Inputs
with col_left:
    st.markdown("### 📋 Subject Score Controls")
    st.caption("Slide to adjust marks obtained out of 100 for each examination.")

    def render_slider(icon, label, key, default):
        score = st.slider(
            f"{icon} {label}",
            min_value=0,
            max_value=100,
            value=default,
            key=key,
            label_visibility="visible",
        )
        return score

    sub_col1, sub_col2 = st.columns(2, gap="medium")
    
    with sub_col1:
        hindi = render_slider("📖", "Hindi", "hindi_val", 65)
        science = render_slider("🔬", "Science", "science_val", 72)
        history = render_slider("🏛️", "History", "history_val", 68)

    with sub_col2:
        english = render_slider("🗣️", "English", "english_val", 75)
        maths = render_slider("📐", "Maths", "maths_val", 80)
        geography = render_slider("🌍", "Geography", "geog_val", 70)

    # Calculate Totals
    total_score = hindi + english + science + maths + history + geography
    percentage = (total_score / 600.0) * 100.0

    st.write("")
    predict_btn = st.button("Evaluate Qualification 🚀", type="primary", use_container_width=True)

# Right Column: Live Summary & Predictions
with col_right:
    st.markdown("### 📊 Performance Analytics")

    # Metrics Overview
    stat1, stat2 = st.columns(2)
    with stat1:
        st.metric("Total Score", f"{total_score} / 600")
    with stat2:
        st.metric("Aggregate Percentage", f"{percentage:.2f}%")

    # Progress bar with dynamic coloring indication
    st.progress(min(max(total_score / 600.0, 0.0), 1.0))
    st.write("")

    # Align input keys with exact pickle features: 'Geograpgy' keeps internal model spelling[cite: 1]
    features = pd.DataFrame(
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

    if predict_btn:
        with st.spinner("Analyzing performance with KNN..."):
            time.sleep(0.5)
            outcome = model.predict(features)[0]

        if outcome == 1:
            st.balloons()
            st.markdown(
                f"""
                <div class="status-pass">
                    <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800;">🎉 PASS</h1>
                    <p style="margin: 0.6rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        Eligible for promotion. Scored <strong>{percentage:.1f}%</strong> overall.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.snow()
            st.markdown(
                f"""
                <div class="status-fail">
                    <h1 style="margin: 0; font-size: 2.8rem; font-weight: 800;">⚠️ FAIL</h1>
                    <p style="margin: 0.6rem 0 0 0; font-size: 1.1rem; opacity: 0.95;">
                        Does not meet qualification criteria. Scored <strong>{percentage:.1f}%</strong> overall.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Model Probability Breakdown
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
            st.write("")
            st.write("**Model Probability Spread:**")
            
            prob_df = pd.DataFrame(
                {"Confidence": [probs[0], probs[1]]},
                index=["Fail", "Pass"]
            )
            st.bar_chart(prob_df, y="Confidence")
    else:
        st.info("💡 Adjust any slider on the left and click **Evaluate Qualification** to view pass/fail predictions.")
