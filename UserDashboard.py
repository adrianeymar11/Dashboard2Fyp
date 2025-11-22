# ============================================================
# USER-FOCUSED MENTAL HEALTH RISK DASHBOARD (ALIGNED TO BACKEND)
# Author: Adrian Anthony A/L R. Vikneswaran (UTP)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ----------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------
st.set_page_config(
    page_title="Digital Wellbeing Predictor",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------------------
# STYLE SETTINGS
# ----------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #DFF7F9 0%, #F8FBFC 100%);
    font-family: 'Helvetica Neue', sans-serif;
}
h1, h2, h3 {
    color: #007C91;
    font-weight: 700;
}
.main-card {
    background-color: white;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    padding: 25px;
    margin-top: 20px;
}
.stButton>button {
    border-radius: 10px;
    font-weight: 600;
    width: 100%;
    height: 50px;
}
div.stButton > button:first-child {
    background-color: #007C91;
    color: white;
}
div.stButton > button:first-child:hover {
    background-color: #005C68;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# EXCEL LOGGING HELPER
# ----------------------------------------
EXCEL_PATH = "dashboard_predictions.xlsx"

def save_submission_to_excel(record: dict, excel_path: str = EXCEL_PATH):
    df_new = pd.DataFrame([record])
    if os.path.exists(excel_path):
        try:
            df_existing = pd.read_excel(excel_path)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        except Exception:
            df_out = df_new
    else:
        df_out = df_new
    df_out.to_excel(excel_path, index=False, engine="openpyxl")

# ----------------------------------------
# LOAD MODEL
# ----------------------------------------
model_path = "RandomForest_best_pipeline1.pkl"
if not os.path.exists(model_path):
    st.error("⚠️ Trained model not found.")
    st.stop()

model = joblib.load(model_path)

# Extract backend feature names
preproc = model.named_steps["preproc"]
numeric_features = preproc.transformers_[0][2]
categorical_features = preproc.transformers_[1][2]
ALL_FEATURES = list(numeric_features) + list(categorical_features)

# ----------------------------------------
# LOGIN SYSTEM
# ----------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Digital Wellbeing Login</h1>", unsafe_allow_html=True)

users = {"adrian": "1234", "guest": "0000", "jahnani": "Jahnani25"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    username = st.text_input("👤 Username")
    password = st.text_input("🔒 Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()

# ----------------------------------------
# MAIN PREDICTION PANEL (AFTER LOGIN)
# ----------------------------------------
if st.session_state.logged_in:

    # ✅ SIDEBAR LOGOUT BUTTON ADDED
    st.sidebar.success(f"👋 Logged in as: {st.session_state.username}")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.experimental_rerun()

    # MAIN CARD
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.header("📊 Digital Wellbeing Assessment")

    age = st.number_input("🎂 Age", 1, 100, 21)
    tech = st.number_input("🖥️ Technology Usage (hours/day)", 0.0, 24.0, 3.0)
    social = st.number_input("📱 Social Media Usage (hours/day)", 0.0, 24.0, 2.0)
    gaming = st.number_input("🎮 Gaming Usage (hours/day)", 0.0, 24.0, 1.0)
    screen = st.number_input("📺 Daily Screen Time (hours/day)", 0.0, 24.0, 6.0)
    stress = st.number_input("😰 Stress Level (0–10)", 0.0, 10.0, 5.0)
    sleep = st.number_input("💤 Sleep Duration (hours)", 0.0, 24.0, 7.0)
    activity = st.number_input("🏃 Physical Activity (hours/day)", 0.0, 24.0, 1.0)

    user_input = pd.DataFrame([{
        "Age": age,
        "Technology_Usage_Hours": tech,
        "Social_Media_Usage_Hours": social,
        "Gaming_Usage_Hours": gaming,
        "Daily_Screen_Time_Hours": screen,
        "Stress_Level": stress,
        "Sleep_Hours": sleep,
        "Physical_Activity_Hours": activity,
    }])

    # Align columns
    def align(df, cols):
        new = df.copy()
        for c in cols:
            if c not in new:
                new[c] = 0
        return new[cols]

    aligned = align(user_input, ALL_FEATURES)

    # PREDICTION
    if st.button("🔍 Predict Risk Level"):

        pred = model.predict(aligned)[0]
        probs = model.predict_proba(aligned)[0]
        confidence = max(probs) * 100

        color_map = {"Low": "#00A676", "Medium": "#FACC15", "High": "#E53935"}
        bg = color_map.get(pred, "#0097A7")

        st.markdown(f"""
        <div style='background:{bg};border-radius:12px;padding:20px;text-align:center;'>
            <h3 style='color:white;'>Predicted Risk: <b>{pred}</b></h3>
            <p style='color:white;'>Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.subheader("📝 Personalized Recommendations")

        if pred.lower() == "low":
            st.info("""
            ✔ You are in **Low Risk**.
            - Maintain healthy sleep routines (7–9 hours).
            - Balance digital and physical activities.
            - Keep positive interactions.
            """)

        elif pred.lower() == "medium":
            st.warning("""
            ⚠ Moderate Risk.
            - Reduce screen time at night.
            - Exercise 30 minutes daily.
            - Take hourly digital breaks.
            """)

        elif pred.lower() == "high":
            st.error("""
            🔥 High Risk.
            - Seek support if needed.
            - Limit gaming & social media.
            - Practice mindfulness.
            - Fix sleep schedule.
            """)

        # LOGGING
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": st.session_state.username,
            **user_input.iloc[0].to_dict(),
            "prediction": pred,
            "confidence_pct": round(confidence, 2)
        }

        save_submission_to_excel(record)

    st.markdown("</div>", unsafe_allow_html=True)
