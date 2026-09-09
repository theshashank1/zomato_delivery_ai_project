import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = "models"
ETA_MODEL = os.path.join(MODEL_DIR, "eta_linear_regression.pkl")
LATE_MODEL = os.path.join(MODEL_DIR, "late_logistic_regression.pkl")
INFO_MODEL = os.path.join(MODEL_DIR, "model_info.pkl")

st.set_page_config(page_title="Zomato Delivery AI", page_icon="🍔", layout="centered")


# Load pipelines that already contain preprocessing + model.
@st.cache_resource
def load_models():
    with open(ETA_MODEL, "rb") as f:
        eta = pickle.load(f)
    with open(LATE_MODEL, "rb") as f:
        late = pickle.load(f)
    with open(INFO_MODEL, "rb") as f:
        info = pickle.load(f)
    return eta, late, info

st.title("🍔 Zomato Delivery AI")
st.caption("One order → two AI answers: ETA in minutes + Late Delivery risk")

if not all(os.path.exists(p) for p in [ETA_MODEL, LATE_MODEL, INFO_MODEL]):
    st.error("Model files are missing. Run `python train_models.py` first.")
    st.stop()

eta_model, late_model, info = load_models()

st.subheader("Order details")
col1, col2 = st.columns(2)
with col1:
    distance = st.number_input("Distance (km)", min_value=0.5, max_value=30.0, value=5.0, step=0.5)
    age = st.number_input("Delivery person age", min_value=18, max_value=60, value=25)
    rating = st.slider("Delivery rating", 1.0, 5.0, 4.5, 0.1)
    vehicle_condition = st.slider("Vehicle condition (0–2)", 0, 2, 1)
with col2:
    multiple_deliveries = st.selectbox("Additional deliveries", [0, 1, 2, 3], index=0)
    traffic = st.selectbox("Traffic", ["Low", "Medium", "High", "Jam"], index=1)
    weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Fog", "Stormy", "Windy", "Sandstorms"], index=0)
    festival = st.selectbox("Festival", ["No", "Yes"], index=0)

row = {
    "distance_km": distance,
    "Delivery_person_Age": age,
    "Delivery_person_Ratings": rating,
    "Vehicle_condition": vehicle_condition,
    "multiple_deliveries": multiple_deliveries,
    "Road_traffic_density": traffic,
    "Weather_conditions": weather,
    "Festival": festival,
}
X = pd.DataFrame([row])

if st.button("Predict Delivery", type="primary", use_container_width=True):
    feature_cols = info.get("feature_cols", info.get("features", []))
    
    # If model expects dummy columns, encode them
    if feature_cols and feature_cols[0] != "distance_km" or len(feature_cols) > 8:
        X_encoded = pd.get_dummies(X)
        X_encoded = X_encoded.reindex(columns=feature_cols, fill_value=0)
    else:
        X_encoded = X

    eta = float(eta_model.predict(X_encoded)[0])
    late_prob = float(late_model.predict_proba(X_encoded)[0, 1])
    late_class = int(late_model.predict(X_encoded)[0])

    eta = max(1, eta)
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predicted ETA", f"{eta:.0f} min")
    with c2:
        st.metric("Late probability", f"{late_prob * 100:.0f}%")

    if late_class == 1:
        st.warning(f"⚠️ High late-delivery risk — likely more than {info['late_threshold']} minutes.")
    else:
        st.success(f"✅ Likely on time — at or below {info['late_threshold']} minutes.")

    with st.expander("See model inputs"):
        st.dataframe(X, use_container_width=True, hide_index=True)

st.divider()
st.caption("Teaching demo: Linear Regression answers HOW LONG; Logistic Regression answers HOW LIKELY.")
