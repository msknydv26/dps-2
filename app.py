import streamlit as st
import pandas as pd
import pickle
import requests
import io

# Title
st.set_page_config(page_title="Price Predictor", layout="centered")
st.title("💸 Dynamic Price Predictor")

# Load model from Google Drive
@st.cache_resource
def load_model():
    try:
        file_id = "1G3RA7pDFouY8Ob7hpFmEH3dZ8AQDtP0u"
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        return pickle.load(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Input fields
inputs = {
    "day_of_week": st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]),
    "season": st.selectbox("Season", ["Winter", "Spring", "Summer", "Fall"]),
    "location": st.selectbox("Location", ["Boston", "Chicago", "Miami", "Seattle", "Los Angeles", "New York", "San Francisco", "Austin"]),
    "listing_type": st.selectbox("Listing Type", ["Airbnb", "Hostel", "Hotel"]),
    "event": st.selectbox("Event Nearby", ["Yes", "No"]),
    "base_price": st.slider("Base Price", 50, 500, 100),
    "demand_index": st.slider("Demand Index", 0.0, 1.0, 0.5),
    "competitor_avg_price": st.slider("Competitor Avg Price", 40, 600, 100),
    "occupancy_rate": st.slider("Occupancy Rate (%)", 30, 100, 60),
    "customer_rating": st.slider("Customer Rating", 3.0, 5.0, 4.5),
    "lead_time": st.slider("Lead Time (days)", 0, 59, 10),
    "weather_score": st.slider("Weather Score", 0.0, 1.0, 0.5),
    "discount_offered": st.slider("Discount (%)", 0.0, 20.0, 5.0),
}

if st.button("Predict Price"):
    df = pd.DataFrame([inputs])
    try:
        prediction = model.predict(df)[0]
        st.success(f"Predicted Price: ₹{round(prediction, 2)}")
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
