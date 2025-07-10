import streamlit as st
import pandas as pd
import altair as alt
import gzip
import pickle

st.set_page_config(page_title="Dynamic Pricing Dashboard + Prediction", layout="wide")

# Load compressed model
@st.cache_resource
def load_model():
    with gzip.open("dynamic_pricing_model_compressed.pkl.gz", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_dynamic_pricing_data_10000.csv")

df = load_data()

st.title("📊 Dynamic Pricing Dashboard + Prediction")

# Sidebar filters
st.sidebar.header("Filters")
location = st.sidebar.multiselect("Location", options=df["location"].unique(), default=df["location"].unique())
listing_type = st.sidebar.multiselect("Listing Type", options=df["listing_type"].unique(), default=df["listing_type"].unique())
season = st.sidebar.multiselect("Season", options=df["season"].unique(), default=df["season"].unique())
event = st.sidebar.multiselect("Event", options=df["event"].unique(), default=df["event"].unique())

filtered_df = df[
    (df["location"].isin(location)) &
    (df["listing_type"].isin(listing_type)) &
    (df["season"].isin(season)) &
    (df["event"].isin(event))
]

# KPIs
avg_price = filtered_df["final_price"].mean()
avg_occ = filtered_df["occupancy_rate"].mean()
avg_discount = filtered_df["discount_offered"].mean()
revenue_estimate = (filtered_df["final_price"] * (filtered_df["occupancy_rate"] / 100)).sum()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average Final Price", f"${avg_price:.2f}")
col2.metric("Average Occupancy Rate", f"{avg_occ:.2f}%")
col3.metric("Average Discount", f"{avg_discount:.2f}%")
col4.metric("Revenue Estimate", f"${revenue_estimate:,.0f}")

st.markdown("---")

# Visualizations
st.subheader("Visualizations")

chart1 = (
    alt.Chart(filtered_df)
    .mark_circle(size=60)
    .encode(
        x="demand_index",
        y="final_price",
        color="location",
        tooltip=["location", "final_price"],
    )
    .interactive()
)
st.altair_chart(chart1, use_container_width=True)

chart2 = (
    alt.Chart(filtered_df)
    .mark_bar()
    .encode(x="discount_offered", y="count()", color="listing_type")
    .properties(height=300)
)
st.altair_chart(chart2, use_container_width=True)

st.markdown("---")

# Prediction form
st.subheader("Predict Final Price")

with st.form("prediction_form"):
    base_price = st.number_input("Base Price ($)", min_value=10.0, max_value=1000.0, value=150.0)
    demand_index = st.slider("Demand Index", 0.0, 1.0, 0.5)
    competitor_avg_price = st.number_input("Competitor Avg Price ($)", min_value=10.0, max_value=1000.0, value=140.0)
    occupancy_rate = st.slider("Occupancy Rate (%)", 0.0, 100.0, 75.0)
    customer_rating = st.slider("Customer Rating", 1.0, 5.0, 4.2)
    lead_time = st.number_input("Lead Time (days)", min_value=0, max_value=180, value=14)
    weather_score = st.slider("Weather Score", 0.0, 1.0, 0.8)

    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame(
            [[
                base_price,
                demand_index,
                competitor_avg_price,
                occupancy_rate,
                customer_rating,
                lead_time,
                weather_score,
            ]],
            columns=[
                "base_price",
                "demand_index",
                "competitor_avg_price",
                "occupancy_rate",
                "customer_rating",
                "lead_time",
                "weather_score",
            ],
        )
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Final Price: ${prediction:.2f}")

st.markdown("---")

# Show filtered dataset
st.subheader("Filtered Dataset")
st.dataframe(filtered_df, use_container_width=True)

# Download filtered data
csv = filtered_df.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered Data as CSV", csv, "filtered_dynamic_pricing.csv", "text/csv")
