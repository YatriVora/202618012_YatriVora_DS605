import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NYC Airbnb Price Predictor", page_icon="🏠")

# ---------- Cached setup: rebuild the neighbourhood encoding + load model ----------
@st.cache_resource
def load_artifacts():
    df = pd.read_csv('AB_NYC_2019.csv')
    df = df[df['price'] > 0].copy()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    df['price_capped'] = df['price'].clip(*df['price'].quantile([0.01, 0.99]))
    df['log_price'] = np.log1p(df['price_capped'])

    reference_date = df['last_review'].max()
    df['has_reviews'] = df['last_review'].notna().astype(int)
    df['days_since_last_review'] = (reference_date - df['last_review']).dt.days
    sentinel = df['days_since_last_review'].max() + 1
    df['days_since_last_review'] = df['days_since_last_review'].fillna(sentinel)

    # rebuild the same smoothed neighbourhood -> mean log-price mapping used in training
    global_mean_price = df['log_price'].mean()
    m = 20
    stats = df.groupby('neighbourhood')['log_price'].agg(['mean', 'count'])
    neighbourhood_map = ((stats['mean'] * stats['count'] + global_mean_price * m)
                          / (stats['count'] + m))

    pipeline_bundle = joblib.load('final_model_pipeline.pkl')
    return pipeline_bundle, neighbourhood_map, global_mean_price, sentinel

pipeline_bundle, neighbourhood_map, global_mean_price, sentinel = load_artifacts()
model = pipeline_bundle['model']
selected_features = pipeline_bundle['selected_features']

TIMES_SQUARE = (40.7580, -73.9855)
JFK_AIRPORT = (40.6413, -73.7781)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

# ---------- UI ----------
st.title("🏠 NYC Airbnb Nightly Price Predictor")
st.write("Enter listing details to estimate a nightly price.")

col1, col2 = st.columns(2)
with col1:
    room_type = st.selectbox("Room type", ["Entire home/apt", "Private room", "Shared room"])
    neighbourhood_group = st.selectbox("Borough", ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"])
    neighbourhood = st.selectbox("Neighbourhood", sorted(neighbourhood_map.index.tolist()))
    latitude = st.number_input("Latitude", value=40.7500, format="%.6f")
    longitude = st.number_input("Longitude", value=-73.9800, format="%.6f")
    minimum_nights = st.number_input("Minimum nights", min_value=1, max_value=365, value=3)

with col2:
    has_reviews = st.checkbox("Has this listing been reviewed before?", value=True)
    days_since_last_review = st.number_input(
        "Days since last review", min_value=0, value=30
    ) if has_reviews else sentinel
    number_of_reviews = st.number_input("Number of reviews", min_value=0, value=10)
    reviews_per_month = st.number_input("Reviews per month", min_value=0.0, value=1.0, step=0.1)
    calculated_host_listings_count = st.number_input("Host's total listings", min_value=1, value=1)
    availability_365 = st.slider("Days available per year", 0, 365, 180)
    listing_name = st.text_input("Listing title (optional)", "Cozy private room near subway")

if st.button("Predict Price"):
    row = {}
    row['latitude'] = latitude
    row['longitude'] = longitude
    row['distance_to_center'] = haversine_distance(latitude, longitude, *TIMES_SQUARE)
    row['distance_to_jfk'] = haversine_distance(latitude, longitude, *JFK_AIRPORT)
    row['minimum_nights'] = min(minimum_nights, 365)
    row['number_of_reviews'] = number_of_reviews
    row['reviews_per_month'] = reviews_per_month
    row['has_reviews'] = int(has_reviews)
    row['days_since_last_review'] = days_since_last_review
    row['review_density'] = number_of_reviews / (days_since_last_review + 1)
    row['calculated_host_listings_count'] = calculated_host_listings_count
    row['is_multi_listing_host'] = int(calculated_host_listings_count > 1)
    row['availability_365'] = availability_365
    row['occupancy_proxy'] = 365 - availability_365
    row['name_length'] = len(listing_name)
    row['neighbourhood_te'] = neighbourhood_map.get(neighbourhood, global_mean_price)

    host_bucket = pd.cut([calculated_host_listings_count], bins=[0,1,2,5,10,np.inf],
                          labels=['1','2','3-5','6-10','10+'])[0]
    avail_bucket = pd.cut([availability_365], bins=[-1,0,90,270,365],
                           labels=['none','low','medium','high'])[0]
    min_nights_bucket = pd.cut([row['minimum_nights']], bins=[0,3,14,np.inf],
                                labels=['short','medium','long'])[0]

    input_df = pd.DataFrame([row])
    input_df[f'room_type_{room_type}'] = 1
    input_df[f'neighbourhood_group_{neighbourhood_group}'] = 1
    input_df[f'availability_bucket_{avail_bucket}'] = 1
    input_df[f'minimum_nights_bucket_{min_nights_bucket}'] = 1
    input_df[f'host_listings_bucket_{host_bucket}'] = 1
    input_df[f'room_type_neighbourhood_{room_type}_{neighbourhood_group}'] = 1

    # align to the exact feature set the model was trained on; anything missing -> 0
    input_df = input_df.reindex(columns=selected_features, fill_value=0)

    log_pred = model.predict(input_df)[0]
    price_pred = np.expm1(log_pred)

    st.success(f"### Estimated price: ${price_pred:.2f} / night")
    st.caption("Estimate based on a tuned XGBoost model trained on NYC Airbnb listing data.")
