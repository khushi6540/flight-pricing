import streamlit as st
import pandas as pd
import joblib

# Load compressed model and encoders
model = joblib.load("xgb_flight_price_model.pkl.gz")
label_encoders = joblib.load("label_encoders.pkl.gz")

st.title("✈️ Flight Price Predictor")

# Input widgets for all features except duration
airline = st.selectbox("Airline", label_encoders['airline'].classes_)
source_city = st.selectbox("Source City", label_encoders['source_city'].classes_)
departure_time = st.selectbox("Departure Time", label_encoders['departure_time'].classes_)
stops = st.selectbox("Stops", label_encoders['stops'].classes_)
arrival_time = st.selectbox("Arrival Time", label_encoders['arrival_time'].classes_)
destination_city = st.selectbox("Destination City", label_encoders['destination_city'].classes_)
flight_class = st.selectbox("Class", label_encoders['class'].classes_)
days_left = st.number_input("Days Left for Departure", min_value=0, max_value=365, value=1)

# Encode user inputs
input_dict = {
    "airline": label_encoders['airline'].transform([airline])[0],
    "source_city": label_encoders['source_city'].transform([source_city])[0],
    "departure_time": label_encoders['departure_time'].transform([departure_time])[0],
    "stops": label_encoders['stops'].transform([stops])[0],
    "arrival_time": label_encoders['arrival_time'].transform([arrival_time])[0],
    "destination_city": label_encoders['destination_city'].transform([destination_city])[0],
    "class": label_encoders['class'].transform([flight_class])[0],
    "days_left": days_left
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Price"):
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Flight Price: ₹{pred:.2f}")
