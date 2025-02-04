import streamlit as st
import joblib
import numpy as np

# Load the model and scalers
model = joblib.load('PM25_Modeling_XGBoost_Scenario_6.pkl')
scalerX = joblib.load('scalerX_6.pkl')  # You need to save your input scaler
scalery = joblib.load('scalery_6.pkl')  # You need to save your target scaler

st.title("PM2.5 Prediction App 🌫️")
st.markdown("Predict PM2.5 levels based on weather parameters")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=25.0)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=3.0)

with col2:
    rainfall = st.number_input("Total Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0)
    evaporation = st.number_input("Evaporation (mm)", min_value=0.0, max_value=50.0, value=5.0)
    sunshine = st.number_input("Sunshine Hours", min_value=0.0, max_value=24.0, value=6.0)

# Create feature array in correct order
input_features = np.array([[humidity, temperature, wind_speed, 
                           rainfall, evaporation, sunshine]])

# Scale the input features
scaled_input = scalerX.transform(input_features)

# Make prediction
if st.button("Predict PM2.5"):
    scaled_prediction = model.predict(scaled_input)
    prediction = scalery.inverse_transform(scaled_prediction.reshape(-1, 1))
    
    st.subheader("Prediction Result")
    st.success(f"Predicted PM2.5 Level: {prediction[0][0]:.2f} μg/m³")
    
    # Add interpretation
    if prediction[0][0] > 30:
        st.warning("⚠️ Air quality is unhealthy!")
    else:
        st.info("✅ Air quality is within safe limits")