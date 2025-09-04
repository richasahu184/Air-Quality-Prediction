import streamlit as st
import joblib
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load the model, scaler, and label encoder at the start
try:
    model = joblib.load('air_quality_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    st.write("Model, Scaler, and Label Encoder loaded successfully.")
except Exception as e:
    st.write(f"Error loading model, scaler, or label encoder: {e}")

# Initialize session state for prediction result
if "aqi_result" not in st.session_state:
    st.session_state.aqi_result = None

# Streamlit app UI
st.title('🌫️ Air Quality Prediction')

# Input fields for pollutants
pm25 = st.number_input('PM2.5', min_value=0.0, value=None)
pm10 = st.number_input('PM10', min_value=0.0, value=None)
no2 = st.number_input('NO2', min_value=0.0, value=None)
so2 = st.number_input('SO2', min_value=0.0, value=None)
co = st.number_input('CO', min_value=0.0, value=None)
temperature = st.number_input('Temperature', min_value=-100.0, max_value=100.0, value=None)

# Predict button
if st.button('Predict'):
    try:
        if any(v is None for v in [pm25, pm10, no2, so2, co, temperature]):
            st.warning("⚠️ Please enter all values.")
        else:
            features = np.array([[pm25, pm10, no2, so2, co, temperature]])
            features_scaled = scaler.transform(features)
            predicted_aqi = model.predict(features_scaled)
            predicted_category = label_encoder.inverse_transform([int(round(predicted_aqi[0]))])[0]
            st.success(f"✅ The predicted Air Quality is: **{predicted_category}**")
            st.session_state.aqi_result = predicted_category
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# Email form UI
st.markdown("---")
st.subheader("📧 Send Report via Email")
name = st.text_input("Enter your Name:")
email = st.text_input("Enter your Email Address:")

def send_email(name, email, result):
    sender_email = "manimanip1622@gmail.com"
    sender_password = "qwnnibowhgsgplud"  # Use app-specific password for Gmail

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = email
    msg["Subject"] = "Air Quality Prediction Report"

    body = f"""
    Hello {name},

    Based on the data you provided, the predicted Air Quality is: {result}

    Tips:
    - If air quality is Poor or Very Poor, limit outdoor activity.
    - Use air purifiers if possible and stay hydrated.
    - Monitor updates from your local environmental agency.

    Stay safe!

    Regards,
    Air Quality AI System
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        return str(e)

if st.button("Send Report"):
    if not name or not email:
        st.error("Please enter both your name and email.")
    elif not st.session_state.aqi_result:
        st.error("Please predict the air quality first.")
    else:
        result = st.session_state.aqi_result
        status = send_email(name, email, result)
        if status is True:
            st.success("✅ Report sent successfully!")
        else:
            st.error(f"❌ Failed to send email: {status}")
