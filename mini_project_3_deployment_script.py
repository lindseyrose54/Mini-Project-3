import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load the RF model
model = load_model('lstm_model.keras')  # Make sure the .h5 file is in the same directory

st.title('Hotel Cancellation Rates Based on Economic Triggers')
st.write('Predicting whether a guest will cancel their reservation')

Days_til_booking = st.number_input('Enter number of days between booking and check out')
Month_of_arrival = st.number_input('Enter the month of arrival (0-12)')
Gross_domestic_product = st.number_input('Enter GDP')
Interest_rate = st.number_input('Enter Interest Rate')
Inflation_chg = st.number_input('Enter Inflation Change')
Inflation = st.number_input('Enter Inflation')
CPI_avg = st.number_input('Enter Consumer Price Index Average')
CPI_hotels = st.number_input('Enter Consumer Price Index for Hotels')
Fuel_prc = st.number_input('Enter average Fuel Price')
Unemployment_rate = st.number_input('Enter Unemployment Rate')
Disposable_Income_per_Capita = st.number_input('Enter Disposable Income')
Consumer_Sentiment_towards_Economy = st.number_input('Enter CSI')

if st.button('Predict Cancelation Rate'):
    # Prepare features
    features = [Days_til_booking, Month_of_arrival, Gross_domestic_product, Interest_rate,
                Inflation_chg, Inflation, CPI_avg, CPI_hotels, Fuel_prc, Unemployment_rate, Disposable_Income_per_Capita, Consumer_Sentiment_towards_Economy]

    # Convert to NumPy array and reshape for LSTM: (batch_size, time_steps, features)
    features = np.array(features).reshape(1, 1, -1)

    # Make prediction
    prediction_LSTM = model.predict(features)[0][0]

    # Display prediction
    st.write(f'Predicton Score: {prediction_LSTM:.2f}% — 0 means No Cancellation, 1 means Cancellation')
