import streamlit as st
import joblib

model = joblib.load('lstm_model.pkl')


st.title('Hotel Cancelation rates based off economic triggers')
st.write('Predictions utilizing stacking ensemble and LSTM')

Days_til_booking = st.number_input('Enter number of days between booking and check out')
Month_of_arrival = st.number_input('Enter the month of arrival (0-12)')
gross_domestic_product = st.number_input('Enter number')
Intrest_rate = st.number_input('Enter Interest Rate')
Inflation_chg = st.number_input('Enter Inflation Change')
Inflation = st.number_input('Enter Inflation')
CPI_avg = st.number_input('Enter CPI Average')
CPI_hotels = st.number_input('Enter CPI Hotels')
Fuel_prc = st.number_input('Enter average Fuel Price')
Unemployment_rate = st.number_input('Enter Unemployment Rate')

if st.button('Predict Cancelation Rate'):
  features = [Days_til_booking, Month_of_arrival, GDP, Intrest_rate, Inflation_chg, Inflation, CPI_avg, CPI_hotels, Fuel_prc, Unemployment_rate]
  
  prediction_LSTM = model.predict([features])[0]

  
  st.write(f'LSTM Prediction: ${prediction_LSTM}')
