import pandas as pd
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt

st.title('North Cyprus CPI Forecast') 

# Load data
df = pd.read_csv('inflation88seti.csv')  

# Make sure index is datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Set monthly frequency 
df.index.freq = 'MS'

# Train model
model = auto_arima(df['KKTC_CPI'])

# Forecast  
n_periods = 24
fc = model.predict(n_periods=n_periods)

# Make forecast dataframe
index_of_fc = pd.date_range(start='2023-07-01', periods=n_periods, freq='MS')
forecast_df = pd.DataFrame(fc, index=index_of_fc, columns=['Prediction'])

# Plot
fig, ax = plt.subplots()
df['KKTC_CPI'].plot(ax=ax)
forecast_df.plot(ax=ax)

# Display plot
st.pyplot(fig)
