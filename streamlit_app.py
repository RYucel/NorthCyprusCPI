import pandas as pd
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('data.csv') 

# Train autoARIMA model
model = auto_arima(df['NorthCyprusCPI'], start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1, trace=True, error_action='ignore', suppress_warnings=True)

# Forecast into future
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = pd.date_range(start='2023-07-01', end='2025-06-30', freq='MS')

# Make DataFrame for forecast
forecast_df = pd.DataFrame(fc, index=index_of_fc, columns=['Prediction'])
forecast_df.index.name = 'Date'

# Plot
fig, ax = plt.subplots()
df['NorthCyprusCPI'].plot(ax=ax)
forecast_df.plot(ax=ax, color='r')
ax.fill_between(forecast_df.index, 
                confint[:, 0],
                confint[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')
ax.set_ylabel('CPI')

# Show plot in app
st.pyplot(fig)
