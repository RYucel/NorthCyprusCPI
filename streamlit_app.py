import pandas as pd
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import mpld3

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('inflation88seti.csv')  

# Convert date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

# Set date as index 
df.set_index('Date', inplace=True)

# Convert dates to month start
df.index = df.index.to_period('M').to_timestamp() 

# Set frequency 
df.index.freq = 'MS'  

# Redefine df 
df = df.copy()  

# Slider for forecast periods
periods = st.slider('Periods', min_value=6, max_value=24, value=12)

# Train model
model = auto_arima(df['KKTC_CPI']) 

# Interactive plot 
fig, ax = plt.subplots()
df['KKTC_CPI'].plot(ax=ax)

# Tooltips
tooltip = mpld3.plugins.PointLabelTooltip(df.index, labels=df['KKTC_CPI'].round(2))

def plot_forecast():

  # Forecast
  fc = model.predict(periods)    

  # Plot 
  ax.clear()
  df['KKTC_CPI'].plot(ax=ax)
  fc.plot(ax=ax, legend=False)

  # Tooltip
  mpld3.plugins.connect(fig, tooltip)

# Initial call
plot_forecast()  

# On slider change 
st.pyplot(fig)
st.slider.on_change(plot_forecast)
