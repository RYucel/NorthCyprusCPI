import pandas as pd 
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt

st.title('North Cyprus CPI Forecast')

# Load data
# ...Code to load and prepare data

# Slider to select forecast period
periods = st.slider('Forecast periods', min_value=6, max_value=24, value=12) 

# Train model
model = auto_arima(df['KKTC_CPI'])

# Interactive plot
fig, ax = plt.subplots() 
df['KKTC_CPI'].plot(ax=ax)

def plot_forecast():
  # Make forecasts
  fc = model.predict(periods)  
  
  # Add tooltips
  tooltip = mpld3.plugins.PointLabelTooltip(fc.index, labels=fc['Prediction'].round(2))
  mpld3.plugins.connect(fig, tooltip)
  
  # Plot forecasts
  ax.clear()
  df['KKTC_CPI'].plot(ax=ax)
  fc.plot(ax=ax, legend=False)
  
# Call when slider changes
st.pyplot(fig) 
plot_forecast()

st.slider.on_change(plot_forecast)
