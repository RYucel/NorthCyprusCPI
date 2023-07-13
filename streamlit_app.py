import pandas as pd  
import streamlit as st
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import mpld3

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('inflation88seti.csv')   

# Prepare datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True) 
df.index = df.index.to_period('M').to_timestamp()
df.index.freq = 'MS'  

# Define df 
df = df.copy()

# Slider 
st.session_state['slider'] = st.slider('Periods', min_value=6, max_value=24, value=12) 

# Access value
periods = st.session_state['slider']  

# Model
model = auto_arima(df['KKTC_CPI'])

# Tooltips
tooltip = mpld3.plugins.PointLabelTooltip(df.index, labels=df['KKTC_CPI'].round(2))

# Plot
fig, ax = plt.subplots()
df['KKTC_CPI'].plot(ax=ax)

def plot_forecast():

  fc = model.predict(periods)  

  ax.clear()
  df['KKTC_CPI'].plot(ax=ax)
  fc.plot(ax=ax, legend=False)

  mpld3.plugins.connect(fig, tooltip)
  
# Initial call 
plot_forecast()

# On change
st.session_state['slider'].on_change(plot_forecast) 

st.pyplot(fig)
