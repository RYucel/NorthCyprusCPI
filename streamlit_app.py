import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pmdarima import auto_arima

st.title('North Cyprus CPI Forecast')

# Load data 
df = pd.read_csv('inflation88seti.csv')

# Prepare data
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 
df.set_index('Date', inplace=True)
df.index = df.index.to_period('M').to_timestamp()
df.index.freq = 'MS'  

# Model
model = auto_arima(df['KKTC_CPI'])

# Create figure
fig = go.Figure()

# In-sample forecasts
f, conf_int = model.predict_in_sample(return_conf_int=True)

# Add in-sample ribbon
fig.add_trace(go.Scatter(
    x=conf_int.index, 
    y=conf_int[:,0],
    fill='tonexty', 
    mode='lines',
    line_color='blue'
))

fig.add_trace(go.Scatter(
    x=conf_int.index,
    y=conf_int[:,1],
    fill='tonexty',
    mode='lines',
    line_color='blue'  
))

# Out-of-sample forecasts
fc = model.predict(periods)
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = pd.date_range(start=df.index[-1], periods=len(fc), freq='MS')

# Add forecast line 
fig.add_trace(go.Scatter(
    x=fdf.index, 
    y=fdf['Forecast'],
    mode='lines',
    name='Forecast'
))

# Display figure
st.plotly_chart(fig)
