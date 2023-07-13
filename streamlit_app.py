import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('inflation88seti.csv')

# Prepare datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.index = df.index.to_period('M').to_timestamp() 
df.index.freq = 'MS'

# Slider
periods = st.slider('Periods', min_value=6, max_value=24, value=12)

# Model
model = auto_arima(df['KKTC_CPI'])

# Generate forecasts
fc = model.predict(periods)
fdf = pd.DataFrame(fc, columns=['Forecast']) 

# Create forecast index
fdf.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=len(fc), freq='MS')

# Plotly chart 
fig = px.line(df, x=df.index, y='KKTC_CPI')
fig.add_scatter(x=fdf.index, y=fdf['Forecast'], mode='lines', name='Forecast')
fig.update_traces(hovertemplate='Date: %{x}<br>Value: %{y}')

st.plotly_chart(fig)
