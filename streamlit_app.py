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

# Forecasts
fc = model.predict(periods)
fc_df = pd.DataFrame(fc, columns=['KKTC_CPI']) 

fdf = pd.DataFrame(fc, columns=['KKTC_CPI'])
fdf.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=len(fc), freq='MS')

# Add forecast column to fdf
fdf = pd.concat([fdf, fc_df['KKTC_CPI']], axis=1)

# YoY Forecast
fdf['YoY Forecast'] = fdf['KKTC_CPI'].pct_change(periods=12) * 100

# CPI Chart
fig1 = px.line(df, x=df.index, y='KKTC_CPI')
fig1.add_scatter(x=fdf.index, y=fdf['KKTC_CPI'], mode='lines', name='Forecast')
fig1.update_traces(hovertemplate='Date: %{x}<br>CPI: %{y}') 

# YoY Change  
df['YoY Change'] = df['KKTC_CPI'].pct_change(periods=12) * 100

fig2 = px.line(df, x=df.index, y='YoY Change')
fig2.add_scatter(x=fdf.index, y=fdf['YoY Forecast'], mode='lines', name='Forecast')
fig2.update_traces(hovertemplate='Date: %{x}<br>YoY Change: %{y:.2f}%')

st.plotly_chart(fig1)
st.plotly_chart(fig2)
