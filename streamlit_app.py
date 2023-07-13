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

# Copy actual data
df['Forecast'] = df['KKTC_CPI']  

# Slider
periods = st.slider('Periods', min_value=6, max_value=24, value=12)

# Model
model = auto_arima(df['KKTC_CPI'])  

# Forecasts
fc = model.predict(periods)
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = pd.date_range(start=df.index[-1], periods=len(fc), freq='MS') 

# Append forecasts  
df['Forecast'] = df['Forecast'].append(fdf['Forecast'])
df = df.append(fdf)

# Plot
fig = px.line(df, x=df.index, y='Forecast')
st.plotly_chart(fig)

# Calculate YoY  
df['YoY Change'] = df['Forecast'].pct_change(periods=12) * 100

# Display table
st.table(df['YoY Change'].round(2))
