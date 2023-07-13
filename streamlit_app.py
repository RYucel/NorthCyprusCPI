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
fdf = pd.DataFrame(fc, columns=['Forecast']) 
fdf.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=len(fc), freq='MS')

# Plotly chart
fig = px.line(df, x=df.index, y='KKTC_CPI')
fig.add_scatter(x=fdf.index, y=fdf['Forecast'], mode='lines', name='Forecast')
fig.update_traces(hovertemplate='Date: %{x}<br>Value: %{y}')
st.plotly_chart(fig)

# Actual YoY changes
yoy_change = df['KKTC_CPI'].pct_change(periods=12) * 100
recent_yoy = yoy_change.tail(12)

# Display actuals table
st.subheader('Actual YoY Changes')
st.table(recent_yoy.round(2)) 

# Forecast YoY changes
fdf['YoY Forecast'] = fdf['Forecast'].pct_change(periods=12).shift(12) * 100  

# Display forecasts table
st.subheader('Forecasted YoY Changes')
st.table(fdf['YoY Forecast'].round(2))
