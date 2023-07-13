import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pmdarima import auto_arima

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('inflation88seti.csv')

# Prepare datetime
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)
df.index = df.index.to_period('M').to_timestamp()
df.index.freq = 'MS'

# Copy data
forecast_df = df['KKTC_CPI'].to_frame('Forecast')

# Model 
model = auto_arima(df['KKTC_CPI'])

# Figure
fig = go.Figure()

# In-sample forecasts
f, conf_int = model.predict_in_sample(return_conf_int=True)

# Get forecast index
fcst_index = f.index

# Add in-sample ribbon
fig.add_trace(go.Scatter(
    x=fcst_index,
    y=conf_int[:,0],
    fill='tonexty',
    mode='lines',
    line_color='blue'))
    
fig.add_trace(go.Scatter(
    x=fcst_index,
    y=conf_int[:,1],
    fill='tonexty',
    mode='lines',
    line_color='blue'))
    
# Out-of-sample forecasts
periods = 12    
fc, conf_int = model.predict(periods, return_conf_int=True)
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = pd.date_range(start=df.index[-1], periods=len(fc), freq='MS')

# Add out-sample ribbon
fig.add_trace(go.Scatter(
    x=fdf.index,
    y=conf_int[:,0],
    fill='tonexty',  
    mode='lines',
    line_color='green'))

fig.add_trace(go.Scatter(
    x=fdf.index,
    y=conf_int[:,1],
    fill='tonexty',
    mode='lines',
    line_color='green'))
    
# Add forecast line
fig.add_trace(go.Scatter(
    x=fdf.index,
    y=fdf['Forecast'],
    mode='lines',
    name='Forecast'))
    
# Second plot 
fig2 = px.line(forecast_df, x=forecast_df.index, y='YoY Change')

# Filtered YoY table
yoy_table = forecast_df.loc[forecast_df.index >= '2020', 'YoY Change']
yoy_table = yoy_table.sort_index(ascending=False)

# Display plots and table  
st.plotly_chart(fig)
st.plotly_chart(fig2) 
st.table(yoy_table.round(2))
