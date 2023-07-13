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

# Copy data
forecast_df = df['KKTC_CPI'].to_frame('Forecast') 

# Slider
periods = st.slider('Periods', min_value=6, max_value=24, value=12)

# Model
model = auto_arima(df['KKTC_CPI'])

# In-sample forecasts
f, conf_int = model.predict_in_sample(return_conf_int=True) 

# Add in-sample cone
fig.add_ribbon(x=conf_int.index, ymin=conf_int[:,0], 
               ymax=conf_int[:,1], fillcolor='blue', opacity=0.2)

# Out-of-sample forecasts
fc = model.predict(periods) 
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = pd.date_range(start=df.index[-1], periods=len(fc), freq='MS')  

# Add out-of-sample cone
fig.add_ribbon(x=fdf.index, ymin=conf_int[:,0],
               ymax=conf_int[:,1], fillcolor='green', opacity=0.2)
               
# Plot forecast line               
fig.add_scatter(x=fdf.index, y=fdf['Forecast'], mode='lines', name='Forecast')

# Display plot
st.plotly_chart(fig)
