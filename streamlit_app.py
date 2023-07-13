import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima


st.title('North Cyprus CPI Forecast')

# Initialize session state
if "slider" not in st.session_state:
    st.session_state["slider"] = None

# Load data  
df = pd.read_csv('inflation88seti.csv')

# Prepare dataframe
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) 
df.set_index('Date', inplace=True)
df.index = df.index.to_period('M').to_timestamp()
df.index.freq = 'MS'
df = df.copy()

# Slider
periods = st.slider('Periods', min_value=6, max_value=24, value=12)  

# Model
model = auto_arima(df['KKTC_CPI'])

# Plotly figure
fig = px.line(df, x=df.index, y='KKTC_CPI')

# Add forecast
fc = model.predict(periods)
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = df.index[-1] + pd.DateOffset(months=1):df.index[-1] + pd.DateOffset(months=len(fc))
fig.add_scatter(x=fdf.index, y=fdf['Forecast'], mode='lines', name='Forecast')

# Tooltips
fig.update_traces(hovertemplate='Date: %{x}<br>Value: %{y}')

st.plotly_chart(fig)
# Initial call
plot_forecast()  

# Register on change callback
st.session_state["slider"].on_change(plot_forecast)

st.pyplot(fig)
