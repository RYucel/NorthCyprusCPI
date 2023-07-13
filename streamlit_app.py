import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error

st.title('North Cyprus CPI Forecast')

# Load data
df = pd.read_csv('inflation88seti.csv')

# Prepare data
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

# Forecasts
fc = model.predict(periods)
fdf = pd.DataFrame(fc, columns=['Forecast'])
fdf.index = pd.date_range(start=df.index[-1], periods=len(fc), freq='MS')

# Concatenate
forecast_df = pd.concat([forecast_df, fdf])

# Calculate YoY
forecast_df['YoY Change'] = forecast_df['Forecast'].pct_change(
    periods=12) * 100

# Filter table
yoy_table = forecast_df.loc[forecast_df.index >= '2020', 'YoY Change']

# Sort table
yoy_table = yoy_table.sort_index(ascending=False)

# Plot 1
fig1 = px.line(forecast_df, x=forecast_df.index, y='Forecast')

# Plot 2
fig2 = px.line(forecast_df, x=forecast_df.index, y='YoY Change')

# Model metric
smape = mean_absolute_percentage_error(
    df['KKTC_CPI'], model.predict_in_sample())
st.markdown(f"**SMAPE:** {smape:.2%}")

# Display plots
st.plotly_chart(fig1)
st.plotly_chart(fig2)

# Display table
st.table(yoy_table.round(2))
# python -m streamlit run app.py --server.port 8502
