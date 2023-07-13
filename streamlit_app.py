import pandas as pd
import streamlit as st
import plotly.express as px
from pmdarima import auto_arima

# Load and prepare data
@st.cache
def load_data():
    df = pd.read_csv("data.csv")
    df["Date"] = pd.to_datetime(df["Date"]) 
    df.set_index("Date", inplace=True)
    df.index = df.index.to_period("M").to_timestamp()
    df.index.freq = "MS"
    return df

df = load_data()

# Create model
@st.cache
def create_model(df):
    model = auto_arima(df["KKTC_CPI"])
    return model

model = create_model(df)

# Generate forecasts
def generate_forecasts(model, periods):
   fc = model.predict(periods)
   fdf = pd.DataFrame(fc, columns=["Forecast"])
   fdf.index = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=len(fc), freq="MS")
   return fdf

# Plotly chart
def plot_chart(df, fdf):
    fig = px.line(df, x=df.index, y="KKTC_CPI") 
    fig.add_scatter(x=fdf.index, y=fdf["Forecast"], mode="lines", name="Forecast")
    fig.update_traces(hovertemplate="Date: %{x}<br>Value: %{y}")
    st.plotly_chart(fig)
    
# Run app
periods = st.slider("Periods", min_value=6, max_value=24, value=12)
fdf = generate_forecasts(model, periods) 
plot_chart(df, fdf)
