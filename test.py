import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

def clean_data(data):
    data_1 = data[['Date','SPX','GS1M']]
    portfolio_data = data_1.dropna(how='all').reset_index(drop='index')
    portfolio_data['Date'] = pd.to_datetime(portfolio_data['Date'])
    portfolio_data['Date'] = portfolio_data['Date'].dt.strftime('%d-%m-%Y')
    return portfolio_data

    # Load data
    df1 = pd.read_csv("data.csv")  # Change "data.csv" to the actual file name
    df = clean_data(df1)


# Function to forecast SPX excess returns
def forecast_excess_returns(df, insample_size):
    # Compute excess returns
    df['Excess_Returns'] = df['SPX'].pct_change() * 100
    df = df.dropna()

    # Perform forecasting
    model = SimpleExpSmoothing(df['Excess_Returns'].iloc[:insample_size])
    fit_model = model.fit()
    forecast = fit_model.forecast(len(df) - insample_size)

    return forecast

# Streamlit app
def main():
    st.title('SPX Excess Returns Forecasting')

    # Sidebar - Select insample size
    insample_size = st.sidebar.slider('Select insample size', min_value=10, max_value=len(df)-1, value=50)

    # Forecast and plot
    forecast = forecast_excess_returns(df, insample_size)
    actual = df['Excess_Returns'].values[insample_size:]

    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(forecast, label='Forecast')
    plt.title('SPX Excess Returns Forecast vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Excess Returns (%)')
    plt.legend()
    st.pyplot()

if __name__ == '__main__':
    main()
