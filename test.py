import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
import plotly.express as px

# Sample data provided
data = {
    'Date': pd.date_range(start='2014-01-01', end='2023-12-01', freq='MS'),
    'SPX': [
        1782.589966, 1859.449951, 1872.339966, 1883.949951, 1923.569946, 1960.22998, 1930.670044, 2003.369995, 1972.290039, 2018.050049, 2067.560059, 2058.899902,
        1994.98999, 2104.5, 2067.889893, 2085.51001, 2107.389893, 2063.110107, 2103.840088, 1972.180054, 1920.030029, 2079.360107, 2080.409912, 2043.939941,
        1940.23999, 1932.22998, 2059.73999, 2065.300049, 2096.949951, 2098.860107, 2173.600098, 2170.949951, 2168.27002, 2126.149902, 2198.810059, 2238.830078,
        2278.870117, 2363.639893, 2362.719971, 2384.199951, 2411.800049, 2423.409912, 2470.300049, 2471.649902, 2519.360107, 2575.26001, 2647.580078, 2673.610107,
        2823.810059, 2713.830078, 2640.870117, 2648.050049, 2705.27002, 2718.370117, 2816.290039, 2901.52002, 2913.97998, 2711.73999, 2760.169922, 2506.850098,
        2704.100098, 2784.48999, 2834.399902, 2945.830078, 2752.060059, 2941.76001, 2980.379883, 2926.459961, 2976.73999, 3037.560059, 3140.97998, 3230.780029,
        3225.52002, 2954.219971, 2584.590088, 2912.429932, 3044.310059, 3100.290039, 3271.120117, 3500.310059, 3363, 3269.959961, 3621.629883, 3756.070068,
        3714.23999, 3811.149902, 3972.889893, 4181.169922, 4204.109863, 4297.5, 4395.259766, 4522.680176, 4307.540039, 4605.379883, 4567, 4766.180176, 4515.549805,
        4373.939941, 4530.410156, 4131.930176, 4132.149902, 3785.379883, 4130.290039, 3955, 3585.620117, 3871.97998, 4080.110107, 3839.5, 4076.600098, 3970.149902,
        4109.310059, 4169.47998, 4179.830078, 4450.379883, 4588.959961, 4507.660156, 4288.049805, 4193.799805, 4567.799805, 4769.830078
    ],
    'GS1M': [
        0.02, 0.05, 0.05, 0.02, 0.03, 0.02, 0.02, 0.03, 0.01, 0.02, 0.04, 0.03,
        0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.03, 0.04, 0.01, 0.01, 0.07, 0.17,
        0.23, 0.26, 0.25, 0.19, 0.23, 0.22, 0.26, 0.26, 0.19, 0.24, 0.3, 0.42,
        0.5, 0.48, 0.66, 0.75, 0.73, 0.84, 0.97, 0.98, 0.99, 1, 1.09, 1.2,
        1.3, 1.38, 1.64, 1.66, 1.71, 1.81, 1.89, 1.94, 2.04, 2.17, 2.24, 2.37,
        2.4, 2.43, 2.45, 2.43, 2.4, 2.22, 2.15, 2.07, 1.99, 1.73, 1.58, 1.55,
        1.53, 1.58, 0.37, 0.11, 0.1, 0.13, 0.11, 0.08, 0.09, 0.09, 0.09, 0.09,
        0.08, 0.08, 0.04, 0.02, 0.02, 0.01, 0.03, 0.05, 0.04, 0.05, 0.06, 0.07,
        0.04, 0.05, 0.18, 0.31, 0.58, 1.06, 1.85, 2.28, 2.61, 3.32, 3.87, 3.9,
        4.52, 4.64, 4.49, 4.17, 5.49, 5.2, 5.39, 5.54, 5.53, 5.57, 5.53, 5.54
    ]
}
df = pd.DataFrame(data)

# Convert date to mm-yyyy format
df['MonthYear'] = df['Date'].dt.strftime('%m-%Y')

# Streamlit app
st.title('Portfolio Optimisation Tool')

# Find min and max months
min_month = df['Date'].dt.to_period('M').min()
max_month = df['Date'].dt.to_period('M').max()

# User input for in-sample period end month
st.sidebar.write("### Select In-Sample Period")
in_sample_end_month = st.sidebar.selectbox("Select end month for in-sample period",
                                           options=pd.period_range(start=min_month+23, end=max_month, freq='M'))

# User input for out-of-sample period end month
st.sidebar.write("### Select Out-of-Sample Period")
out_sample_start_month = in_sample_end_month + 1
max_allowed_month = out_sample_start_month + 23
out_sample_end_month = st.sidebar.selectbox("Select end month for out-of-sample period (Please select a period of less than 2 years from start date for better prediction)",
                                             options=pd.period_range(start=out_sample_start_month, end=max_allowed_month, freq='M'))

# Display selected periods
st.write("## Selected Periods")
st.write(f"In-Sample Period: {min_month} to {in_sample_end_month}")
st.write(f"Out-of-Sample Period (Forecasting period): {out_sample_start_month} to {out_sample_end_month}")

# Convert periods to timestamps
in_sample_end_month_timestamp = pd.Timestamp(in_sample_end_month.end_time)
out_sample_end_month_timestamp = pd.Timestamp(out_sample_end_month.end_time)

# Filter data for in-sample and out-of-sample periods
in_sample_data = df[df['Date'] <= in_sample_end_month_timestamp]
out_sample_data = df[(df['Date'] > in_sample_end_month_timestamp) and (df['Date'] <= out_sample_end_month_timestamp)]

# Fit SARIMAX models
spx_model = SARIMAX(in_sample_data['SPX'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
spx_results = spx_model.fit()
gs1m_model = SARIMAX(in_sample_data['GS1M'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
gs1m_results = gs1m_model.fit()

# Forecast
spx_forecast = spx_results.get_forecast(steps=len(out_sample_data))
gs1m_forecast = gs1m_results.get_forecast(steps=len(out_sample_data))

# Confidence intervals
spx_conf_int = spx_forecast.conf_int()
gs1m_conf_int = gs1m_forecast.conf_int()

# Plotting with Plotly Express
fig_spx = px.line()
fig_spx.add_scatter(x=in_sample_data['Date'], y=in_sample_data['SPX'], mode='lines', name='In-Sample SPX')
fig_spx.add_scatter(x=out_sample_data['Date'], y=out_sample_data['SPX'], mode='lines', name='Out-of-Sample SPX', line=dict(color='green'))
fig_spx.add_scatter(x=out_sample_data['Date'], y=spx_forecast.predicted_mean, mode='lines', name='Forecasted SPX', line=dict(color='red'))
fig_spx.add_scatter(x=out_sample_data['Date'], y=spx_conf_int.iloc[:, 0], mode='lines', fill=None, line=dict(color='pink'), name='Lower CI SPX')
fig_spx.add_scatter(x=out_sample_data['Date'], y=spx_conf_int.iloc[:, 1], mode='lines', fill='tonexty', line=dict(color='pink'), name='Upper CI SPX')
fig_spx.update_layout(title='SPX Forecasting', xaxis_title='Period', yaxis_title='SPX')
st.plotly_chart(fig_spx)

fig_gs1m = px.line()
fig_gs1m.add_scatter(x=in_sample_data['Date'], y=in_sample_data['GS1M'], mode='lines', name='In-Sample GS1M')
fig_gs1m.add_scatter(x=out_sample_data['Date'], y=out_sample_data['GS1M'], mode='lines', name='Out-of-Sample GS1M', line=dict(color='green'))
fig_gs1m.add_scatter(x=out_sample_data['Date'], y=gs1m_forecast.predicted_mean, mode='lines', name='Forecasted GS1M', line=dict(color='blue'))
fig_gs1m.add_scatter(x=out_sample_data['Date'], y=gs1m_conf_int.iloc[:, 0], mode='lines', fill=None, line=dict(color='lightblue'), name='Lower CI GS1M')
fig_gs1m.add_scatter(x=out_sample_data['Date'], y=gs1m_conf_int.iloc[:, 1], mode='lines', fill='tonexty', line=dict(color='lightblue'), name='Upper CI GS1M')
fig_gs1m.update_layout(title='GS1M Forecasting', xaxis_title='Period', yaxis_title='GS1M')
st.plotly_chart(fig_gs1m)
