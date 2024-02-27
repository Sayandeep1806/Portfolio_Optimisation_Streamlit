import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Function to prepare data for LSTM
def prepare_lstm_data(series, n_steps):
    X, y = [], []
    for i in range(len(series)):
        end_ix = i + n_steps
        if end_ix > len(series)-1:
            break
        seq_x, seq_y = series[i:end_ix], series[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Function to build LSTM model
def build_lstm_model(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

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

# Streamlit app
st.title('Portfolio Optimization Tool')

# User input for in-sample period end month
in_sample_end_month = st.sidebar.selectbox("Select end month for in-sample period",
                                           options=pd.period_range(start=pd.Period('2016-01'), end=pd.Period('2022-12'), freq='M'))

# User input for out-of-sample period end month
out_sample_end_month = st.sidebar.selectbox("Select end month for out-of-sample period",
                                            options=pd.period_range(start=pd.Period('2023-01'), end=pd.Period('2023-12'), freq='M'))

# Convert periods to timestamps
in_sample_end_month_timestamp = pd.Timestamp(in_sample_end_month.end_time)
out_sample_end_month_timestamp = pd.Timestamp(out_sample_end_month.end_time)

# Prepare the data for LSTM
spx_data = df[['Date', 'SPX']].set_index('Date')
gs1m_data = df[['Date', 'GS1M']].set_index('Date')

# Normalize the data
scaler_spx = MinMaxScaler()
scaled_spx_data = scaler_spx.fit_transform(spx_data)

scaler_gs1m = MinMaxScaler()
scaled_gs1m_data = scaler_gs1m.fit_transform(gs1m_data)

# Choose the number of time steps
n_steps = 12

# Prepare the LSTM data
X_spx, y_spx = prepare_lstm_data(scaled_spx_data, n_steps)
X_gs1m, y_gs1m = prepare_lstm_data(scaled_gs1m_data, n_steps)

# Reshape data for LSTM (samples, timesteps, features)
n_features_spx = 1
X_spx = X_spx.reshape((X_spx.shape[0], X_spx.shape[1], n_features_spx))

n_features_gs1m = 1
X_gs1m = X_gs1m.reshape((X_gs1m.shape[0], X_gs1m.shape[1], n_features_gs1m))

# Build LSTM models
model_spx = build_lstm_model(n_steps, n_features_spx)
model_gs1m = build_lstm_model(n_steps, n_features_gs1m)

# Fit LSTM models
model_spx.fit(X_spx, y_spx, epochs=200, verbose=0)
model_gs1m.fit(X_gs1m, y_gs1m, epochs=200, verbose=0)

# Predictions
forecast_period = len(df) - len(spx_data)
forecast_spx = []
forecast_gs1m = []
for i in range(forecast_period):
    x_input_spx = scaled_spx_data[-n_steps:].reshape((1, n_steps, n_features_spx))
    x_input_gs1m = scaled_gs1m_data[-n_steps:].reshape((1, n_steps, n_features_gs1m))

    yhat_spx = model_spx.predict(x_input_spx, verbose=0)[0]
    forecast_spx.append(yhat_spx[0])

    yhat_gs1m = model_gs1m.predict(x_input_gs1m, verbose=0)[0]
    forecast_gs1m.append(yhat_gs1m[0])

    scaled_spx_data = np.append(scaled_spx_data, yhat_spx.reshape(1, -1), axis=0)
    scaled_gs1m_data = np.append(scaled_gs1m_data, yhat_gs1m.reshape(1, -1), axis=0)

# Inverse scaling
forecast_spx = scaler_spx.inverse_transform(np.array(forecast_spx).reshape(-1, 1))
forecast_gs1m = scaler_gs1m.inverse_transform(np.array(forecast_gs1m).reshape(-1, 1))

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['SPX'], label='Actual SPX')
plt.plot(pd.date_range(start=out_sample_end_month_timestamp + pd.DateOffset(months=1), periods=forecast_period, freq='M'), forecast_spx, label='Forecasted SPX')
plt.title('SPX Forecasting')
plt.xlabel('Date')
plt.ylabel('SPX')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt)

plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['GS1M'], label='Actual GS1M')
plt.plot(pd.date_range(start=out_sample_end_month_timestamp + pd.DateOffset(months=1), periods=forecast_period, freq='M'), forecast_gs1m, label='Forecasted GS1M')
plt.title('GS1M Forecasting')
plt.xlabel('Date')
plt.ylabel('GS1M')
plt.legend()
plt.xticks(rotation=45)
st.pyplot(plt)
