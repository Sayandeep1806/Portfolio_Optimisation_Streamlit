import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
import plotly.express as px
import plotly.graph_objects as go

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

# Convert annualized returns to monthly returns
df['GS1M_Monthly_Returns'] = ((1 + (df['GS1M'] / 12)) ** (1/12)) - 1

# Function to forecast SPX values
def forecast_SPX(spx_data, in_sample_start_month,in_sample_end_month, out_sample_end_month):
    forecasts = []
    
    # Iterate over each month in the forecasting period
    while in_sample_end_month < out_sample_end_month:
        # Subset the data for the current in-sample period
        in_sample_data = spx_data[spx_data['Date'].dt.to_period('M').between(in_sample_start_month, in_sample_end_month)]
        
        # Fit SARIMAX model to the in-sample data
        model = SARIMAX(in_sample_data['SPX'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit()
        
        # Forecast the SPX value for the next month
        forecast_month = in_sample_end_month + 1
        forecast_value = results.forecast(steps=1)
        forecasts.append((forecast_month, forecast_value.values[0]))
        
        # Update the in-sample period for the next iteration
        in_sample_end_month += 1
        in_sample_start_month += 1
    
    return forecasts

# Streamlit app
st.title('Portfolio Optimisation Tool')

# Ask the user to enter the risk aversion
st.write("## Select User Risk Appetite")
risk_aversion = st.slider("Risk Aversion (γ): Most Risk Taking <------> Most Risk Averse ",min_value=1, max_value=10.0, step=0.1, value=2.0)
st.write("Risk Aversion (γ) = ", risk_aversion)

# Find min and max months
min_month = df['Date'].dt.to_period('M').min()
max_month = df['Date'].dt.to_period('M').max()

# User input for in-sample period end month
st.sidebar.write("### Select In-Sample Period")
st.sidebar.write(f"The data is available from {min_month} to {max_month}. Please allow atleast 2 years of data for training the model.")
in_sample_start_month = st.sidebar.selectbox("Select start month for in-sample period",
                                           options=pd.period_range(start=min_month, end=max_month-24, freq='M'))
in_sample_end_month = st.sidebar.selectbox("Select end month for initial in-sample period",
                                           options=pd.period_range(start=in_sample_start_month+23, end=max_month, freq='M'))

# User input for out-of-sample period end month
st.sidebar.write("### Select Out-of-Sample Period")
out_sample_start_month = in_sample_end_month + 1
max_allowed_month = max_month + 1
out_sample_end_month = st.sidebar.selectbox("Select end month for forecasting period",
                                             options=pd.period_range(start=out_sample_start_month, end=max_allowed_month, freq='M'))

# Display selected periods
st.write("## Selected Periods")
st.write(f"In-Sample Period: From {in_sample_start_month} to {in_sample_end_month}")
st.write(f"Out-of-Sample Period (Forecasting period): From {out_sample_start_month} to {out_sample_end_month}")

# Function call to forecast SPX values
forecasts = forecast_SPX(df, in_sample_start_month, in_sample_end_month, out_sample_end_month)

# Prepare data for plotting
forecast_dates = [forecast[0].to_timestamp() for forecast in forecasts]
forecast_values = [forecast[1] for forecast in forecasts]

actual_dates = df['Date']
actual_values = df['SPX']

# Filter data for plotting
filtered_df = df[(df['Date'] >= pd.Timestamp(in_sample_start_month.to_timestamp())) & (df['Date'] <= pd.Timestamp(out_sample_end_month.to_timestamp()))]



# Create DataFrame for plotting upper and lower bounds
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Actual_SPX': df[df['Date'].isin([forecast[0].to_timestamp() for forecast in forecasts])]['SPX'].values,
    'Forecasted_SPX': forecast_values,
    'GS1M': df[df['Date'].isin([forecast[0].to_timestamp() for forecast in forecasts])]['GS1M'].values,
    'GS1M_Monthly_Returns': df[df['Date'].isin([forecast[0].to_timestamp() for forecast in forecasts])]['GS1M_Monthly_Returns'].values
})

# Calculating upper and lower bounds
forecast_df['Upper_Bound'] = forecast_df['Forecasted_SPX'] + 100  # Adjust upper bound as needed
forecast_df['Lower_Bound'] = forecast_df['Forecasted_SPX'] - 100  # Adjust lower bound as needed

# Plotting the results
fig = px.line(filtered_df, x='Date', y='SPX', title='Actual SPX Values vs Forecasted SPX Values')

# Add forecasted values and bounds as shaded area
fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecasted_SPX'], mode='lines', line=dict(color='red'), name='Forecasted SPX')
fig.add_trace(go.Scatter(
    x=forecast_df['Date'].tolist() + forecast_df['Date'].tolist()[::-1],
    y=forecast_df['Upper_Bound'].tolist() + forecast_df['Lower_Bound'].tolist()[::-1],
    fill='toself',
    fillcolor='rgba(255,192,203,0.5)',
    line_color='rgba(255,192,203,0)',
    showlegend=False
))

# Add legend for actual SPX values
fig.add_trace(go.Scatter(x=actual_dates, y=actual_values, mode='lines', name='Actual SPX', line=dict(color='blue')))

st.plotly_chart(fig)

# Calculating returns
forecast_df['Actual_SPX_Returns'] = np.log(forecast_df['Actual_SPX'] / forecast_df['Actual_SPX'].shift(1))
forecast_df['Forecasted_SPX_Returns'] = np.log(forecast_df['Forecasted_SPX'] / forecast_df['Forecasted_SPX'].shift(1))
# Replacing the first 'NaN' values of Actual and Forecasted returns using the last SPX value of the in-sample data
forecast_df['Actual_SPX_Returns'][0] = np.log(forecast_df['Actual_SPX'].iloc[0] /filtered_df['SPX'].iloc[-1])
forecast_df['Forecasted_SPX_Returns'][0] = np.log(forecast_df['Forecasted_SPX'].iloc[0] /filtered_df['SPX'].iloc[-1])

# Plot Actual and Forecasted Excess Returns on SPX
fig_returns = go.Figure()
fig_returns.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Actual_SPX_Returns'], mode='lines', name='Actual SPX Returns'))
fig_returns.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecasted_SPX_Returns'], mode='lines', name='Forecasted SPX Returns'))
fig_returns.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['GS1M_Monthly_Returns'], mode='lines', name='GS1M Monthly Returns'))
fig_returns.update_layout(title='Actual vs Forecasted Returns on SPX vs US Treasury Returns', xaxis_title='Period', yaxis_title='Excess Returns')
st.plotly_chart(fig_returns)

# Calculating excess returns	
forecast_df['Actual_SPX_Excess_Returns'] = forecast_df['Actual_SPX_Returns'] - forecast_df['GS1M_Monthly_Returns']
forecast_df['Forecasted_SPX_Excess_Returns'] = forecast_df['Forecasted_SPX_Returns'] - forecast_df['GS1M_Monthly_Returns']

# Calculating Variance
mean_of_SPX_returns = np.log(filtered_df['SPX'] / filtered_df['SPX'].shift(1)).mean()    # Finding mean of actual SPX returns   
forecast_df['Actual_SPX_Variance'] = (forecast_df['Actual_SPX_Excess_Returns']- mean_of_SPX_returns) ** 2 
forecast_df['Forecasted_SPX_Variance'] = (forecast_df['Forecasted_SPX_Excess_Returns']- mean_of_SPX_returns) ** 2 

# Finding weights for SPX
def assign_weight(ER,Var):
    if ER<=0:
        return 0
    else:
        wt = ER/Var
        return max(0, min(wt, 100))

forecast_df['initial_weights_actual'] = forecast_df.apply(lambda row: assign_weight(row['Actual_SPX_Excess_Returns'], row['Actual_SPX_Variance']), axis=1)
forecast_df['initial_weights_forecasted'] = forecast_df.apply(lambda row: assign_weight(row['Forecasted_SPX_Excess_Returns'], row['Forecasted_SPX_Variance']), axis=1)

forecast_df['risk_adjusted_weights_actual'] = forecast_df['initial_weights_actual']/risk_aversion
forecast_df['risk_adjusted_weights_forecasted'] = forecast_df['initial_weights_forecasted']/risk_aversion

# Finding Portfolio Returns over the out-of-sample period 
forecast_df['portfolio_returns_actual'] = ((forecast_df['risk_adjusted_weights_actual']*forecast_df['Actual_SPX_Returns'])
                                           + ((100-forecast_df['risk_adjusted_weights_actual'])*forecast_df['GS1M_Monthly_Returns']))/100
forecast_df['portfolio_returns_forecasted'] = ((forecast_df['risk_adjusted_weights_forecasted']*forecast_df['Forecasted_SPX_Returns'])
                                           + ((100-forecast_df['risk_adjusted_weights_forecasted'])*forecast_df['GS1M_Monthly_Returns']))/100
                                           
# Finding the volatility
forecast_df['volatility_actual_returns'] = forecast_df['portfolio_returns_actual'].std()
forecast_df['volatility_forecasted_returns'] = forecast_df['portfolio_returns_forecasted'].std()  
                                
# Plot Actual and Forecasted portfolio returns
fig_returns_port = go.Figure()
fig_returns_port.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['portfolio_returns_actual'], mode='lines', name='Actual Portfolio Returns'))
fig_returns_port.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['portfolio_returns_forecasted'], mode='lines', name='Forecasted Portfolio Returns'))
fig_returns_port.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['GS1M_Monthly_Returns'], mode='lines', name='GS1M Monthly Returns'))
fig_returns_port.update_layout(title='Actual vs Forecasted Returns on Portfolio vs US Treasury Returns', xaxis_title='Period', yaxis_title='Portfolio Returns')
st.plotly_chart(fig_returns_port)

# Modifying the Date in dataframe
forecast_df['Date'] = forecast_df['Date'].dt.date
forecast_df = forecast_df.set_index("Date")

# Renaming columns
forecast_df = forecast_df.rename(columns = {'Actual_SPX':'Actual SPX','Forecasted_SPX':'Forecasted SPX',
                                            'Actual_SPX_Returns':'Actual SPX Returns','Forecasted_SPX_Returns':'Forecasted SPX Returns',
                                            'GS1M_Monthly_Returns':'US Treasury Returns',
                                            'risk_adjusted_weights_actual':'Actual risk adjusted weights for SPX (out of 100)','risk_adjusted_weights_forecasted':'Forecasted risk adjusted weights for SPX (out of 100)',
                                            'portfolio_returns_actual':'Actual portfolio returns','portfolio_returns_forecasted':'Forecasted portfolio returns',
                                            'volatility_actual_returns':'Actual returns volatility','volatility_forecasted_returns':'Forecasted returns volatility'})

# Display the data in tabular format
st.write("## Portfolio Performance Evaluation Data")
st.dataframe(forecast_df[['Actual SPX','Forecasted SPX','Actual SPX Returns','Forecasted SPX Returns','US Treasury Returns',
                          'Actual risk adjusted weights for SPX (out of 100)','Forecasted risk adjusted weights for SPX (out of 100)',
                          'Actual portfolio returns','Forecasted portfolio returns',
                          'Actual returns volatility','Forecasted returns volatility']])
