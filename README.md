# Portfolio_Optimisation_Streamlit
A small-scale portfolio optimization tool that allows the user to interact through a web application. The goal is to decide the weight to be assigned to a broad stock index for portfolio optimisation given the past data and user risk aversion.

Documentation for Portfolio Optimization Tool

Introduction:
The Portfolio Optimization Tool is a Streamlit web application designed to assist users in optimizing their investment portfolio based on their risk aversion. This tool utilizes historical data of the S&P 500 index (SPX) and US Treasury bond yields to forecast future SPX values and optimize portfolio allocation.

Key Features:

User Risk Appetite Selection: Allows users to select their risk aversion level using a slider. This risk aversion parameter, denoted as γ, determines the user's preference for risk-taking versus risk aversion.

In-Sample and Out-of-Sample Period Selection: Users can select the time period for model training (in-sample) and forecasting (out-of-sample). The tool recommends using at least two years of historical data for training the forecasting model.

Forecasting SPX Values: Utilizes the SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model to forecast future SPX values for the out-of-sample period. The model is trained using historical SPX data within the selected in-sample period. For each new out-of-sample period the model has been trained starting from the in-sample period till one period before the forecasting period iteratively.

[Note: The hyperparameters have been selected using using grid-search approach on the above SARIMAX model. We have also tried using LSTM and FB Prophet libraries, but due to small sample size the predicted results were not good. Alternative to FB Prophet would have been neural Prophet library which performs better on small sample size, but since it is in early stage of development and not stable we have refrained from using the same.]

Calculating Returns: Calculate log returns at time t and t+1.
Returns = log([SPX]t+1 /[SPX]t)

Excess Returns: Returns[SPX] - US Treasury monthly returns
[Note: Convert Annualised GS1M returns to monthly returns]

Variance: Returns[SPX] - Mean of Returns[SPX]

Visualizations:

Actual vs. Forecasted SPX Values: Displays a line plot comparing actual SPX values with forecasted SPX values for the selected periods.
Actual vs. Forecasted Returns: Shows the actual and forecasted returns of the SPX index along with the monthly returns of US Treasury bonds.
Actual vs. Forecasted Portfolio Returns: Illustrates the actual and forecasted returns of the optimized portfolio compared to US Treasury bond returns.

Portfolio Optimization:

Risk-Adjusted Weights: Calculates the risk-adjusted weights for investing in SPX based on the user's risk aversion. Higher risk aversion leads to lower weights assigned to SPX.
Portfolio Returns: Computes the portfolio returns based on the allocated weights for SPX and US Treasury bonds. The actual and forecasted portfolio returns are displayed.

Usage Instructions:

Select User Risk Appetite: Adjust the risk aversion slider to specify the level of risk tolerance.
Select In-Sample Period: Choose the start and end months for the in-sample period, ensuring a sufficient amount of historical data is available for training.
Select Out-of-Sample Period: Specify the end month for the forecasting period.
Review Selected Periods: View the selected in-sample and out-of-sample periods to ensure accuracy.
Analyze Results: Review the visualizations and tabular data to understand the forecasted SPX values, returns, and portfolio performance.

Conclusion:

The Portfolio Optimization Tool provides users with insights into optimizing their investment portfolios based on their risk preferences. By leveraging historical data and forecasting techniques, users can make informed decisions to achieve their investment goals while managing risk effectively.
