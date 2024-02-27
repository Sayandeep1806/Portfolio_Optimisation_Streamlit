import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# Sample data provided
data = "data.csv"
data_read = pd.read_csv(data)
data_1 = data_read[['Date','SPX','GS1M']]
data_2 = data_1.dropna(how='all').reset_index(drop='index')
df = data_2

# Convert date to mm-yyyy format
df['MonthYear'] = df['Date'].dt.strftime('%m-%Y')

# Streamlit app
st.title('Date Selection App')

# Find min and max months
min_month = df['Date'].dt.to_period('M').min()
max_month = df['Date'].dt.to_period('M').max()

# User input for in-sample period end month
st.sidebar.write("### Select In-Sample Period")
in_sample_end_month = st.sidebar.selectbox("Select end month for in-sample period",
                                           options=pd.period_range(start=min_month, end=max_month, freq='M'))

# User input for out-of-sample period end month
st.sidebar.write("### Select Out-of-Sample Period")
out_sample_start_month = in_sample_end_month + 1
max_allowed_month = out_sample_start_month + 24
out_sample_end_month = st.sidebar.selectbox("Select end month for out-of-sample period",
                                             options=pd.period_range(start=out_sample_start_month, end=max_allowed_month, freq='M'))

# Display selected periods
st.write("## Selected Periods")
st.write(f"In-Sample Period: {min_month} to {in_sample_end_month}")
st.write(f"Out-of-Sample Period: {out_sample_start_month} to {out_sample_end_month}")

# Display data for selected periods
in_sample_data = df[df['Date'].dt.to_period('M') <= in_sample_end_month.end_time]
out_sample_data = df[(df['Date'].dt.to_period('M') >= out_sample_start_month.start_time) &
                     (df['Date'].dt.to_period('M') <= out_sample_end_month.end_time)]

st.write("## Data for Selected Periods")
st.write("### In-Sample Period")
st.write(in_sample_data)

st.write("### Out-of-Sample Period")
st.write(out_sample_data)
