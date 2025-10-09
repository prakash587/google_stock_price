import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

st.title("Stock Price Predictor App")

# Input stock
stock = st.text_input("Enter the Stock Symbol", "GOOG")

# Date range
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download stock data
google_data = yf.download(stock, start, end)

# Flatten column names to match Colab style
google_data.columns = ['_'.join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col) for col in google_data.columns]

st.subheader("Stock Data")
st.write(google_data)

# Split test data
splitting_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data['Close_'+stock][splitting_len:])

# Plotting function for moving averages
def plot_moving_average(values, title):
    fig, ax = plt.subplots(figsize=(15,6))
    ax.plot(google_data.index, google_data['Close_'+stock], label='Close Price', color='blue')
    ax.plot(google_data.index, values, label=title, color='orange')
    ax.set_title(f"{stock} {title}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Calculate moving averages
google_data['MA_for_250_days'] = google_data['Close_'+stock].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close_'+stock].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close_'+stock].rolling(100).mean()

# Plot moving averages
st.subheader("Moving Averages")
plot_moving_average(google_data['MA_for_250_days'], 'MA for 250 Days')
plot_moving_average(google_data['MA_for_200_days'], 'MA for 200 Days')
plot_moving_average(google_data['MA_for_100_days'], 'MA for 100 Days')

# Plot 100-day vs 250-day together
fig, ax = plt.subplots(figsize=(15,6))
ax.plot(google_data.index, google_data['Close_'+stock], label='Close Price', color='blue')
ax.plot(google_data.index, google_data['MA_for_100_days'], label='MA 100 Days', color='orange')
ax.plot(google_data.index, google_data['MA_for_250_days'], label='MA 250 Days', color='green')
ax.set_title(f"{stock} Close Price with MA 100 & 250 Days")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Scale test data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test.values.reshape(-1,1))

# Prepare sequences for model
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Load model and predict
model = load_model("models/Latest_stock_price_model.keras")
predictions = model.predict(x_data)

# Inverse transform
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

# Prepare DataFrame for plotting
ploting_data = pd.DataFrame(
    {
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    },
    index = google_data.index[splitting_len+100:]
)

st.subheader("Original vs Predicted Test Data")
st.write(ploting_data)

# Plot original vs predicted
st.subheader('Original Close Price vs Predicted Close Price')

fig, ax = plt.subplots(figsize=(15,6))

# Historical data not used in test
ax.plot(google_data.index[:splitting_len+100],
        google_data['Close_'+stock][:splitting_len+100],
        label="Data - not used", color='blue')

# Original test data
ax.plot(ploting_data.index,
        ploting_data['original_test_data'],
        label="Original Test Data", color='green')

# Predicted test data
ax.plot(ploting_data.index,
        ploting_data['predictions'],
        label="Predicted Test Data", color='orange')

# Formatting
ax.set_title(f"{stock} Close Price Prediction")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
ax.grid(True)

st.pyplot(fig)
