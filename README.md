📈 Google Stock Price Prediction (LSTM)

Predicting Google (GOOG) stock closing prices using LSTM (Long Short-Term Memory) networks on 20 years of Yahoo Finance data.

⚙️ Tools Used

yfinance – fetch stock data

pandas, numpy – data processing

matplotlib – visualization

scikit-learn – MinMaxScaler

tensorflow / keras – LSTM model

🧠 Model Summary

Input: Last 100 days of closing prices

Layers:

LSTM(128, return_sequences=True)

LSTM(64)

Dense(25)

Dense(1)

Optimizer: adam

Loss: mean_squared_error

RMSE ≈ 9.38

📊 Key Steps

Download data: yfinance.download("GOOG")

Compute moving averages (100 & 250 days)

Normalize using MinMaxScaler

Train LSTM (70% train / 30% test)

Predict and plot results

🖼️ Visualization

Plots of:

Closing Price trend

Moving Averages

Predicted vs Actual prices

💾 Save Model
model.save("Latest_stock_price_model.keras")

🚀 Run It
git clone https://github.com/<your-username>/google-stock-lstm.git
cd google-stock-lstm
pip install -r requirements.txt
python google_stock_lstm.py

📚 Future Work

Add more epochs

Try GRU or Bidirectional LSTM

Add indicators like RSI, MACD

Deploy via Streamlit/Flask

🧑‍💻 Author

Prakash khadka
📧 prakash.khadka.pk2@gmail.com

🔗 prakash587
