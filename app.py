import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# --- Load BTC data ---
@st.cache_data
def load_btc_data():
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="10y", interval="1d")
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_7d'] = df['Daily_Return'].rolling(window=7).std()
    df['RSI'] = calculate_rsi(df)
    df['EMA_8'] = calculate_ema(df, 8)
    df['EMA_21'] = calculate_ema(df, 21)
    df['EMA_55'] = calculate_ema(df, 55)
    df['EMA_200'] = calculate_ema(df, 200)
    macd, signal, hist = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist
    df['Relative_Volume'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    df = df.dropna()
    return df

# --- Technical Indicator Functions ---
def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    short_ema = calculate_ema(data, short_period)
    long_ema = calculate_ema(data, long_period)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# --- Forecasting function ---
def forecast_next_5_days(model, scaler_X, scaler_y, df_features, lookback):
    last_seq = df_features.iloc[-lookback:].values
    preds = []
    for _ in range(5):
        input_seq = scaler_X.transform(last_seq).reshape(1, lookback, -1)
        next_scaled = model.predict(input_seq)[0][0]
        next_price = scaler_y.inverse_transform([[next_scaled]])[0][0]
        preds.append(next_price)

        # Update sequence (assume features don't change)
        next_features = last_seq[-1].copy()
        close_index = df_features.columns.get_loc('Close') if 'Close' in df_features.columns else -1
        if close_index != -1:
            next_features[close_index] = next_price
        last_seq = np.vstack([last_seq[1:], next_features])
    return preds

# --- Streamlit UI ---
st.set_page_config(page_title="Bitcoin Price Forecast", layout="wide")
st.title("📈 Bitcoin 5-Day Price Forecast with LSTM")

model_choice = st.selectbox("Select Lookback Model:", ["1-day", "5-day", "7-day"])
lookback = int(model_choice.split("-")[0])

# Load data and preprocess
df = load_btc_data()
X = df.drop(columns=['Close'])
y = df['Close']
dates = df.index

# Load model and scalers
model = load_model(f"saved_models/lstm_{lookback}day.keras")
scaler_X = joblib.load(f"saved_models/scaler_X_{lookback}day.pkl")
scaler_y = joblib.load(f"saved_models/scaler_y_{lookback}day.pkl")

# Create sequences
X_seq, y_seq = [], []
for i in range(lookback, len(df)):
    X_seq.append(X.iloc[i - lookback:i].values)
    y_seq.append(y.iloc[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq).reshape(-1, 1)

# 🛠️ DEBUG: Check feature mismatch
print("🔍 Current X shape:", X.shape)
print("🔍 Scaler expects this many features:", scaler_X.n_features_in_)
print("🔍 Current X columns:", list(X.columns))

# Scale
X_seq_scaled = scaler_X.transform(X_seq.reshape(-1, X.shape[1])).reshape(X_seq.shape)
y_seq_scaled = scaler_y.transform(y_seq)

# Predict last 8% for r2 evaluation
split_index = int(len(X_seq_scaled) * 0.92)
X_test_scaled = X_seq_scaled[split_index:]
y_test_scaled = y_seq_scaled[split_index:]
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test_scaled)

# Calculate R²
from sklearn.metrics import r2_score
r2 = r2_score(y_actual, y_pred)
st.write(f"**Model R² Score:** `{r2:.4f}`")

# --- Forecast next 5 days ---
preds = forecast_next_5_days(model, scaler_X, scaler_y, df[X.columns], lookback)
future_dates = [dates[-1] + pd.Timedelta(days=i + 1) for i in range(5)]

st.subheader("🔮 Next 5-Day Forecast")
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price (USD)': preds})
st.dataframe(forecast_df.set_index('Date').style.format({"Predicted Close Price (USD)": "${:,.2f}"}))

# --- Optional Chart ---
st.subheader("📊 Actual vs Predicted (Test Data)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_actual[-100:], label='Actual')
ax.plot(y_pred[-100:], label='Predicted')
ax.legend()
st.pyplot(fig)
