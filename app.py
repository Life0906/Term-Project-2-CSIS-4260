import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Bitcoin Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Header
st.title("Bitcoin Price Prediction App")
st.markdown("""
This app uses LSTM neural networks to predict Bitcoin prices based on historical data and technical indicators.
""")

# Sidebar
st.sidebar.header("Model Parameters")
lookback_period = st.sidebar.selectbox("Select Lookback Period (Days)", [1, 5, 7], index=1)
prediction_days = st.sidebar.slider("Number of Days to Predict", 1, 5, 5)

# Functions
@st.cache_data(ttl=3600)
def load_data():
    """Load BTC data from Yahoo Finance and calculate technical indicators"""
    btc = yf.Ticker("BTC-USD")
    df = btc.history(period="10y", interval="1d")
    
    # Drop unnecessary columns
    df = df.drop(columns=['Dividends', 'Stock Splits'])
    return df

def calculate_sma(data, period):
    return data['Close'].rolling(window=period).mean()

def calculate_ema(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_rsi(data, period=14):
    """Calculates the Relative Strength Index."""
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, short_period=12, long_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence."""
    short_ema = calculate_ema(data, short_period)
    long_ema = calculate_ema(data, long_period)
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def prepare_features(df):
    """Prepare features for prediction"""
    # Calculate RSI
    df['RSI'] = calculate_rsi(df)

    # Calculate MACD
    macd, signal, hist = calculate_macd(df)
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    df['MACD_Hist'] = hist

    # Calculate Simple moving averages
    df['SMA_10'] = calculate_sma(df, 10)
    df['SMA_20'] = calculate_sma(df, 20)
    df['SMA_40'] = calculate_sma(df, 40)
    df['SMA_50'] = calculate_sma(df, 50)
    df['SMA_200'] = calculate_sma(df, 200)

    # Calculate Exponential Moving Averages
    df['EMA_8'] = calculate_ema(df, 8)
    df['EMA_21'] = calculate_ema(df, 21)
    df['EMA_55'] = calculate_ema(df, 55)
    df['EMA_200'] = calculate_ema(df, 200)

    # Create crossover features
    df['SMA_10_20_cross'] = (df['SMA_10'] > df['SMA_20']).astype(int)
    df['SMA_10_20_cross_lagged'] = df['SMA_10_20_cross'].shift(1).fillna(0).astype(int)
    df['SMA_10_20_crossover_signal'] = df['SMA_10_20_cross'] - df['SMA_10_20_cross_lagged']

    df['SMA_20_50_cross'] = (df['SMA_20'] > df['SMA_50']).astype(int)
    df['SMA_20_50_cross_lagged'] = df['SMA_20_50_cross'].shift(1).fillna(0).astype(int)
    df['SMA_20_50_crossover_signal'] = df['SMA_20_50_cross'] - df['SMA_20_50_cross_lagged']

    df['SMA_50_200_cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
    df['SMA_50_200_cross_lagged'] = df['SMA_50_200_cross'].shift(1).fillna(0).astype(int)
    df['SMA_50_200_crossover_signal'] = df['SMA_50_200_cross'] - df['SMA_50_200_cross_lagged']

    # Feature Engineering
    # Price Returns
    df['Daily_Return'] = df['Close'].pct_change()

    # Volatility
    df['Volatility_7d'] = df['Daily_Return'].rolling(window=7).std()

    # Relative Volume
    volume_rolling_avg_20d = df['Volume'].rolling(window=20).mean()
    df['Relative_Volume'] = df['Volume'] / volume_rolling_avg_20d

    # Remove columns ending with '_cross' or '_cross_lagged'
    cols_to_remove = [col for col in df.columns if col.endswith('_cross') or col.endswith('_cross_lagged')]
    df = df.drop(columns=cols_to_remove)

    # Removing the SMA columns to avoid co-relation with EMA. EMA is proven to be more effective way to predict prices
    sma_columns_to_drop = ['SMA_10', 'SMA_20', 'SMA_40', 'SMA_50', 'SMA_200']
    df = df.drop(columns=sma_columns_to_drop)

    df = df.dropna()
    
    return df

def load_model_and_scalers(lookback):
    """Load the trained model and scalers"""
    # For the deployed app, we'll need to have these files in the repository
    try:
        model = load_model(f"saved_models/lstm_{lookback}day.keras")
        scaler_X = joblib.load(f"saved_models/scaler_X_{lookback}day.pkl")
        scaler_y = joblib.load(f"saved_models/scaler_y_{lookback}day.pkl")
        return model, scaler_X, scaler_y
    except Exception as e:
        st.error(f"Error loading model and scalers: {e}")
        return None, None, None

def make_predictions(df, model, scaler_X, scaler_y, lookback, days_to_predict):
    """Make predictions for future days"""
    features = df.drop(columns=['Close'])
    
    future_preds = []
    last_seq = features.iloc[-lookback:].values
    
    for _ in range(days_to_predict):
        # Transform and reshape data for prediction
        input_seq = scaler_X.transform(last_seq).reshape(1, lookback, features.shape[1])
        
        # Make prediction
        pred_scaled = model.predict(input_seq)[0][0]
        pred_unscaled = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        future_preds.append(pred_unscaled)
        
        # Simulate future features - assume latest features with updated Close price
        next_features = last_seq[-1].copy()
        close_index = features.columns.get_loc('Close') if 'Close' in features.columns else -1
        if close_index != -1:
            next_features[close_index] = pred_unscaled
        last_seq = np.vstack([last_seq[1:], next_features])
    
    # Generate future dates
    last_date = df.index[-1]
    future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days_to_predict)]
    
    # Create prediction dataframe
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': future_preds
    })
    
    return pred_df

# Main app logic
def main():
    # Load raw data
    with st.spinner('Loading Bitcoin data...'):
        raw_df = load_data()
    
    # Show recent data
    st.subheader("Recent Bitcoin Prices")
    st.dataframe(raw_df.tail().style.format({'Open': '${:.2f}', 'High': '${:.2f}', 
                                             'Low': '${:.2f}', 'Close': '${:.2f}'}))
    
    # Process data
    with st.spinner('Calculating technical indicators...'):
        processed_df = prepare_features(raw_df.copy())
    
    # Load model and scalers
    with st.spinner('Loading prediction model...'):
        model, scaler_X, scaler_y = load_model_and_scalers(lookback_period)
        
        if model is None:
            st.error("Failed to load the model. Please check if model files exist.")
            return
    
    # Make predictions
    with st.spinner('Making predictions...'):
        predictions = make_predictions(
            processed_df, 
            model, 
            scaler_X, 
            scaler_y, 
            lookback_period, 
            prediction_days
        )
    
    # Display predictions
    st.subheader(f"Bitcoin Price Predictions (Next {prediction_days} Days)")
    st.dataframe(predictions.style.format({'Predicted_Price': '${:.2f}'}))
    
    # Plot
    st.subheader("Historical Prices and Predictions")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot historical data (last 60 days)
    hist_data = raw_df.tail(60)
    ax.plot(hist_data.index, hist_data['Close'], label='Historical Price', color='blue')
    
    # Plot predictions
    ax.plot(predictions['Date'], predictions['Predicted_Price'], label='Predicted Price', color='red', linestyle='--')
    
    # Format plot
    ax.set_title('Bitcoin Price: Historical vs Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Technical indicator visualization (optional)
    if st.checkbox("Show Technical Indicators"):
        st.subheader("Technical Indicators")
        
        # Select which indicators to show
        indicators = st.multiselect(
            "Select indicators to display",
            options=["RSI", "MACD", "EMA_8", "EMA_21", "EMA_55", "EMA_200"],
            default=["RSI", "MACD"]
        )
        
        if indicators:
            last_days = st.slider("Number of days to display", 30, 365, 90)
            data_to_plot = processed_df.tail(last_days)
            
            if "RSI" in indicators:
                st.subheader("Relative Strength Index (RSI)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data_to_plot.index, data_to_plot['RSI'], color='purple')
                ax.axhline(y=70, color='r', linestyle='-', alpha=0.3)
                ax.axhline(y=30, color='g', linestyle='-', alpha=0.3)
                ax.set_title('RSI')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            if "MACD" in indicators:
                st.subheader("Moving Average Convergence Divergence (MACD)")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(data_to_plot.index, data_to_plot['MACD'], label='MACD Line', color='blue')
                ax.plot(data_to_plot.index, data_to_plot['MACD_Signal'], label='Signal Line', color='red')
                ax.bar(data_to_plot.index, data_to_plot['MACD_Hist'], label='Histogram', color='green', alpha=0.3)
                ax.set_title('MACD')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Plot selected EMAs
            ema_indicators = [ind for ind in indicators if ind.startswith("EMA")]
            if ema_indicators:
                st.subheader("Exponential Moving Averages")
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot price
                ax.plot(data_to_plot.index, data_to_plot['Close'], label='Price', color='black', alpha=0.5)
                
                # Plot selected EMAs
                colors = ['red', 'blue', 'green', 'purple']
                for i, ema in enumerate(ema_indicators):
                    ax.plot(data_to_plot.index, data_to_plot[ema], label=ema, color=colors[i % len(colors)])
                
                ax.set_title('Price and EMAs')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
