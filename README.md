# Bitcoin Price Prediction App

This is a Streamlit web application that predicts Bitcoin prices using LSTM neural networks with different lookback periods.

## Features

- Real-time Bitcoin price data fetching from Yahoo Finance
- Technical indicator calculation (RSI, MACD, EMAs)
- Price prediction for customizable future periods
- Interactive charts and visualizations
- Trained LSTM models with 1, 5, and 7-day lookback periods

## Live Demo

You can access the live app at: [Your Streamlit app URL after deployment]

## Installation and Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/bitcoin-price-predictor.git
   cd bitcoin-price-predictor
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Model Details

The application uses LSTM (Long Short-Term Memory) neural networks trained on Bitcoin price data with various technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Multiple Exponential Moving Averages
- Volatility measures
- Volume indicators

Three models are provided with different lookback periods (1, 5, and 7 days) to capture short, medium, and longer-term patterns.

## Deployment

This app is deployed using Streamlit Community Cloud, which provides free hosting for Streamlit applications directly from GitHub repositories.

## Directory Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── saved_models/               # Pre-trained LSTM models and scalers
│   ├── lstm_1day.keras         # 1-day lookback model
│   ├── lstm_5day.keras         # 5-day lookback model
│   ├── lstm_7day.keras         # 7-day lookback model
│   ├── scaler_X_1day.pkl       # Feature scalers
│   ├── scaler_y_1day.pkl       # Target scalers
│   └── ...
└── README.md                   # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
