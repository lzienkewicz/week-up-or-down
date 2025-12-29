# Stock Price Prediction with LSTM

A machine learning application that predicts whether a stock's price will go up or down next week using LSTM (Long Short-Term Memory) neural networks and technical indicators.

## Features

- Fetches weekly stock data from Alpaca Markets API
- Automatically adjusts for stock splits
- Calculates technical indicators:
  - MACD (Moving Average Convergence Divergence)
  - RSI (Relative Strength Index)
  - Multiple Moving Averages (5, 10, 20, 50-week)
  - Volume indicators
- Normalizes data for machine learning
- Visualizes all indicators with interactive Plotly charts
- Uses LSTM model to predict next week's price direction
- Provides training/testing accuracy and prediction confidence

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Get your Alpaca API credentials:
   - Sign up at [Alpaca Markets](https://alpaca.markets/)
   - Get your API Key and Secret Key from the dashboard

3. Edit `stock_predictor.py` and replace the placeholder API credentials:
```python
API_KEY = "YOUR_API_KEY_HERE"
SECRET_KEY = "YOUR_SECRET_KEY_HERE"
```

## Usage

Run the script:
```bash
python stock_predictor.py
```

When prompted, enter a ticker symbol (e.g., AAPL, TSLA, MSFT).

The script will:
1. Fetch 3 years of weekly data
2. Calculate technical indicators
3. Display interactive charts with normalized data
4. Train an LSTM model
5. Output prediction results

## Output

The final report includes:
- **Training Accuracy**: Model accuracy on training data
- **Testing Accuracy**: Model accuracy on test data
- **Probability of Price Increasing**: Confidence level (0-100%)
- **Predicted Direction**: UP (≥50%) or DOWN (<50%)

## Example Output

```
============================================================
STOCK PREDICTION RESULTS
============================================================
Ticker: AAPL
Training Accuracy: 65.43%
Testing Accuracy: 58.82%
Probability of Price Increasing: 72.35%
Predicted Direction: UP (UP ≥ 50%, DOWN < 50%)
============================================================

Current week closing price: $185.23
Prediction: Next week's close will be UP
```

## How It Works

1. **Data Collection**: Fetches weekly OHLCV data from Alpaca with split adjustments
2. **Feature Engineering**: Calculates MACD, RSI, moving averages, and volume indicators
3. **Normalization**: Scales all features to 0-1 range using MinMaxScaler
4. **Sequence Creation**: Creates 10-week sequences for time series prediction
5. **LSTM Training**: Trains a multi-layer LSTM network with dropout for regularization
6. **Prediction**: Predicts if next week's close will be higher than current week's close

## Technical Indicators

- **MACD**: Trend-following momentum indicator
- **RSI**: Measures overbought/oversold conditions (0-100)
- **Moving Averages**: 5, 10, 20, and 50-week averages
- **Volume**: Trading volume with 5-week average

## Model Architecture

- LSTM Layer 1: 100 units with return sequences
- Dropout: 20%
- LSTM Layer 2: 50 units
- Dropout: 20%
- Dense Layer: 25 units (ReLU)
- Output Layer: 1 unit (Sigmoid for binary classification)

## Disclaimer

This tool is for educational purposes only. Stock predictions are inherently uncertain and should not be the sole basis for investment decisions. Past performance does not guarantee future results.

## License

MIT License
