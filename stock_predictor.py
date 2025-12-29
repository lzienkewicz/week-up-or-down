import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Alpaca API credentials
API_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"
SECRET_KEY = "XXXXXXXXXXXXXXXXXXXXXXXX"

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp1 = data['close'].ewm(span=fast, adjust=False).mean()
    exp2 = data['close'].ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_rsi(data, periods=14):
    """Calculate RSI indicator"""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_moving_averages(data):
    """Calculate various moving averages"""
    ma_periods = [5, 10, 20, 50]
    mas = {}
    for period in ma_periods:
        mas[f'ma_{period}'] = data['close'].rolling(window=period).mean()
    return mas

def fetch_stock_data(ticker, api_key, secret_key):
    """Fetch stock data from Alpaca API"""
    print(f"Fetching data for {ticker}...")

    # Try without authentication (rate-limited but should work for testing)
    # According to Alpaca docs, API keys are optional for historical data
    client = StockHistoricalDataClient()

    # Fetch 3 years of weekly data to have enough for training
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3*365)

    request_params = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Week,
        start=start_date,
        end=end_date,
        adjustment='split'  # Adjust for stock splits
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df

    # If multi-index, get data for the specific ticker
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level='symbol')

    # Reset index to make timestamp a column
    df = df.reset_index()

    return df

def prepare_features(df):
    """Calculate all technical indicators and prepare features"""
    print("Calculating technical indicators...")

    # Calculate MACD
    df['macd'], df['macd_signal'], df['macd_histogram'] = calculate_macd(df)

    # Calculate RSI
    df['rsi'] = calculate_rsi(df)

    # Calculate Moving Averages
    mas = calculate_moving_averages(df)
    for ma_name, ma_values in mas.items():
        df[ma_name] = ma_values

    # Calculate volume moving average
    df['volume_ma_5'] = df['volume'].rolling(window=5).mean()

    # Drop rows with NaN values (from indicator calculations)
    df = df.dropna()

    # Create target variable: 1 if next week's close > this week's close, 0 otherwise
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Drop the last row (no target for it)
    df = df[:-1]

    return df

def normalize_data(df, feature_columns):
    """Normalize features using MinMaxScaler"""
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df_normalized, scaler

def plot_data(df, ticker):
    """Plot normalized stock data with technical indicators"""
    print("Generating visualizations...")

    # Create subplots
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'{ticker} - Normalized Close Price with Moving Averages',
            'Normalized Volume',
            'MACD',
            'RSI',
            'Prediction Target (Price Direction)'
        ),
        row_heights=[0.25, 0.15, 0.2, 0.2, 0.2]
    )

    # Plot 1: Close price with moving averages
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'],
                             name='Close', line=dict(color='blue', width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_5'],
                             name='MA-5', line=dict(color='orange', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_10'],
                             name='MA-10', line=dict(color='red', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_20'],
                             name='MA-20', line=dict(color='green', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['ma_50'],
                             name='MA-50', line=dict(color='purple', width=1)),
                  row=1, col=1)

    # Plot 2: Volume
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['volume'],
                         name='Volume', marker_color='lightblue'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['volume_ma_5'],
                             name='Volume MA-5', line=dict(color='darkblue', width=2)),
                  row=2, col=1)

    # Plot 3: MACD
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd'],
                             name='MACD', line=dict(color='blue', width=2)),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['macd_signal'],
                             name='Signal', line=dict(color='red', width=1)),
                  row=3, col=1)
    fig.add_trace(go.Bar(x=df['timestamp'], y=df['macd_histogram'],
                         name='Histogram', marker_color='gray'),
                  row=3, col=1)

    # Plot 4: RSI
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['rsi'],
                             name='RSI', line=dict(color='purple', width=2)),
                  row=4, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)

    # Plot 5: Target (price direction)
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['target'],
                             name='Target (1=Up, 0=Down)',
                             mode='markers+lines',
                             marker=dict(color=df['target'], colorscale='RdYlGn', size=8)),
                  row=5, col=1)

    # Update layout
    fig.update_layout(
        height=1400,
        showlegend=True,
        title_text=f"{ticker} Stock Analysis - Normalized Data for ML Model",
        hovermode='x unified'
    )

    fig.update_xaxes(title_text="Date", row=5, col=1)
    fig.update_yaxes(title_text="Normalized Price", row=1, col=1)
    fig.update_yaxes(title_text="Normalized Volume", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="RSI", row=4, col=1)
    fig.update_yaxes(title_text="Direction", row=5, col=1)

    fig.show()

def create_sequences(data, target, seq_length=10):
    """Create sequences for LSTM model"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Build LSTM model for binary classification"""
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification (up or down)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    # Get ticker from user
    ticker = input("Enter ticker symbol (e.g., AAPL, TSLA, MSFT): ").upper()

    # Fetch data
    try:
        df = fetch_stock_data(ticker, API_KEY, SECRET_KEY)
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Please check your API credentials and ticker symbol.")
        return

    if len(df) < 100:
        print(f"Insufficient data for {ticker}. Need at least 100 weekly bars.")
        return

    print(f"Fetched {len(df)} weeks of data for {ticker}")

    # Prepare features
    df = prepare_features(df)

    # Define feature columns for modeling
    feature_columns = [
        'close', 'volume', 'open', 'high', 'low',
        'macd', 'macd_signal', 'macd_histogram',
        'rsi', 'ma_5', 'ma_10', 'ma_20', 'ma_50',
        'volume_ma_5'
    ]

    # Normalize data
    df_normalized, scaler = normalize_data(df, feature_columns)

    # Plot the normalized data
    plot_data(df_normalized, ticker)

    # Prepare data for LSTM
    print("\nPreparing data for LSTM model...")
    sequence_length = 10  # Use 10 weeks of data to predict next week

    X_data = df_normalized[feature_columns].values
    y_data = df_normalized['target'].values

    X, y = create_sequences(X_data, y_data, sequence_length)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Don't shuffle time series
    )

    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Build and train model
    print("\nBuilding LSTM model...")
    model = build_lstm_model((sequence_length, len(feature_columns)))

    print(model.summary())

    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Make prediction for the most recent data
    latest_sequence = X[-1].reshape(1, sequence_length, len(feature_columns))
    prediction_prob = model.predict(latest_sequence, verbose=0)[0][0]
    prediction_direction = "UP" if prediction_prob >= 0.5 else "DOWN"

    # Display results
    print("\n" + "="*60)
    print("STOCK PREDICTION RESULTS")
    print("="*60)
    print(f"Ticker: {ticker}")
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    print(f"Testing Accuracy: {test_accuracy*100:.2f}%")
    print(f"Probability of Price Increasing: {prediction_prob*100:.2f}%")
    print(f"Predicted Direction: {prediction_direction} (UP ≥ 50%, DOWN < 50%)")
    print("="*60)

    # Additional context
    current_price = df['close'].iloc[-1]
    print(f"\nCurrent week closing price: ${current_price:.2f}")
    print(f"Prediction: Next week's close will be {prediction_direction}")

if __name__ == "__main__":
    main()
