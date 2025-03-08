import os
import mysql.connector
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for matplotlib
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from datetime import datetime, timedelta
import yfinance as yf
import tensorflow as tf
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model

# ========== Configure Your Database ==========
db_config = {
    'host': '192.168.0.17',
    'user': 'carterdossias',
    'password': 'dossias1',
    'database': 'stocks'
}

app = Flask(__name__)

def load_model_and_objects(ticker):
    """
    Loads the LSTM model, scaler, and look_back parameter from the models/ directory.
    Raises an exception if files are not found.
    """
    model_path = f"models/{ticker}_lstm_model.h5"
    scaler_path = f"models/{ticker}_scaler.pkl"
    look_back_path = f"models/{ticker}_look_back.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No file or directory found at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No file or directory found at {scaler_path}")
    if not os.path.exists(look_back_path):
        raise FileNotFoundError(f"No file or directory found at {look_back_path}")

    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(look_back_path, 'rb') as f:
        look_back = pickle.load(f)
    
    return model, scaler, look_back

def fetch_historical_data(ticker):
    """
    Fetch all available historical data (date, close) from MySQL for the given ticker.
    Returns a DataFrame with ascending date order, columns: ['date', 'close'].
    """
    query = f"""
        SELECT Date_ AS date, Close_ AS close
        FROM {ticker}_stock_data
        ORDER BY Date_ ASC;
    """
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def iterative_forecast(model, scaler, data, look_back, steps_ahead):
    """
    Predict the closing price 'steps_ahead' days after the last known date
    using iterative, one-day-ahead forecasting.
    """
    # Convert last 'look_back' values to a flat list of floats
    forecast_sequence = data[-look_back:, 0].tolist()
    
    for _ in range(steps_ahead):
        X = np.array(forecast_sequence[-look_back:]).reshape(1, look_back, 1)
        scaled_pred = model.predict(X)
        # Append the predicted (scaled) value
        forecast_sequence.append(scaled_pred[0, 0])
    
    # The final appended value is the scaled prediction for 'steps_ahead' days out
    predicted_scaled = forecast_sequence[-1]
    predicted_price = scaler.inverse_transform([[predicted_scaled]])[0, 0]
    return float(predicted_price)

def get_actual_price_yfinance(ticker, target_date):
    """
    Use yfinance to get the actual closing price for 'ticker' on 'target_date'.
    Returns None if no data is found or if date is invalid (weekend/holiday).
    """
    target_str = target_date.strftime('%Y-%m-%d')
    next_day_str = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Use yfinance's Ticker object for more reliable historical data retrieval
    ticker_obj = yf.Ticker(ticker)
    data = ticker_obj.history(start=target_str, end=next_day_str)
    if not data.empty and 'Close' in data.columns:
        price = float(data['Close'].iloc[0])
        print(f"DEBUG: Actual price for {ticker} on {target_str} is {price}")
        return price
    else:
        print(f"DEBUG: No actual price data found for {ticker} on {target_str}")
        return None

def create_plot(historical_df, target_date, predicted_price=None, actual_price=None):
    """
    Creates a Matplotlib figure showing historical data, and if available,
    the predicted point and the actual price point. Returns a base64-encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot historical data
    ax.plot(historical_df['date'], historical_df['close'], label="Historical Close", marker='o')
    
    # Draw a vertical line at the last known date
    last_known_date = historical_df['date'].iloc[-1]
    ax.axvline(last_known_date, color='gray', linestyle='--', label="Last Known Data")
    
    # Plot predicted point if provided
    if predicted_price is not None:
        ax.plot(target_date, predicted_price, 'ro', label="Predicted Close")
        ax.text(target_date, predicted_price, f' {predicted_price:.2f}', color='red')
    
    # Plot actual price if provided
    if actual_price is not None:
        ax.plot(target_date, actual_price, 'go', label="Actual Close")
        ax.text(target_date, actual_price, f' {actual_price:.2f}', color='green')
    
    ax.set_title("Stock Price Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    
    # Convert to PNG and base64-encode
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return encoded

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker', '').strip().upper()
        date_str = request.form.get('date', '').strip()
        
        # Basic input validation
        if not ticker or not date_str:
            return render_template('index.html', error="Please enter both ticker and date.")
        
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            return render_template('index.html', error="Invalid date format. Use YYYY-MM-DD.")
        
        # Load model and associated objects
        try:
            model, scaler, look_back = load_model_and_objects(ticker)
        except Exception as e:
            return render_template('index.html', error=f"Could not load model for {ticker}: {e}")
        
        # Fetch full historical data from MySQL
        historical_df = fetch_historical_data(ticker)
        if historical_df.empty:
            return render_template('index.html', error=f"No historical data found for {ticker}.")
        
        # Determine how to run the forecast based on the target date
        today = datetime.today()
        if target_date > historical_df['date'].iloc[-1]:
            # Future date: use full historical data
            last_date = historical_df['date'].iloc[-1]
            steps_ahead = (target_date.date() - last_date.date()).days
            if steps_ahead < 1:
                steps_ahead = 1  # Safety, though it should be > 0 here
            data_for_forecast = scaler.transform(historical_df[['close']].values)
        else:
            # Past date (or on the last known date):
            # Use only data prior to the target date for backtesting.
            subset_df = historical_df[historical_df['date'] < target_date]
            if subset_df.empty:
                return render_template('index.html', error=f"No historical data available before {target_date.strftime('%Y-%m-%d')}.")
            last_date = subset_df['date'].iloc[-1]
            steps_ahead = (target_date.date() - last_date.date()).days
            if steps_ahead < 1:
                steps_ahead = 1
            data_for_forecast = scaler.transform(subset_df[['close']].values)
        
        # Run the iterative forecast using the appropriate data subset and steps_ahead
        predicted_price = iterative_forecast(model, scaler, data_for_forecast, look_back, steps_ahead)
        
        # If the target date is before today, attempt to fetch the actual closing price
        actual_price = None
        actual_msg = None
        if target_date.date() < today.date():
            actual_price = get_actual_price_yfinance(ticker, target_date)
            if actual_price is None:
                actual_msg = "There is no closing price data for the specified day (possible weekend or holiday)."
        
        # Generate the plot. For visualization, we use the full historical data.
        plot_png = create_plot(historical_df, target_date, predicted_price, actual_price)
        
        return render_template(
            'index.html',
            ticker=ticker,
            date_str=date_str,
            predicted_price=predicted_price,
            actual_price=actual_price,
            actual_msg=actual_msg,
            plot_png=plot_png
        )
    
    # GET request: just display the form
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=7979)