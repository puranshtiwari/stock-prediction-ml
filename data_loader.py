import streamlit as st
import pandas as pd
import numpy as np

def generate_synthetic():
    np.random.seed(42)
    n = 1260
    close = 150 + np.cumsum(np.random.randn(n) * 1.5)
    close = np.abs(close) + 50
    df = pd.DataFrame({
        "date"  : pd.date_range("2019-01-01", periods=n, freq="B"),
        "open"  : close * (1 + np.random.randn(n)*0.003),
        "high"  : close * (1 + np.abs(np.random.randn(n))*0.008),
        "low"   : close * (1 - np.abs(np.random.randn(n))*0.008),
        "close" : close,
        "volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    })
    return df

@st.cache_data(show_spinner=False)
def fetch_data(ticker, period):
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")
        if df.empty:
            return None, "No data found. Check ticker symbol."
        df.reset_index(inplace=True)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df[["date", "open", "high", "low", "close", "volume"]]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df, None
    except Exception as e:
        return generate_synthetic(), None