import streamlit as st
import pandas as pd
import numpy as np

FEATURE_COLS = [
    "sma_5","sma_10","sma_20","sma_50",
    "ema_5","ema_10","ema_20",
    "macd","macd_signal","macd_hist",
    "rsi","roc_5","roc_10","roc_20",
    "bb_width","bb_pos","atr_pct",
    "vol_ratio","obv_signal",
    "daily_ret","log_ret","range_pct","gap","body",
    "price_sma10_ratio","price_sma20_ratio","price_sma50_ratio",
    "lag_ret_1","lag_ret_2","lag_ret_3","lag_ret_5","lag_ret_10",
    "ret_vol_5","ret_vol_20",
]

@st.cache_data(show_spinner=False)
def build_features(df):
    d = df.copy()
    for w in [5,10,20,50]:
        d[f"sma_{w}"] = d["close"].rolling(w).mean()
        d[f"price_sma{w}_ratio"] = d["close"] / d[f"sma_{w}"]
    for w in [5,10,20]:
        d[f"ema_{w}"] = d["close"].ewm(span=w, adjust=False).mean()

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    delta = d["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - (100/(1 + gain/(loss+1e-10)))

    for w in [5,10,20]:
        d[f"roc_{w}"] = d["close"].pct_change(w)*100

    sma20 = d["close"].rolling(20).mean()
    std20 = d["close"].rolling(20).std()
    d["bb_upper"] = sma20 + 2*std20
    d["bb_lower"] = sma20 - 2*std20
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"])/(sma20+1e-10)
    d["bb_pos"]   = (d["close"] - d["bb_lower"])/(d["bb_upper"]-d["bb_lower"]+1e-10)

    hl = d["high"] - d["low"]
    hc = (d["high"] - d["close"].shift(1)).abs()
    lc = (d["low"]  - d["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    d["atr"]     = tr.rolling(14).mean()
    d["atr_pct"] = d["atr"]/(d["close"]+1e-10)

    vol_ma = d["volume"].rolling(20).mean()
    d["vol_ratio"]  = d["volume"]/(vol_ma+1e-10)
    d["obv"]        = (np.sign(d["close"].diff())*d["volume"]).fillna(0).cumsum()
    d["obv_signal"] = d["obv"].ewm(span=10, adjust=False).mean()

    d["daily_ret"] = d["close"].pct_change()
    d["log_ret"]   = np.log(d["close"]/d["close"].shift(1))
    d["range_pct"] = (d["high"]-d["low"])/(d["close"]+1e-10)
    d["gap"]       = (d["open"]-d["close"].shift(1))/(d["close"].shift(1)+1e-10)
    d["body"]      = (d["close"]-d["open"]).abs()/(d["close"]+1e-10)

    for lag in [1,2,3,5,10]:
        d[f"lag_ret_{lag}"] = d["daily_ret"].shift(lag)
    d["ret_vol_5"]  = d["daily_ret"].rolling(5).std()
    d["ret_vol_20"] = d["daily_ret"].rolling(20).std()

    d["target"] = (d["close"].shift(-1) > d["close"]).astype(int)
    d.dropna(inplace=True)
    d.reset_index(drop=True, inplace=True)
    return d