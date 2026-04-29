# 📈 StockML — Multi-Model Prediction Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stock-prediction-ml-puransh.streamlit.app/)

**🚀 Live Demo:** [https://stock-prediction-ml-puransh.streamlit.app/](https://stock-prediction-ml-puransh.streamlit.app/)

## 📌 Overview
StockML is an interactive, machine-learning-powered web dashboard built with Streamlit. It allows users to predict the next-day direction (UP/DOWN) of major stocks using 7 different classification models. The app dynamically fetches historical data, engineers 34 complex technical indicators, and ranks the models based on their F1 Scores using TimeSeriesSplit Cross-Validation.

## ✨ Key Features
* **Automated Data Pipeline:** Fetches real-time historical data via Yahoo Finance (`yfinance`).
* **Robust Feature Engineering:** Automatically computes 34 technical indicators including MACD, RSI, Bollinger Bands, ATR, OBV, and various Moving Averages.
* **Multi-Model Comparison:** Trains and compares 7 different machine learning algorithms side-by-side:
  * Random Forest
  * XGBoost
  * Gradient Boosting
  * Logistic Regression
  * AdaBoost
  * K-Nearest Neighbors
  * SVM (RBF)
* **Interactive Visualizations:** Uses Plotly to generate beautiful, interactive charts including Candlestick overlays, RSI/MACD subplots, Confusion Matrices, and Feature Importance rankings.

## 📂 Project Structure
The code is organized into a clean, modular structure for easy maintenance and scaling:

```text
stock-prediction-ml/
│
├── requirements.txt      # Dependencies needed to run the app
├── data_loader.py        # Handles fetching stock data & generating synthetic fallbacks
├── features.py           # Technical indicator and feature engineering logic
├── models.py             # ML models, training loop, and TimeSeriesSplit CV
├── plots.py              # Plotly chart generation and layout configurations
└── app.py                # Main Streamlit UI, sidebar, and layout orchestration
