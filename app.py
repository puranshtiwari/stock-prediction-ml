import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np

# ── Module Imports ────────────────────────────────────────────
from data_loader import fetch_data
from features import build_features, FEATURE_COLS
from models import train_models
from plots import (
    plot_candlestick, plot_rsi, plot_macd, plot_metrics_bar,
    plot_f1_ranking, plot_confusion_matrix, plot_feature_importance, plot_price_history
)

# ── Page config (MUST be first Streamlit call) ────────────────
st.set_page_config(
    page_title="StockML — Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Sora', sans-serif !important; }
.stApp { background: #080e1a; }
.block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #0d1525 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
[data-testid="stSidebar"] .stMarkdown p { color: #8b9ab5; font-size: 13px; }

/* ── Metric cards ── */
[data-testid="stMetric"] { background: #111a2e; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1rem 1.25rem !important; }
[data-testid="stMetricLabel"] { font-size: 11px !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7a99 !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 700 !important; color: #e8edf8 !important; }
[data-testid="stMetricDelta"] svg { display: none; }
[data-testid="stMetricDelta"] { font-size: 12px !important; }

/* ── Buttons ── */
.stButton > button { background: linear-gradient(135deg, #1a6fd4 0%, #0d4fa3 100%) !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'Sora', sans-serif !important; font-weight: 600 !important; font-size: 14px !important; padding: 0.6rem 1.5rem !important; transition: all 0.2s !important; letter-spacing: 0.02em; }
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 8px 24px rgba(26,111,212,0.35) !important; }

/* ── Selectbox / Inputs ── */
.stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput > div > div > input { background: #111a2e !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 8px !important; color: #e8edf8 !important; font-family: 'Sora', sans-serif !important; }
.stSelectbox label, .stTextInput label, .stSlider label, .stNumberInput label { color: #8b9ab5 !important; font-size: 13px !important; font-weight: 500 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0d1525; border-radius: 12px; padding: 4px; gap: 4px; border: 1px solid rgba(255,255,255,0.06); }
.stTabs [data-baseweb="tab"] { border-radius: 9px !important; color: #6b7a99 !important; font-family: 'Sora', sans-serif !important; font-size: 13px !important; font-weight: 500 !important; padding: 0.5rem 1.25rem !important; }
.stTabs [aria-selected="true"] { background: #1a2a45 !important; color: #38bdf8 !important; }

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { margin-top: 0.5rem; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #38bdf8 !important; }

/* ── Dataframe ── */
.stDataFrame { border: 1px solid rgba(255,255,255,0.07) !important; border-radius: 10px !important; overflow: hidden; }

/* ── Section headers ── */
.section-title { font-size: 22px; font-weight: 700; color: #e8edf8; margin: 0 0 4px 0; letter-spacing: -0.02em; }
.section-sub { font-size: 13px; color: #6b7a99; margin-bottom: 1.5rem; }
.pill { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; padding: 3px 10px; border-radius: 20px; margin-right: 6px; }
.pill-blue   { background: rgba(56,189,248,0.12); color: #38bdf8; }
.pill-green  { background: rgba(52,211,153,0.12); color: #34d399; }
.pill-orange { background: rgba(251,146,60,0.12); color: #fb923c; }
.pill-purple { background: rgba(167,139,250,0.12); color: #a78bfa; }

/* ── Info card ── */
.info-card { background: #0d1525; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════════════
if "results"   not in st.session_state: st.session_state.results   = None
if "y_true"    not in st.session_state: st.session_state.y_true    = None
if "all_preds" not in st.session_state: st.session_state.all_preds = None
if "df"        not in st.session_state: st.session_state.df        = None
if "ticker"    not in st.session_state: st.session_state.ticker    = "AAPL"


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 0 0 1rem 0;'>
      <div style='font-size:22px; font-weight:700; color:#e8edf8; letter-spacing:-0.02em;'>📈 StockML</div>
      <div style='font-size:12px; color:#6b7a99; margin-top:2px;'>Multi-Model Prediction Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**Stock Configuration**")
    ticker_options = {
        "Apple (AAPL)": "AAPL",
        "Google (GOOGL)": "GOOGL",
        "Microsoft (MSFT)": "MSFT",
        "Tesla (TSLA)": "TSLA",
        "Reliance (RELIANCE.NS)": "RELIANCE.NS",
        "TCS (TCS.NS)": "TCS.NS",
        "Infosys (INFY.NS)": "INFY.NS",
        "Custom Ticker": "__custom__",
    }
    ticker_choice = st.selectbox("Select Stock", list(ticker_options.keys()))
    if ticker_options[ticker_choice] == "__custom__":
        ticker = st.text_input("Enter Ticker Symbol", value="AAPL").upper()
    else:
        ticker = ticker_options[ticker_choice]

    period = st.select_slider(
        "Historical Period",
        options=["1y", "2y", "3y", "5y"],
        value="5y"
    )

    st.divider()
    st.markdown("**Model Configuration**")

    all_model_names = ["Logistic Regression","K-Nearest Neighbors","Random Forest",
                       "Gradient Boosting","AdaBoost","XGBoost","SVM (RBF)"]
    selected_models = st.multiselect(
        "Models to Train",
        all_model_names,
        default=["Random Forest", "XGBoost", "Gradient Boosting", "Logistic Regression"]
    )

    n_splits = st.slider("TimeSeriesSplit Folds", 3, 10, 5)
    n_est    = st.slider("n_estimators (tree models)", 50, 300, 100, step=50)

    st.divider()

    run_btn = st.button("🚀  Run Full Pipeline", use_container_width=True)

    st.divider()
    st.markdown("""
    <div style='font-size:11px; color:#3d4f6b; line-height:1.6;'>
    <b style='color:#4a6a9b'>How to use:</b><br>
    1. Select a stock ticker<br>
    2. Choose time period<br>
    3. Pick models to compare<br>
    4. Click Run Pipeline<br><br>
    Data via Yahoo Finance (yfinance)
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 1.5rem;'>
  <div style='font-size:30px; font-weight:700; color:#e8edf8; letter-spacing:-0.03em; line-height:1.2;'>
    Stock Direction Prediction
  </div>
  <div style='font-size:14px; color:#6b7a99; margin-top:6px;'>
    <span class='pill pill-blue'>7 Models</span>
    <span class='pill pill-green'>34 Features</span>
    <span class='pill pill-orange'>TimeSeriesSplit CV</span>
    <span class='pill pill-purple'>Binary Classification</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Run Pipeline ─────────────────────────────────────
if run_btn:
    if not selected_models:
        st.error("Please select at least one model.")
    else:
        with st.spinner(f"Fetching {ticker} data..."):
            df_raw, err = fetch_data(ticker, period)

        if err:
            st.error(err)
        else:
            st.session_state.ticker = ticker
            with st.spinner("Building 34 technical indicators..."):
                df_feat = build_features(df_raw)
            st.session_state.df = df_feat

            results, y_true, all_preds = train_models(df_feat, n_splits, n_est, selected_models)
            st.session_state.results   = results
            st.session_state.y_true    = y_true
            st.session_state.all_preds = all_preds
            st.success(f"✅ Pipeline complete! Best model: **{results.iloc[0]['Model']}** (F1 = {results.iloc[0]['F1 Score']})")


# ═══════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊 Overview  ",
    "  📈 Charts  ",
    "  🏆 Model Results  ",
    "  🔲 Confusion Matrices  ",
    "  🌟 Feature Importance  ",
])

# ─────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────
with tab1:
    if st.session_state.df is None:
        st.markdown("""
        <div style='text-align:center; padding: 4rem 2rem; color: #3d4f6b;'>
          <div style='font-size:48px; margin-bottom:1rem;'>📈</div>
          <div style='font-size:20px; font-weight:600; color:#4a6080; margin-bottom:0.5rem;'>No Data Yet</div>
          <div style='font-size:14px;'>Select a stock and click <b>Run Full Pipeline</b> in the sidebar to get started.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = st.session_state.df
        ticker_name = st.session_state.ticker

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            up_pct = df["target"].mean() * 100
            st.metric("UP Days", f"{up_pct:.1f}%")
        with col3:
            st.metric("DOWN Days", f"{100-up_pct:.1f}%")
        with col4:
            st.metric("Features", f"{len(FEATURE_COLS)}")
        with col5:
            returns = df["daily_ret"].dropna()
            ann_vol = returns.std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{ann_vol:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        st.plotly_chart(plot_price_history(df, ticker_name), use_container_width=True)

        st.markdown('<div class="section-title" style="font-size:16px;">Raw Data Preview</div>', unsafe_allow_html=True)
        preview = df[["date","open","high","low","close","volume","target"]].tail(10).copy()
        preview["target"] = preview["target"].map({1: "⬆ UP", 0: "⬇ DOWN"})
        st.dataframe(preview.style.applymap(
            lambda v: "color:#34d399" if v=="⬆ UP" else ("color:#f87171" if v=="⬇ DOWN" else ""),
            subset=["target"]
        ), use_container_width=True, hide_index=True)

        if st.session_state.results is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:16px; margin-bottom:0.75rem;">🏆 Best Model Summary</div>', unsafe_allow_html=True)
            best = st.session_state.results.iloc[0]
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("Best Model", best["Model"].split()[0])
            b2.metric("Accuracy",  f"{best['Accuracy']*100:.2f}%")
            b3.metric("Precision", f"{best['Precision']*100:.2f}%")
            b4.metric("Recall",    f"{best['Recall']*100:.2f}%")
            b5.metric("F1 Score",  f"{best['F1 Score']*100:.2f}%")


# ─────────────────────────────────────────────────────
#  TAB 2 — CHARTS
# ─────────────────────────────────────────────────────
with tab2:
    if st.session_state.df is None:
        st.info("Run the pipeline first to see charts.")
    else:
        df = st.session_state.df
        ticker_name = st.session_state.ticker

        st.plotly_chart(plot_candlestick(df, ticker_name), use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_rsi(df), use_container_width=True)
        with c2:
            st.plotly_chart(plot_macd(df), use_container_width=True)


# ─────────────────────────────────────────────────────
#  TAB 3 — MODEL RESULTS
# ─────────────────────────────────────────────────────
with tab3:
    if st.session_state.results is None:
        st.info("Run the pipeline first to see model results.")
    else:
        results = st.session_state.results

        st.plotly_chart(plot_metrics_bar(results), use_container_width=True)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_f1_ranking(results), use_container_width=True)
        with c2:
            st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="font-size:16px; margin-bottom:1rem;">Full Metrics Table</div>', unsafe_allow_html=True)
            display_df = results.drop(columns=["_preds"]).copy()
            for col in ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_df, use_container_width=True, hide_index=True)

        csv = results.drop(columns=["_preds"]).to_csv(index=False)
        st.download_button(
            "⬇ Download Metrics as CSV",
            csv,
            file_name=f"{st.session_state.ticker}_metrics.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────
#  TAB 4 — CONFUSION MATRICES
# ─────────────────────────────────────────────────────
with tab4:
    if st.session_state.all_preds is None:
        st.info("Run the pipeline first to see confusion matrices.")
    else:
        y_true    = st.session_state.y_true
        all_preds = st.session_state.all_preds
        models    = list(all_preds.keys())
        n_models  = len(models)

        for row_start in range(0, n_models, 3):
            cols = st.columns(3)
            for i, col in enumerate(cols):
                idx = row_start + i
                if idx < n_models:
                    name = models[idx]
                    with col:
                        st.plotly_chart(
                            plot_confusion_matrix(y_true, all_preds[name], name),
                            use_container_width=True
                        )

        st.markdown("""
        <div class='info-card' style='margin-top:1rem;'>
          <div style='font-size:13px; color:#6b7a99; line-height:1.7;'>
            <b style='color:#8b9ab5'>Reading the matrix:</b>
            Top-left = True Negatives (correctly predicted DOWN) |
            Top-right = False Positives (said UP, was DOWN) |
            Bottom-left = False Negatives (said DOWN, was UP) |
            Bottom-right = True Positives (correctly predicted UP)
          </div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────
#  TAB 5 — FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────
with tab5:
    if st.session_state.df is None:
        st.info("Run the pipeline first to see feature importance.")
    else:
        df = st.session_state.df
        with st.spinner("Computing feature importance from Random Forest..."):
            fig_fi = plot_feature_importance(df)
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("""
        <div class='info-card'>
          <b style='color:#8b9ab5; font-size:13px;'>What does this tell us?</b>
          <div style='font-size:13px; color:#6b7a99; margin-top:4px; line-height:1.7;'>
          Features with high importance scores contributed the most to the Random Forest's predictions.
          If technical indicators like RSI, MACD, and lag returns rank high, it validates that
          momentum signals are genuinely predictive. Low-importance features can potentially be
          removed to simplify the model without hurting performance.
          </div>
        </div>
        """, unsafe_allow_html=True)
