import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from features import FEATURE_COLS

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d1525",
    font=dict(family="Sora, sans-serif", color="#8b9ab5"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=20, r=20, t=40, b=20),
)

def plot_candlestick(df, ticker):
    recent = df.tail(120)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25], vertical_spacing=0.03)

    fig.add_trace(go.Candlestick(
        x=recent["date"], open=recent["open"], high=recent["high"],
        low=recent["low"], close=recent["close"],
        increasing_fillcolor="#34d399", increasing_line_color="#34d399",
        decreasing_fillcolor="#f87171", decreasing_line_color="#f87171",
        name="Price", line=dict(width=1)
    ), row=1, col=1)

    for col, color, label in [("sma_20","#38bdf8","SMA 20"), ("sma_50","#fb923c","SMA 50")]:
        if col in recent.columns:
            fig.add_trace(go.Scatter(
                x=recent["date"], y=recent[col], name=label,
                line=dict(color=color, width=1.5, dash="dot"), opacity=0.85
            ), row=1, col=1)

    colors_vol = ["#34d399" if c >= o else "#f87171"
                  for c, o in zip(recent["close"], recent["open"])]
    fig.add_trace(go.Bar(
        x=recent["date"], y=recent["volume"], name="Volume",
        marker_color=colors_vol, opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — Price Chart (Last 120 days)", font=dict(size=14, color="#e8edf8")),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        xaxis_rangeslider_visible=False,
        height=480,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, title_font_color="#6b7a99")
    fig.update_yaxes(title_text="Volume",   row=2, col=1, title_font_color="#6b7a99")
    return fig


def plot_rsi(df):
    recent = df.tail(120)
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,113,113,0.07)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(52,211,153,0.07)", line_width=0)
    fig.add_hline(y=70, line_dash="dash", line_color="#f87171", opacity=0.5, line_width=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#34d399", opacity=0.5, line_width=1)
    fig.add_trace(go.Scatter(
        x=recent["date"], y=recent["rsi"], name="RSI",
        line=dict(color="#a78bfa", width=2),
        fill="tozeroy", fillcolor="rgba(167,139,250,0.06)"
    ))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text="RSI (14)", font=dict(size=13, color="#e8edf8")),
        height=220, showlegend=False,
    )
    fig.update_yaxes(range=[0,100], gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_macd(df):
    recent = df.tail(120)
    colors = ["#34d399" if v >= 0 else "#f87171" for v in recent["macd_hist"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=recent["date"], y=recent["macd_hist"], marker_color=colors, name="Histogram", opacity=0.7))
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["macd"], name="MACD", line=dict(color="#38bdf8", width=1.5)))
    fig.add_trace(go.Scatter(x=recent["date"], y=recent["macd_signal"], name="Signal", line=dict(color="#fb923c", width=1.5, dash="dot")))
    
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text="MACD", font=dict(size=13, color="#e8edf8")),
        height=220, barmode="relative",
        legend=dict(orientation="h", y=1.1, font=dict(size=10)),
    )
    return fig


def plot_metrics_bar(results_df):
    metrics  = ["Accuracy","Precision","Recall","F1 Score","ROC-AUC"]
    colors   = ["#38bdf8","#a78bfa","#34d399","#fb923c","#f472b6"]
    models   = results_df["Model"].tolist()

    fig = go.Figure()
    for metric, color in zip(metrics, colors):
        vals = results_df[metric].values
        fig.add_trace(go.Bar(
            name=metric, x=models, y=vals,
            marker_color=color, opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(size=10, color="#8b9ab5"),
        ))

    fig.update_layout(
        PLOTLY_LAYOUT,
        barmode="group",
        title=dict(text="All Models — Metrics Comparison", font=dict(size=14, color="#e8edf8")),
        legend=dict(orientation="h", y=1.05, font=dict(size=11)),
        height=420,
    )
    fig.update_yaxes(range=[0, 1.15], gridcolor="rgba(255,255,255,0.05)")
    fig.update_xaxes(tickfont=dict(size=11), gridcolor="rgba(255,255,255,0.03)")
    return fig


def plot_f1_ranking(results_df):
    df_sorted = results_df.sort_values("F1 Score")
    best_f1   = df_sorted["F1 Score"].max()
    colors = ["#38bdf8" if v == best_f1 else "#1e2d47" for v in df_sorted["F1 Score"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["F1 Score"], y=df_sorted["Model"],
        orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for v in df_sorted["F1 Score"]],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text="F1 Score Ranking", font=dict(size=14, color="#e8edf8")),
        height=320,
    )
    fig.update_xaxes(range=[0, 1.1], gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    labels = ["DOWN", "UP"]
    z_text = [[str(v) for v in row] for row in cm]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=z_text, texttemplate="%{text}",
        colorscale=[[0,"#0d1525"],[0.5,"#1a3a6b"],[1,"#38bdf8"]],
        showscale=False, textfont=dict(size=18, color="white"),
    ))
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text=f"{model_name}", font=dict(size=12, color="#e8edf8")),
        height=260,
    )
    fig.update_xaxes(title_text="Predicted", gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title_text="Actual", gridcolor="rgba(0,0,0,0)")
    return fig


def plot_feature_importance(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X = df[FEATURE_COLS].values
    y = df["target"].values
    rf = RandomForestClassifier(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(StandardScaler().fit_transform(X), y)

    feat_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": rf.feature_importances_})
    feat_df = feat_df.sort_values("Importance").tail(20)
    best_val = feat_df["Importance"].max()
    colors   = ["#38bdf8" if v >= best_val*0.7 else "#1e3a55" for v in feat_df["Importance"]]

    fig = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Feature"],
        orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for v in feat_df["Importance"]],
        textposition="outside", textfont=dict(size=9),
    ))
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text="Top 20 Feature Importances (Random Forest)", font=dict(size=14, color="#e8edf8")),
        height=520,
    )
    fig.update_xaxes(range=[0, feat_df["Importance"].max()*1.3], gridcolor="rgba(255,255,255,0.05)")
    return fig


def plot_price_history(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["close"], name="Close Price",
        line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.05)"
    ))
    fig.update_layout(
        PLOTLY_LAYOUT,
        title=dict(text=f"{ticker} — Full Price History", font=dict(size=14, color="#e8edf8")),
        height=300, showlegend=False,
    )
    return fig