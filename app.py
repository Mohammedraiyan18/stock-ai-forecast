import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from model import get_stock_data, train_model, predict_next

# ⚙️ CONFIG
st.set_page_config(page_title="AI Stock App", layout="wide")

# 🎨 UI STYLE
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #0b0f2a, #120b2e, #1a1440);
    color: white;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Sidebar text WHITE */
section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Input box */
.stTextInput input {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
    padding: 10px;
}

/* Label */
label {
    color: white !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #ff004c, #6c2bd9);
    color: white;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
}

/* Card */
.card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
}
            [data-testid="stMetricValue"] {
    color: white !important;
}

[data-testid="stMetricLabel"] {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# 🧭 SIDEBAR
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Menu",
    ["Dashboard", "About"],
    label_visibility="collapsed"
)

# ---------------- DASHBOARD ----------------
if page == "Dashboard":

    st.markdown(
        "<h1 style='text-align:center;'>AI Stock Forecast App</h1>",
        unsafe_allow_html=True
    )

    popular_tickers = [
        "AAPL",
        "TSLA",
        "MSFT",
        "GOOGL",
        "AMZN",
        "INFY.NS",
        "RELIANCE.NS"
    ]

    ticker = st.selectbox(
        "Choose Stock Symbol",
        popular_tickers,
        index=0
    )
    period = st.selectbox(
        "Select Time Range",
        [
            "1 Month",
            "3 Months",
            "6 Months",
            "1 Year",
            "3 Years",
            "5 Years",
            "All Data"
        ],
        index=6
    )

    if st.button("Generate Analysis"):

        data = get_stock_data(ticker)

        if data is None or data.empty:
            st.error("No data found.")
            st.stop()

        data.index = pd.to_datetime(data.index)
        if period == "1 Month":
            recent_data = data.tail(22).copy()
        elif period == "3 Months":
            recent_data = data.tail(66).copy()
        elif period == "6 Months":
            recent_data = data.tail(132).copy()
        elif period == "1 Year":
            recent_data = data.tail(252).copy()
        elif period == "3 Years":
            recent_data = data.tail(756).copy()
        elif period == "5 Years":
            recent_data = data.tail(1260).copy()
        else:
            recent_data = data.copy()

        close_series = pd.to_numeric(
            recent_data["Close"],
            errors="coerce"
        )

        recent_data["MA20"] = close_series.rolling(20).mean()
        recent_data["MA50"] = close_series.rolling(50).mean()

        model = train_model(data)

        last_price = float(close_series.iloc[-1])

        prediction = predict_next(model, last_price)

        change = prediction - last_price
        change_percent = (change / last_price) * 100

        # 📦 Cards
        col1, col2, col3 = st.columns(3)

        col1.markdown(
            f"""
            <div class='card'>
                <h3>Last Price</h3>
                <h2>{last_price:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col2.markdown(
            f"""
            <div class='card'>
                <h3>Predicted Price</h3>
                <h2>{prediction:.2f}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col3.markdown(
            f"""
            <div class='card'>
                <h3>Change %</h3>
                <h2>{change_percent:.2f}%</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.subheader("📊 Market Insights")
        m1, m2, m3 = st.columns(3)
        m1.metric("52 Week High", f"${close_series.max():.2f}")
        m2.metric("52 Week Low", f"${close_series.min():.2f}")
        m3.metric("Average Price", f"${close_series.mean():.2f}")

        # 📈 Graph
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=close_series,
                mode="lines",
                name="Stock Price",
                line=dict(
                    color="#00E5FF",
                    width=3
                ),
                fill="tozeroy",
                fillcolor="rgba(0,229,255,0.12)"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data["MA20"],
                mode="lines",
                name="20-Day MA",
                line=dict(
                    color="#FF4CFF",
                    width=2
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data["MA50"],
                mode="lines",
                name="50-Day MA",
                line=dict(
                    color="#FFD700",
                    width=2
                )
            )
        )

        fig.update_layout(
            title=dict(
                text=f"Stock Price Trend - {period}",
                font=dict(
                    color="white",
                    size=24
                )
            ),         
            xaxis_title="Date",
            yaxis_title="Closing Price",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            height=550,
            hovermode="x unified",
            legend=dict(
                font=dict(color="white"),
                bgcolor="rgba(0,0,0,0.3)"
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)",
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.2)",
                autorange=True
            )
        )

        st.plotly_chart(fig, width="stretch")
        st.subheader("📈 Trend Analysis")
        if recent_data["MA20"].iloc[-1] > recent_data["MA50"].iloc[-1]:
            st.success("🟢 Bullish Trend Detected")
        else:
            st.error("🔴 Bearish Trend Detected")
        st.subheader("🤖 AI Prediction Summary")
        st.info(
              f"""
        Current Price: ${last_price:.2f}
        Predicted Price: ${prediction:.2f}
        Expected Movement: {change_percent:.2f}%
        Stock Symbol: {ticker}
        """
        )
        with st.expander("📋 View Recent Market Data"):
            st.dataframe(data.tail(15))
# ---------------- ABOUT ----------------
else:

    st.title("About")

    st.markdown("""
### 📈 Project Overview

AI Stock Forecast App is a machine learning dashboard that analyzes stock market trends and predicts future prices using historical data.

### 🚀 Features

✅ Real-Time Stock Data

✅ AI Price Forecasting

✅ Interactive Charts

✅ Moving Average Analysis

✅ Trend Detection

### 🧠 Machine Learning

Linear Regression Model

Features Used:

• Close Price

• Lag 1

• Lag 2

### 🛠 Technologies

• Python

• Streamlit

• Plotly

• Pandas

• NumPy

• Scikit-Learn

• Yahoo Finance

### 🎯 Use Cases

• Stock Analysis

• Investment Research

• Financial Visualization

• Market Monitoring

### 🔮 Future Improvements

• LSTM Models

• News Sentiment Analysis

• Portfolio Optimization

• Crypto Forecasting
""")