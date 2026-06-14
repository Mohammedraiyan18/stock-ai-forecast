import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def get_stock_data(ticker):
    try:
        data = yf.download(
            ticker,
            start="2010-01-01",
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[['Close']].dropna()

        if len(data) < 200:
            data = yf.download(
                "AAPL",
                start="2010-01-01",
                auto_adjust=True,
                progress=False
            )

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            data = data[['Close']].dropna()

        return data

    except:
        data = yf.download(
            "AAPL",
            start="2010-01-01",
            auto_adjust=True,
            progress=False
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        return data[['Close']].dropna()


def train_model(data):
    df = data.copy()

    df["Lag1"] = df["Close"].shift(1)
    df["Lag2"] = df["Close"].shift(2)
    df["Target"] = df["Close"].shift(-1)

    df.dropna(inplace=True)

    X = df[["Close", "Lag1", "Lag2"]]
    y = df["Target"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_next(model, last_price):
    X_input = np.array([[last_price, last_price, last_price]])
    prediction = model.predict(X_input)

    return float(prediction[0])