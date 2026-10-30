"""
TSLA daily forecast: 3 models x 10 trading days.

Same method as the notebook (23 features, log-return target), but one model
is trained for each forecast day h = 1..10 (direct multi-horizon forecasting).
"""

import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

TICKER = "TSLA"
START = "2016-01-01"
HORIZON = 10
RANDOM_STATE = 42

# Best Optuna parameters found in the notebook (tuned with a 5-day target)
XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 3,
    "learning_rate": 0.09575921394871639,
    "min_child_weight": 20,
    "subsample": 0.6976353016281769,
    "colsample_bytree": 0.761988580619456,
    "gamma": 1.3218293415157272,
    "reg_alpha": 0.00011645046010590423,
    "reg_lambda": 0.18436435780078145,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "tree_method": "hist",
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def download_data():
    """Download daily TSLA prices (open, high, low, close, adjclose, volume)."""
    raw = yf.download(
        TICKER, start=START, interval="1d",
        auto_adjust=False, progress=False,
    )
    if raw.empty:
        raise ValueError("No data could be downloaded from Yahoo Finance.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.copy()
    df.columns = [str(c).lower().replace(" ", "") for c in df.columns]
    df = df[["open", "high", "low", "close", "adjclose", "volume"]]
    df.index = pd.to_datetime(df.index)
    df = df.dropna().sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_features(df):
    """Same 23 features as the notebook."""
    data = df.copy()

    # Adjusted prices: adjust open/high/low too, then use adjclose as close
    adjustment = data["adjclose"] / data["close"]
    for col in ["open", "high", "low"]:
        data[col] = data[col] * adjustment
    data["close"] = data["adjclose"]

    log_close = np.log(data["close"])

    for lag in [1, 2, 3, 5, 10, 20, 40, 60]:
        data[f"return_{lag}"] = log_close.diff(lag)

    for window in [5, 10, 20, 50, 100]:
        sma = data["close"].rolling(window).mean()
        data[f"sma_ratio_{window}"] = data["close"] / sma - 1

    daily_return = log_close.diff()
    for window in [5, 10, 20, 40]:
        data[f"volatility_{window}"] = daily_return.rolling(window).std()

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["rsi_14"] = 100 - 100 / (1 + rs)

    for window in [5, 10, 20]:
        avg_volume = data["volume"].rolling(window).mean()
        data[f"volume_ratio_{window}"] = (
            data["volume"] / avg_volume.replace(0, np.nan)
        )

    data["daily_range"] = (data["high"] - data["low"]) / data["close"]
    data["candle_position"] = (
        (data["close"] - data["low"])
        / (data["high"] - data["low"]).replace(0, np.nan)
    )

    features = [
        c for c in data.columns
        if c.startswith(("return_", "sma_ratio_", "volatility_", "volume_ratio_"))
        or c in ["rsi_14", "daily_range", "candle_position"]
    ]
    data = data.replace([np.inf, -np.inf], np.nan)
    return data, features


def make_forecast(df=None):
    """
    Train the models on all data up to today and forecast the next
    HORIZON trading days.

    Returns a long table: one row per (model, horizon).
    """
    if df is None:
        df = download_data()

    data, features = build_features(df)

    latest = data.dropna(subset=features).tail(1)
    if latest.empty:
        raise ValueError("Not enough history to compute the features.")
    origin_date = latest.index[0]
    if origin_date != data.index[-1]:
        raise ValueError("The latest row has missing features.")
    last_price = float(latest["close"].iloc[0])

    # Estimated target dates (weekdays only; holidays are handled later
    # when the real prices are matched by trading-day count)
    target_dates = pd.bdate_range(
        origin_date + pd.Timedelta(days=1), periods=HORIZON
    )

    rows = []
    for h in range(1, HORIZON + 1):
        # Target for this horizon: log-return h trading days ahead
        target = np.log(data["close"].shift(-h) / data["close"])
        train = data[features].assign(target=target).dropna()
        X, y = train[features], train["target"]

        predictions = {"Naive": last_price}

        xgb_model = XGBRegressor(**XGB_PARAMS).fit(X, y)
        predictions["XGBoost"] = last_price * float(
            np.exp(xgb_model.predict(latest[features])[0])
        )

        rf_model = RandomForestRegressor(**RF_PARAMS).fit(X, y)
        predictions["Random Forest"] = last_price * float(
            np.exp(rf_model.predict(latest[features])[0])
        )

        for model_name, price in predictions.items():
            rows.append({
                "origin_date": origin_date.strftime("%Y-%m-%d"),
                "horizon": h,
                "target_date_est": target_dates[h - 1].strftime("%Y-%m-%d"),
                "model": model_name,
                "last_price": round(last_price, 4),
                "predicted_price": round(price, 4),
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    forecast = make_forecast()

    wide = forecast.pivot(
        index=["horizon", "target_date_est"],
        columns="model",
        values="predicted_price",
    )[["Naive", "Random Forest", "XGBoost"]]

    print(f"Forecast made from the close of {forecast['origin_date'].iloc[0]}")
    print(f"Last price: {forecast['last_price'].iloc[0]:.2f} USD\n")
    print(wide.round(2))

    forecast.to_csv("latest_forecast.csv", index=False)
    print("\nSaved: latest_forecast.csv")
