"""
Daily update of the forecast journal (data/forecast_log.csv).

Each run:
1. downloads the latest TSLA prices;
2. fills the real price (and the error) for every past forecast whose
   target day has now happened;
3. adds today's forecast (3 models x 10 trading days), unless it is already
   in the journal. Old forecasts are never modified.
"""

from pathlib import Path

import pandas as pd

import forecast_daily as fd

LOG_PATH = Path("data/forecast_log.csv")

COLUMNS = [
    "origin_date", "horizon", "model", "last_price", "predicted_price",
    "target_date_est", "actual_date", "actual_price", "abs_error", "pct_error",
]
TEXT_COLUMNS = {"origin_date": str, "target_date_est": str, "actual_date": str}


def drop_unfinished_day(df, now=None):
    """Remove today's row if the New York market is not closed yet."""
    now = now if now is not None else pd.Timestamp.now(tz="America/New_York")
    if df.index[-1].date() == now.date() and now.hour < 17:
        return df.iloc[:-1]
    return df


def load_log(path):
    if path.exists():
        return pd.read_csv(path, dtype=TEXT_COLUMNS).reindex(columns=COLUMNS)
    return pd.DataFrame(columns=COLUMNS)


def fill_actuals(log, df):
    """Add the real price for forecasts whose target day is now known."""
    prices = df["adjclose"]
    dates = df.index

    for i in log.index[log["actual_price"].isna()]:
        origin = pd.Timestamp(log.at[i, "origin_date"])
        if origin not in dates:
            continue
        # The real price is the h-th trading day after the origin date
        position = dates.get_loc(origin) + int(log.at[i, "horizon"])
        if position >= len(dates):
            continue  # that day has not happened yet

        actual = float(prices.iloc[position])
        error = abs(float(log.at[i, "predicted_price"]) - actual)
        log.at[i, "actual_date"] = dates[position].strftime("%Y-%m-%d")
        log.at[i, "actual_price"] = round(actual, 4)
        log.at[i, "abs_error"] = round(error, 4)
        log.at[i, "pct_error"] = round(error / actual * 100, 4)
    return log


def update_log(df, path=LOG_PATH, forecast_fn=None):
    """Update the journal. Returns (journal, number of rows added)."""
    forecast_fn = forecast_fn or fd.make_forecast

    log = load_log(path)
    log = fill_actuals(log, df)

    origin = df.index[-1].strftime("%Y-%m-%d")
    if (log["origin_date"] == origin).any():
        added = 0
    else:
        new_rows = forecast_fn(df).reindex(columns=COLUMNS)
        log = new_rows if log.empty else pd.concat([log, new_rows], ignore_index=True)
        added = len(new_rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log, added


if __name__ == "__main__":
    df = drop_unfinished_day(fd.download_data())
    log, added = update_log(df)

    print(f"Latest close used: {df.index[-1].date()}")
    if added:
        print(f"Added today's forecast: {added} rows")
    else:
        print("Today's forecast is already in the journal: nothing added")

    filled = int(log["actual_price"].notna().sum())
    print(f"Journal: {len(log)} rows, {log['origin_date'].nunique()} forecast day(s), "
          f"{filled} row(s) with a real price")
    print(f"Saved: {LOG_PATH}")
