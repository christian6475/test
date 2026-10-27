
import yfinance as yf

# Download Tesla stock data
ticker = "TSLA"

data = yf.download(
    ticker,
    period="1mo",
    interval="1d",
    auto_adjust=False,
    progress=False
)

if data.empty:
    print("No data was downloaded. Please check your internet connection.")
else:
    print("Data downloaded successfully!")
    print(data[["Close"]].tail(10))
