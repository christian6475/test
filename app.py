
import streamlit as st
import yfinance as yf
import pandas as pd


st.set_page_config(
    page_title="TSLA | Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1220 0%, #101b30 55%, #0b1220 100%);
    color: #edf4ff;
}

.top-tag {
    color: #53e0c1;
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 18px;
    display: inline-block;
    border: 2px solid #53e0c1;
    border-radius: 8px;
    padding: 6px 16px;
}

.top-bar {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

.author {
    color: #b9c7dc;
    font-size: 0.95rem;
    text-align: right;
    line-height: 1.5;
}

.author b {
    color: #ffffff;
    font-size: 1.1rem;
}


.block-container {
    max-width: 1250px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #edf4ff !important;
}

p, label {
    color: #b9c7dc;
}

.hero {
    background: linear-gradient(120deg, #142943, #123b49);
    border: 1px solid #28546a;
    border-radius: 22px;
    padding: 34px 36px;
    margin-bottom: 28px;
}

.eyebrow {
    color: #53e0c1;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero-title {
    color: #ffffff;
    font-size: 2.5rem;
    font-weight: 750;
    line-height: 1.2;
    margin-top: 12px;
    margin-bottom: 12px;
}

.hero-description {
    color: #c3d4e8;
    font-size: 1.05rem;
    line-height: 1.7;
    max-width: 780px;
}

.section-heading {
    color: #ffffff;
    font-size: 1.45rem;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 6px;
}

.section-description {
    color: #9fb0c8;
    margin-bottom: 22px;
}

.model-card {
    background: #142137;
    border: 1px solid #293b55;
    border-radius: 16px;
    padding: 23px 21px;
    min-height: 215px;
    margin-bottom: 12px;
}

.model-number {
    color: #53e0c1;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.model-title {
    color: #ffffff;
    font-size: 1.3rem;
    font-weight: 700;
    margin-top: 12px;
    margin-bottom: 12px;
}

.model-description {
    color: #b5c4d9;
    font-size: 0.94rem;
    line-height: 1.65;
}

.settings-card {
    background: #142137;
    border: 1px solid #293b55;
    border-radius: 16px;
    padding: 22px 24px;
    margin-top: 10px;
}

.footer {
    color: #8294ae;
    font-size: 0.82rem;
    text-align: center;
    border-top: 1px solid #26364d;
    padding-top: 20px;
    margin-top: 42px;
}

div[data-testid="stAlert"] {
    background-color: #132d3a;
    border: 1px solid #286071;
    border-radius: 12px;
}

#MainMenu, footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# -------------------------
# Header
# -------------------------

st.markdown("""
<div class="top-bar">
    <div class="top-tag">Virtual Internship 2026</div>
    <div class="author">Created by<br><b>Christian Fanevasoa</b></div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    page = st.radio(
        "Menu",
        ["Home", "Prediction"],
        label_visibility="collapsed",
    )


if page == "Prediction":

    @st.cache_data(ttl=900)
    def load_prices(period):
        data = yf.download(
            "TSLA",
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False
        )
        if data.empty:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close.dropna()

    st.markdown(
        '<div class="section-heading">TSLA Price</div>',
        unsafe_allow_html=True
    )

    period = st.selectbox(
        "History period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    close = load_prices(period)

    if close is None or close.empty:
        st.error("No data could be downloaded. Please try again later.")
        st.stop()

    latest_price = float(close.iloc[-1])
    previous_price = float(close.iloc[-2])
    change = latest_price - previous_price
    change_pct = change / previous_price * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Latest available price",
            f"${latest_price:.2f}",
            f"{change:+.2f} ({change_pct:+.2f}%)"
        )

    with col2:
        st.metric("Market date (New York)", close.index[-1].strftime("%Y-%m-%d"))

    st.markdown(
        '<div class="section-heading">Historical Price</div>',
        unsafe_allow_html=True
    )

    st.line_chart(close.rename("Closing Price"))

    st.stop()
st.markdown("""
<div class="hero">
    <div class="eyebrow">FINANCIAL DATA ANALYTICS</div>
    <div class="hero-title">TSLA Forecasting Dashboard</div>
    <div class="hero-description">
        A forecasting interface for analysing Tesla stock closing prices,
        comparing machine learning models, and evaluating their predictive
        performance over the next 10 trading days.
    </div>
</div>
""", unsafe_allow_html=True)




# -------------------------
# Project overview
# -------------------------
st.markdown(
    '<div class="section-heading">Project Overview</div>',
    unsafe_allow_html=True
)

st.markdown(
    "This project is a three-month pilot that monitors daily forecasts of "
    "Tesla (TSLA) closing prices. Each day, three models forecast the next "
    "10 trading days. Every forecast is recorded and later compared with the "
    "actual closing prices, so we can measure how reliable each model is "
    "over time."
)

st.markdown("**1. Naive Forecast (benchmark)**")
st.markdown(
    "The simplest possible forecast: it assumes that every future closing "
    "price will be equal to the latest observed closing price. It needs no "
    "training and serves as the reference point. A more complex model is "
    "only useful if it performs better than this one."
)

st.markdown("**2. Random Forest**")
st.markdown(
    "An ensemble method that builds many decision trees, each trained on a "
    "different random sample of the historical data, and averages their "
    "predictions. Averaging reduces the risk of overfitting and allows the "
    "model to capture non-linear relationships between past market "
    "behaviour and future prices."
)

st.markdown("**3. XGBoost**")
st.markdown(
    "A gradient boosting method that builds trees one after another. Each "
    "new tree focuses on correcting the errors left by the previous ones, "
    "which often gives high accuracy. It is more sensitive to its settings "
    "than Random Forest and needs careful tuning."
)

st.markdown("**How the models are compared**")
st.markdown(
    "All three models forecast the same trading days using the same data. "
    "Their forecasts are compared with the actual prices using error "
    "metrics (MAE, RMSE and MAPE): the lower the error, the better the model."
)




# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
    TSLA FORECASTING DASHBOARD · DEVELOPMENT VERSION
</div>
""", unsafe_allow_html=True)
