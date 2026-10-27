
import streamlit as st
import yfinance as yf
import pandas as pd


st.set_page_config(
    page_title="TSLA | Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
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
#st.markdown('<div class="top-tag">Virtual Internship</div>', unsafe_allow_html=True)
st.markdown('<div class="top-tag">Virtual Internship</div>', unsafe_allow_html=True)

with st.sidebar:
    page = st.radio(
        "Menu",
        ["Accueil", "Prédiction"],
        label_visibility="collapsed",
    )


if page == "Prédiction":
    st.title("Prédiction")
    st.info("Cette page sera complétée prochainement.")
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
    '<div class="section-description">'
    'Explore the forecasting methods planned for this project.'
    '</div>',
    unsafe_allow_html=True
)


# -------------------------
# Model cards
# -------------------------
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="model-card">
        <div class="model-number">MODEL 01</div>
        <div class="model-title">Naive Forecast</div>
        <div class="model-description">
            A simple benchmark that uses the latest observed closing
            price as the forecast for future trading days.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="model-card">
        <div class="model-number">MODEL 02</div>
        <div class="model-title">Random Forest</div>
        <div class="model-description">
            An ensemble of decision trees that can learn relationships
            between historical market features and future prices.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="model-card">
        <div class="model-number">MODEL 03</div>
        <div class="model-title">XGBoost</div>
        <div class="model-description">
            A gradient boosting method that builds trees sequentially
            to improve predictive performance.
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------
# Forecast settings
# -------------------------
st.markdown(
    '<div class="section-heading">Forecast Settings</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="settings-card">
    <div class="eyebrow">FORECAST HORIZON</div>
    <div style="font-size: 1.7rem; font-weight: 700;
                color: #ffffff; margin-top: 8px;">
        10 Trading Days
    </div>
    <div style="color: #b5c4d9; margin-top: 8px; line-height: 1.6;">
        The dashboard will forecast the next 10 trading days after
        the latest available TSLA closing price.
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "The interface is ready for the next stage. Historical market data, "
    "model training, forecasts, and performance metrics have not yet "
    "been connected."
)


# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
    TSLA FORECASTING DASHBOARD · DEVELOPMENT VERSION
</div>
""", unsafe_allow_html=True)


# -------------------------
# Historical TSLA data
# -------------------------
st.markdown(
    '<div class="section-heading">TSLA Market Data</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Latest available daily closing prices from Yahoo Finance.'
    '</div>',
    unsafe_allow_html=True
)

if st.button("Load latest TSLA data"):
    with st.spinner("Downloading TSLA data..."):
        data = yf.download(
            "TSLA",
            period="1y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

    if data.empty:
        st.error(
            "No data could be downloaded. "
            "Please check your internet connection and try again."
        )
    else:
        # Handle the columns returned by yfinance
        if isinstance(data.columns, pd.MultiIndex):
            close = data["Close"]["TSLA"]
        else:
            close = data["Close"]

        close = close.dropna()

        latest_date = close.index[-1].strftime("%Y-%m-%d")
        latest_price = float(close.iloc[-1])

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Latest closing price", f"${latest_price:.2f}")

        with col2:
            st.metric("Latest available date", latest_date)

        st.subheader("Closing price history")

        chart_data = close.rename("Closing Price").to_frame()
        st.line_chart(chart_data)

        st.subheader("Last 10 observations")

        table = close.tail(10).rename("Closing Price").to_frame()
        table.index.name = "Date"
        st.dataframe(
            table.style.format("${:.2f}"),
            use_container_width=True
        )
