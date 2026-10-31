import altair as alt
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
    padding-top: 4.5rem;
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
        ["Home", "Prediction", "Performance"],
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



    st.markdown(
        '<div class="section-heading">Forecast: next 10 trading days</div>',
        unsafe_allow_html=True
    )

    try:
        log = pd.read_csv("data/forecast_log.csv")
    except FileNotFoundError:
        st.info("The forecast journal is not available yet.")
        st.stop()

    last_origin = log["origin_date"].max()
    latest = log[log["origin_date"] == last_origin]

    st.caption(f"Forecast made from the closing price of {last_origin}.")

    forecast_table = latest.pivot(
        index="target_date_est",
        columns="model",
        values="predicted_price"
    )[["Naive", "Random Forest", "XGBoost"]]
    forecast_table.index = pd.to_datetime(forecast_table.index)

    history = close.tail(60).rename("Historical").to_frame()


    chart_df = (
        pd.concat([history, forecast_table])
        .rename_axis("Date")
        .reset_index()
        .melt("Date", var_name="Series", value_name="Price")
        .dropna()
    )

    zoom_chart = (
        alt.Chart(chart_df)
        .mark_line()
        .encode(
            x="Date:T",
            y=alt.Y("Price:Q", scale=alt.Scale(zero=False)),
            color="Series:N",
        )
    )
    st.altair_chart(zoom_chart, width="stretch")

    

    display_table = forecast_table.copy()
    display_table.index = display_table.index.strftime("%Y-%m-%d")
    display_table.index.name = "Date"
    st.dataframe(
        display_table.style.format("${:.2f}"),
        width="stretch"
    )


    
    st.stop()


if page == "Performance":
    st.title("Performance")
    st.info("This page will be completed soon.")
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
    '<div class="section-heading"> Objective </div>',
    unsafe_allow_html=True
)


st.markdown(
    "The general objective is to build a reliable model that forecasts "
    "Tesla (TSLA) closing prices over the next three months. As a first "
    "stage, the project forecasts the next 10 trading days, every day, "
    "and tests how each model performs against the actual prices."
)

st.markdown(
    "During the three-month pilot, every daily forecast is recorded and "
    "later compared with the actual closing price. This gives a "
    "time-ordered record of how each model performs, and shows which one "
    "is worth improving."
)




# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
    TSLA FORECASTING DASHBOARD · DEVELOPMENT VERSION
</div>
""", unsafe_allow_html=True)
