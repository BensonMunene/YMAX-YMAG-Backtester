import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import strategy_overview
from Ratios_Strat_App import display_ratios_zones_indicator

# ==============================================
# Page Configuration
# ==============================================
st.set_page_config(page_title="YMAX YMAG Backtester", layout="wide")

# ==============================================
# Session-State Initialization (to persist results across reruns)
# ==============================================
if "ymax_df_final" not in st.session_state:
    st.session_state["ymax_df_final"] = None
if "ymag_df_final" not in st.session_state:
    st.session_state["ymag_df_final"] = None
if "perf_df_ymax_final" not in st.session_state:
    st.session_state["perf_df_ymax_final"] = None
if "perf_df_ymag_final" not in st.session_state:
    st.session_state["perf_df_ymag_final"] = None
if "prices_and_stats_df" not in st.session_state:
    st.session_state["prices_and_stats_df"] = None

# ==============================================
# Setting the Data Directory
# ==============================================
data_dir = "data"

# ==============================================
# STRATEGY 1
# ==============================================
def backtest_strategy_1(df, asset="YMAX", initial_investment=10_000):
    """
    Implements Strategy 1 for the chosen asset ("YMAX" or "YMAG"):
      1) If VIX < 20 and VVIX < 100 => Long (No Hedge)
      2) If VIX >= 20 or VVIX >= 100 => Long + Short QQQ
      3) If VIX >= 20 or VVIX >= 100 and correlation < -0.3 => No Investment
    """
    temp_df = df.copy()

    if asset == "YMAX":
        corr_vix_col = "YMAX-VIX Correlation"
        corr_vvix_col = "YMAX-VVIX Correlation"
        price_col = "YMAX"
        div_col = "YMAX Dividends"
    else:
        corr_vix_col = "YMAG-VIX Correlation"
        corr_vvix_col = "YMAG-VVIX Correlation"
        price_col = "YMAG"
        div_col = "YMAG Dividends"

    def strategy_rule(row):
        if row["VIX"] < 20 and row["VVIX"] < 100:
            return "Long (No Hedge)"
        elif row["VIX"] >= 20 or row["VVIX"] >= 100:
            if row[corr_vix_col] < -0.3 or row[corr_vvix_col] < -0.3:
                return "No Investment"
            else:
                return "Long + Short QQQ"
        else:
            return "No Investment"

    temp_df["Strategy"] = temp_df.apply(strategy_rule, axis=1)
    temp_df["Portfolio_Value"] = initial_investment
    temp_df["QQQ_Shares_Short"] = 0
    temp_df["QQQ_Short_Loss"] = 0

    for i in range(1, len(temp_df)):
        prev_val = temp_df.iloc[i - 1]["Portfolio_Value"]
        today_strategy = temp_df.iloc[i]["Strategy"]
        y_price_yest = temp_df.iloc[i - 1][price_col]
        q_price_yest = temp_df.iloc[i - 1]["QQQ"]
        y_price_today = temp_df.iloc[i][price_col]
        y_div_today = temp_df.iloc[i][div_col]

        # Carry forward QQQ shares short by default
        temp_df.at[temp_df.index[i], "QQQ_Shares_Short"] = temp_df.iloc[i - 1]["QQQ_Shares_Short"]

        if today_strategy == "Long (No Hedge)":
            shares_held = prev_val / y_price_yest
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = shares_held * (y_price_today + y_div_today)

        elif today_strategy == "Long + Short QQQ":
            shares_held = prev_val / y_price_yest
            if temp_df.iloc[i - 1]["Strategy"] != "Long + Short QQQ":
                temp_df.at[temp_df.index[i], "QQQ_Shares_Short"] = prev_val / q_price_yest

            qqq_shares_short = temp_df.iloc[i]["QQQ_Shares_Short"]
            q_price_today = temp_df.iloc[i]["QQQ"]
            hedge_pnl = qqq_shares_short * (q_price_yest - q_price_today)

            temp_df.at[temp_df.index[i], "QQQ_Short_Loss"] = -hedge_pnl
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = (
                shares_held * (y_price_today + y_div_today)
            ) + hedge_pnl

        else:  # "No Investment"
            temp_df.at[temp_df.index[i], "Portfolio_Value"] = prev_val

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df

# ==============================================
# STRATEGY 2
# ==============================================
def backtest_strategy_2(
    df,
    asset="YMAX",
    initial_investment=10_000,
    vix_lower=15,
    vix_upper=20,
    vvix_lower=90,
    vvix_upper=100,
    vvix_reentry_upper=95
):
    """
    1) Remain in market if: vix_lower <= VIX <= vix_upper,  vvix_lower <= VVIX < vvix_upper
    2) Re-enter if: vix_lower <= VIX <= vix_upper, vvix_lower <= VVIX <= vvix_reentry_upper
    3) Exit if: VIX < vix_lower or VIX > vix_upper or VVIX < vvix_lower or VVIX >= vvix_upper
    """

    temp_df = df.copy()
    temp_df.sort_values("Date", inplace=True)
    temp_df.reset_index(drop=True, inplace=True)

    # Identify columns for this asset
    if asset == "YMAX":
        price_col = "YMAX"
        div_col   = "YMAX Dividends"
        strategy_label_long = "Long YMAX"
    else:
        price_col = "YMAG"
        div_col   = "YMAG Dividends"
        strategy_label_long = "Long YMAG"

    # Helper functions to apply your chosen thresholds
    def exit_condition(vix, vvix):
        return (vix < vix_lower) or (vix > vix_upper) or (vvix < vvix_lower) or (vvix >= vvix_upper)

    def reentry_condition(vix, vvix):
        return (vix_lower <= vix <= vix_upper) and (vvix_lower <= vvix <= vvix_reentry_upper)

    # Initialize portfolio columns
    temp_df["Portfolio_Value"] = np.nan
    temp_df.loc[0, "Portfolio_Value"] = initial_investment
    temp_df["In_Market"] = False
    temp_df.loc[0, "In_Market"] = False
    temp_df["Shares_Held"] = 0.0
    temp_df.loc[0, "Shares_Held"] = 0.0
    temp_df["Strategy"] = "No Investment"

    # Main loop
    for i in range(1, len(temp_df)):
        temp_df.loc[i, "Portfolio_Value"] = temp_df.loc[i-1, "Portfolio_Value"]
        temp_df.loc[i, "In_Market"]       = temp_df.loc[i-1, "In_Market"]
        temp_df.loc[i, "Shares_Held"]     = temp_df.loc[i-1, "Shares_Held"]

        vix_today    = temp_df.loc[i, "VIX"]
        vvix_today   = temp_df.loc[i, "VVIX"]
        price_today  = temp_df.loc[i, price_col]
        div_today    = temp_df.loc[i, div_col]
        currently_in_market = temp_df.loc[i-1, "In_Market"]

        if currently_in_market:
            # Already in the market: check exit condition
            if exit_condition(vix_today, vvix_today):
                # Exit
                temp_df.loc[i, "In_Market"]   = False
                temp_df.loc[i, "Shares_Held"] = 0.0
                temp_df.loc[i, "Strategy"]    = "No Investment"
            else:
                # Stay invested
                shares_held = temp_df.loc[i, "Shares_Held"]
                new_val = shares_held * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
                temp_df.loc[i, "Strategy"] = strategy_label_long
        else:
            # Currently out of market: check re-entry condition
            if reentry_condition(vix_today, vvix_today):
                cash_available = temp_df.loc[i, "Portfolio_Value"]
                if price_today > 0:
                    shares_bought = cash_available / price_today
                else:
                    shares_bought = 0
                temp_df.loc[i, "Shares_Held"]  = shares_bought
                temp_df.loc[i, "In_Market"]    = True
                temp_df.loc[i, "Strategy"]     = strategy_label_long
                new_val = shares_bought * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
            else:
                temp_df.loc[i, "Strategy"] = "No Investment"

    # Calculate daily returns
    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    temp_df["Portfolio_Value"] = temp_df["Portfolio_Value"].round(2)
    return temp_df

# ==============================================
# STRATEGY 3
# ==============================================
def backtest_strategy_3(df, asset="YMAX", initial_investment=10_000):
    """
    Enter if VIX < 20 and VVIX < 95
    Exit if VIX > 20 or VVIX > 100
    """
    temp_df = df.copy()
    temp_df.sort_values("Date", inplace=True)
    temp_df.reset_index(drop=True, inplace=True)

    if asset == "YMAX":
        price_col = "YMAX"
        div_col = "YMAX Dividends"
        strategy_label_long = "Long YMAX"
    else:
        price_col = "YMAG"
        div_col = "YMAG Dividends"
        strategy_label_long = "Long YMAG"

    temp_df["Portfolio_Value"] = np.nan
    temp_df.loc[0, "Portfolio_Value"] = initial_investment
    temp_df["In_Market"] = False
    temp_df.loc[0, "In_Market"] = False
    temp_df["Shares_Held"] = 0.0
    temp_df.loc[0, "Shares_Held"] = 0.0
    temp_df["Strategy"] = "No Investment"

    def in_market_condition(vix, vvix):
        return (vix < 20) and (vvix < 95)

    def exit_condition(vix, vvix):
        return (vix > 20) or (vvix > 100)

    for i in range(1, len(temp_df)):
        temp_df.loc[i, "Portfolio_Value"] = temp_df.loc[i-1, "Portfolio_Value"]
        temp_df.loc[i, "In_Market"] = temp_df.loc[i-1, "In_Market"]
        temp_df.loc[i, "Shares_Held"] = temp_df.loc[i-1, "Shares_Held"]

        vix_today = temp_df.loc[i, "VIX"]
        vvix_today = temp_df.loc[i, "VVIX"]
        price_today = temp_df.loc[i, price_col]
        div_today = temp_df.loc[i, div_col]

        currently_in_market = temp_df.loc[i-1, "In_Market"]

        if currently_in_market:
            if exit_condition(vix_today, vvix_today):
                temp_df.loc[i, "In_Market"] = False
                temp_df.loc[i, "Shares_Held"] = 0.0
                temp_df.loc[i, "Strategy"] = "No Investment"
            else:
                shares_held = temp_df.loc[i, "Shares_Held"]
                new_val = shares_held * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
                temp_df.loc[i, "Strategy"] = strategy_label_long
        else:
            if in_market_condition(vix_today, vvix_today):
                cash_available = temp_df.loc[i, "Portfolio_Value"]
                if price_today > 0:
                    shares_bought = cash_available / price_today
                    temp_df.loc[i, "Shares_Held"] = shares_bought
                    temp_df.loc[i, "In_Market"] = True
                    temp_df.loc[i, "Strategy"] = strategy_label_long
                    new_val = shares_bought * (price_today + div_today)
                    temp_df.loc[i, "Portfolio_Value"] = new_val
                else:
                    temp_df.loc[i, "Strategy"] = "No Investment"
            else:
                temp_df.loc[i, "Strategy"] = "No Investment"

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df

# ==============================================
# Rolling Correlation + Performance
# ==============================================
def compute_rolling_correlations(df, window):
    returns = df.loc[:, ~df.columns.str.contains("Dividends")].pct_change()
    corr_df = pd.DataFrame(index=df.index)

    # Check available columns dynamically
    if "YMAX" in returns.columns:
        corr_df["YMAX-VIX Correlation"] = returns["YMAX"].rolling(window=window).corr(returns["VIX"])
        corr_df["YMAX-VVIX Correlation"] = returns["YMAX"].rolling(window=window).corr(returns["VVIX"])

    if "YMAG" in returns.columns:
        corr_df["YMAG-VIX Correlation"] = returns["YMAG"].rolling(window=window).corr(returns["VIX"])
        corr_df["YMAG-VVIX Correlation"] = returns["YMAG"].rolling(window=window).corr(returns["VVIX"])

    merged = df.join(corr_df)
    merged.dropna(inplace=True)
    return merged

def calculate_performance_metrics(portfolio_df):
    df = portfolio_df.dropna(subset=["Portfolio_Value"]).copy()
    if len(df) < 2:
        return {}
    start_val = df.iloc[0]["Portfolio_Value"]
    end_val = df.iloc[-1]["Portfolio_Value"]
    total_return = (end_val / start_val - 1) * 100
    num_days = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days
    years = num_days / 365 if num_days > 0 else 1
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    ann_vol = df["Portfolio_Return"].std() * np.sqrt(252) * 100
    risk_free_rate = 0.02
    sharpe = (cagr / 100 - risk_free_rate) / (ann_vol / 100) if ann_vol != 0 else np.nan
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown = df["Portfolio_Value"] / rolling_max - 1
    max_dd = drawdown.min() * 100
    calmar = (cagr / abs(max_dd)) if max_dd != 0 else np.nan
    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Ann. Volatility (%)": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown (%)": max_dd,
        "Calmar Ratio": calmar,
    }

def plot_portfolio_value(df, asset_label="Portfolio Value"):
    fig = px.line(
        df,
        x="Date",
        y="Portfolio_Value",
        title=f"{asset_label} Over Time",
        labels={"Portfolio_Value": "Portfolio Value ($)"},
    )
    fig.update_traces(
        line=dict(width=2),
        hovertemplate="Date=%{x}<br>Portfolio Value=$%{y:.2f}")
    
    fig.update_layout(
        xaxis_title="Date", 
        yaxis_title="Value ($)",
        yaxis=dict(tickformat=".2f"))
    return fig

def plot_strategy_distribution(df):
    strat_counts = df["Strategy"].value_counts()
    fig = px.bar(
        x=strat_counts.index,
        y=strat_counts.values,
        labels={"x": "Strategy", "y": "Number of Days"},
        title="Strategy Distribution Over Time",
    )
    return fig

def plot_drawdown(df):
    rolling_max = df["Portfolio_Value"].cummax()
    drawdown_series = (df["Portfolio_Value"] / rolling_max - 1) * 100
    dd_df = df[["Date"]].copy()
    dd_df["Drawdown"] = drawdown_series
    fig = px.area(
        dd_df,
        x="Date",
        y="Drawdown",
        title="Drawdown Over Time",
        labels={"Drawdown": "Drawdown (%)"},
    )
    fig.update_traces(
        line=dict(width=2), 
        fill="tozeroy",
        hovertemplate="Date=%{x}<br>Drawdown=%{y:.2f}%<extra></extra>")
    
    # Format y-axis ticks to 2 decimal places
    fig.update_layout(yaxis=dict(tickformat=".2f"))
    
    return fig

# ==============================================
# Entry-Exit Plot
# ==============================================
def plot_entry_exit(df, asset_name="YMAX"):
    temp_df = df.copy()
    if "In_Market" not in temp_df.columns:
        temp_df["In_Market"] = temp_df["Strategy"].apply(lambda x: x.startswith("Long"))
    temp_df["Entry"] = (temp_df["In_Market"].shift(1) == False) & (temp_df["In_Market"] == True)
    temp_df["Exit"]  = (temp_df["In_Market"].shift(1) == True) & (temp_df["In_Market"] == False)
    entry_days = temp_df[temp_df["Entry"] == True]
    exit_days  = temp_df[temp_df["Exit"] == True]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=temp_df["Date"],
            y=temp_df[asset_name],
            mode="lines",
            line=dict(color="blue", width=2),
            name=f"{asset_name} Price",
            yaxis="y1"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=entry_days["Date"],
            y=entry_days[asset_name],
            mode="markers",
            marker=dict(symbol="triangle-up", color="green", size=10),
            name="Entry",
            yaxis="y1"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=exit_days["Date"],
            y=exit_days[asset_name],
            mode="markers",
            marker=dict(symbol="triangle-down", color="red", size=10),
            name="Exit",
            yaxis="y1"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=temp_df["Date"],
            y=temp_df["Portfolio_Value"],
            mode="lines",
            line=dict(color="orange", width=2),
            name="Portfolio Value",
            yaxis="y2",
            hovertemplate="<span style='color:orange;'>Portfolio Value=$%{y:.2f}</span><extra></extra>"
        )
    )
    fig.update_layout(
        title=f"Entry & Exit Plot ({asset_name})",
        xaxis=dict(title="Date"),
        yaxis=dict(title=f"{asset_name} Price ($)", side="left"),
        yaxis2=dict(
            title="Portfolio Value ($)",
            side="right",
            overlaying="y",
            position=1.0
        ),
        legend=dict(
            x=0.5, y=1.15,
            xanchor='center',
            orientation="h",
            bgcolor="rgba(255,255,255,0.6)"
        ),
        hovermode="closest"
    )
    return fig

# ==============================================
# Global placeholders
# ==============================================
prices_and_stats_df = None
ymax_df_final = None
ymag_df_final = None
perf_df_ymax_final = None
perf_df_ymag_final = None

# ==============================================
# Sidebar Navigation
# ==============================================
page = st.sidebar.radio("Navigation", ["Backtester", "Strategy Overview", "Ratios Zones Indicator", "About"])

# ==============================================
# BACKTESTER PAGE
# ==============================================
if page == "Backtester":
    st.title("YMAX YMAG Backtester")

    st.header("Strategy Summaries")
    st.markdown(
        """
**Strategy 1:** Uses VIX/VVIX thresholds and correlation with market volatility 
to determine long positions in YMAX/YMAG—with hedging via QQQ when volatility is high.

**Strategy 2:** invests only when VIX is between 15 and 20 and VVIX is between 90 and 100, 
exiting if volatility moves outside these ranges. Its default sliders are VIX Lower=15, 
VIX Upper=20, VVIX Lower=90, VVIX Upper=100, and VVIX Re-Entry=95. This ensures the strategy 
stays in a “safe” volatility band and re-enters if conditions stabilize.

**Strategy 3:** enters whenever VIX < 20 and VVIX < 95, exiting if VIX > 20 or VVIX > 100. 
By default, it uses VIX=20 and VVIX=95, both adjustable via sliders. This simpler approach quickly 
leaves the market if volatility climbs beyond those cutoffs.
    """)
    
    st.subheader("Data Preview")
    st.markdown("This is a preview of the data (complete data) that we are using to run the backtests below.")
    try:
        csv_data = pd.read_csv("All Assets and Dividends.csv")         
         # 2) Prepare data
        csv_data["Date"] = pd.to_datetime(csv_data["Date"])
        csv_data.set_index("Date", inplace=True)
        # Display the recent dates rows of the data
        st.dataframe(csv_data, height=200)
    except FileNotFoundError:
        st.error("Could not find 'All Assets and Dividends.csv'. Make sure it's in the same folder.")

#Explanation for the buttons of assets and strategies        
    st.markdown("""
    ---
    Below, you can choose which asset(s) to backtest and which strategy to apply. 
    Your selections will determine the logic used in the subsequent backtest calculations.
    """)

    col_sel1, col_date, col_freq, col_sel2 = st.columns([1, 1, 1, 1], gap="medium")
    with col_sel1:
        st.subheader("Asset Selection")
        asset_choice = st.radio(
            "Select the asset(s) to backtest:",
            options=["YMAX", "YMAG", "Both"],
            index=2
        )
        
    with col_date:
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=datetime.date(2024, 1, 1))
        end_date = st.date_input("End Date", value=datetime.date(2025, 2, 15))
        
    with col_freq:
        st.subheader("Frequency")
        frequency = st.selectbox(
            "Select data frequency:",
            options=["Daily", "4 hours", "Hourly", "30 Minutes"],
            index=0)
        
    with col_sel2:
        st.subheader("Strategy Selection")
        strategy_choice = st.radio(
            "Select strategy to backtest:",
            options=["Strategy 1", "Strategy 2", "Strategy 3"]
        )
        
    st.markdown("---")
    
     # Strategy 1 parameters (correlation window)
    if strategy_choice == "Strategy 1":
        st.subheader("Parameters for Strategy 1")
        corr_window = st.slider("Select correlation window (days):", min_value=1, max_value=30, value=14)
    else:
        corr_window = 14  # default if not Strategy 1
        
    # After the user selects the strategy_choice...
    if strategy_choice == "Strategy 2":
        st.subheader("Parameters for Strategy 2")
    # Sliders for each threshold
        vix_lower = st.slider("VIX Lower", min_value=0, max_value=50, value=15, step=1)
        vix_upper = st.slider("VIX Upper", min_value=0, max_value=50, value=20, step=1)
        vvix_lower = st.slider("VVIX Lower", min_value=0, max_value=200, value=90, step=1)
        vvix_upper = st.slider("VVIX Upper", min_value=0, max_value=200, value=100, step=1)
        vvix_reentry_upper = st.slider("VVIX Re-Entry Upper", min_value=0, max_value=200, value=95, step=1)
    else:
    # Default parameters 
        vix_lower = 15
        vix_upper = 20
        vvix_lower = 90
        vvix_upper = 100
        vvix_reentry_upper = 95

    # For Strategy 3 parameters with sliders.
    if strategy_choice == "Strategy 3":
        st.subheader("Parameters for Strategy 3")
        vix_threshold = st.slider("Select VIX threshold:", min_value=1, max_value=40, value=20)
        vvix_threshold = st.slider("Select VVIX threshold:", min_value=1, max_value=120, value=95)
    else:
        vix_threshold = 20
        vvix_threshold = 95

    run_backtest = st.button("Run Backtest for Selected Strategy")

    if run_backtest:
        
         # 1)  Load CSV file(s) based on asset selection and frequency
        file_suffix_map = {"Daily": "Daily", "4 hours": "4H", "Hourly": "1H", "30 Minutes": "30M"}
        selected_suffix = file_suffix_map.get(frequency, "Daily")
        
        if asset_choice == "YMAX":
            try:
                all_assets = pd.read_csv(f"{data_dir}/YMAX_VIX_VVIX_QQQ_{selected_suffix}.csv")
            except FileNotFoundError:
                st.error(f"Could not find YMAX_VIX_VVIX_QQQ_{selected_suffix}.csv")
                st.stop()
            all_assets["Date"] = pd.to_datetime(all_assets["Date"])
            all_assets.set_index("Date", inplace=True)
            all_assets.sort_index(inplace=True)
        elif asset_choice == "YMAG":
            try:
                all_assets = pd.read_csv(f"{data_dir}/YMAG_VIX_VVIX_QQQ_{selected_suffix}.csv")
            except FileNotFoundError:
                st.error(f"Could not find YMAG_VIX_VVIX_QQQ_{selected_suffix}.csv")
                st.stop()
            all_assets["Date"] = pd.to_datetime(all_assets["Date"])
            all_assets.set_index("Date", inplace=True)
            all_assets.sort_index(inplace=True)
        elif asset_choice == "Both":
            try:
                df_ymax = pd.read_csv(f"{data_dir}/YMAX_VIX_VVIX_QQQ_{selected_suffix}.csv")
                df_ymag = pd.read_csv(f"{data_dir}/YMAG_VIX_VVIX_QQQ_{selected_suffix}.csv")
            except FileNotFoundError:
                st.error("Could not find one of the required CSV files for both assets.")
                st.stop()
            df_ymax["Date"] = pd.to_datetime(df_ymax["Date"])
            df_ymag["Date"] = pd.to_datetime(df_ymag["Date"])
            df_ymax.set_index("Date", inplace=True)
            df_ymag.set_index("Date", inplace=True)
            df_ymax.sort_index(inplace=True)
            df_ymag.sort_index(inplace=True)
            # Drop duplicate columns (assumed common columns: VIX, VVIX, QQQ) from the YMAG file
            df_ymag = df_ymag.drop(columns=["VIX", "VVIX", "QQQ"], errors="ignore")
            all_assets = df_ymax.join(df_ymag, how="inner")
        
        # 2.1) FILTER by chosen date range
        # Make sure start_date <= end_date, or handle it gracefully
        all_assets = all_assets.loc[str(start_date):str(end_date)]
        
        # 3) Compute rolling correlations using corr_window
        ps_df = compute_rolling_correlations(all_assets, corr_window)
        ps_df.reset_index(inplace=True)
        prices_and_stats_df = ps_df.copy()
        
        # 4) Process asset based on selected strategy
        def process_asset(asset_name, strategy):
            if strategy == "Strategy 1":
                return backtest_strategy_1(ps_df, asset=asset_name)

            elif strategy == "Strategy 2":
                # Pass dynamic slider values for Strategy 2
                return backtest_strategy_2(
                    ps_df,
                    asset=asset_name,
                    vix_lower=vix_lower,
                    vix_upper=vix_upper,
                    vvix_lower=vvix_lower,
                    vvix_upper=vvix_upper,
                    vvix_reentry_upper=vvix_reentry_upper
                )
                
            elif strategy == "Strategy 3":
                return backtest_strategy_3(ps_df, asset=asset_name)
            else:
                return None
        chosen_strat = strategy_choice

        if asset_choice == "Both":
            # YMAX Backtest
            st.markdown(f"## {chosen_strat} Backtest - YMAX")
            ymax_res = process_asset("YMAX", chosen_strat)
            
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                fig_val_ymax = plot_portfolio_value(ymax_res, asset_label=f"{chosen_strat} (YMAX)")
                st.plotly_chart(fig_val_ymax, use_container_width=True, key="val_ymax")
            with row1_col2:
                fig_entry_exit_ymax = plot_entry_exit(ymax_res, asset_name="YMAX")
                st.plotly_chart(fig_entry_exit_ymax, use_container_width=True, key="entry_exit_ymax")
            
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                fig_dd_ymax = plot_drawdown(ymax_res)
                st.plotly_chart(fig_dd_ymax, use_container_width=True, key="dd_ymax")
            with row2_col2:
                fig_strat_ymax = plot_strategy_distribution(ymax_res)
                st.plotly_chart(fig_strat_ymax, use_container_width=True, key="strat_ymax")
            
            ymax_metrics = calculate_performance_metrics(ymax_res)
            if ymax_metrics:
                df_ymax_perf = pd.DataFrame([ymax_metrics], index=["YMAX Strategy"]).round(2)
                st.dataframe(df_ymax_perf)
            else:
                st.info("Not enough data points for YMAX metrics.")
            
            ymax_res["Portfolio_Return"] = (ymax_res["Portfolio_Return"] * 100).round(2)
            ymax_res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
            
            # Store YMAX results in session state
            st.session_state["ymax_df_final"] = ymax_res
            st.session_state["perf_df_ymax_final"] = df_ymax_perf

            # YMAG Backtest
            st.markdown(f"## {chosen_strat} Backtest - YMAG")
            ymag_res = process_asset("YMAG", chosen_strat)
            
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                fig_val_ymag = plot_portfolio_value(ymag_res, asset_label=f"{chosen_strat} (YMAG)")
                st.plotly_chart(fig_val_ymag, use_container_width=True, key="val_ymag")
            with row1_col2:
                fig_entry_exit_ymag = plot_entry_exit(ymag_res, asset_name="YMAG")
                st.plotly_chart(fig_entry_exit_ymag, use_container_width=True, key="entry_exit_ymag")
            
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                fig_dd_ymag = plot_drawdown(ymag_res)
                st.plotly_chart(fig_dd_ymag, use_container_width=True, key="dd_ymag")
            with row2_col2:
                fig_strat_ymag = plot_strategy_distribution(ymag_res)
                st.plotly_chart(fig_strat_ymag, use_container_width=True, key="strat_ymag")
            
            ymag_metrics = calculate_performance_metrics(ymag_res)
            if ymag_metrics:
                df_ymag_perf = pd.DataFrame([ymag_metrics], index=["YMAG Strategy"]).round(2)
                st.dataframe(df_ymag_perf)
            else:
                st.info("Not enough data points for YMAG metrics.")
            
            ymag_res["Portfolio_Return"] = (ymag_res["Portfolio_Return"] * 100).round(2)
            ymag_res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
            
            st.session_state["ymag_df_final"] = ymag_res
            st.session_state["perf_df_ymag_final"] = df_ymag_perf

        else:
            # Single asset
            st.markdown(f"## {chosen_strat} Backtest - {asset_choice}")
            res = process_asset(asset_choice, chosen_strat)
            
            row1_col1, row1_col2 = st.columns(2)
            with row1_col1:
                fig_val = plot_portfolio_value(res, asset_label=f"{chosen_strat} ({asset_choice})")
                st.plotly_chart(fig_val, use_container_width=True, key="val_single")
            with row1_col2:
                fig_entry_exit = plot_entry_exit(res, asset_name=asset_choice)
                st.plotly_chart(fig_entry_exit, use_container_width=True, key="entry_exit_single")
            
            row2_col1, row2_col2 = st.columns(2)
            with row2_col1:
                fig_dd = plot_drawdown(res)
                st.plotly_chart(fig_dd, use_container_width=True, key="dd_single")
            with row2_col2:
                fig_strat = plot_strategy_distribution(res)
                st.plotly_chart(fig_strat, use_container_width=True, key="strat_single")
            
            metrics_res = calculate_performance_metrics(res)
            if metrics_res:
                perf_df = pd.DataFrame([metrics_res], index=[f"{asset_choice} Strategy"]).round(2)
                st.dataframe(perf_df)
            else:
                st.info("Not enough data points for metrics.")
            res["Portfolio_Return"] = (res["Portfolio_Return"] * 100).round(2)
            res.rename(columns={"Portfolio_Return": "Portfolio_Return (%)"}, inplace=True)
            if asset_choice == "YMAX":
                st.session_state["ymax_df_final"] = res.copy()
            else:
                st.session_state["ymag_df_final"] = res.copy()

        # Store prices_and_stats_df in session state
        st.session_state["prices_and_stats_df"] = prices_and_stats_df

        # Display the stored data (preview)
        if st.session_state["ymax_df_final"] is not None:
            st.subheader("YMAX Trading Results")
            st.dataframe(st.session_state["ymax_df_final"], height=300)

        if st.session_state["ymag_df_final"] is not None:
            st.subheader("YMAG Trading Results")
            st.dataframe(st.session_state["ymag_df_final"], height=300)

    else:
        st.info("Click 'Run Backtest for Selected Strategy' to see results.")

        
# ==============================================
# STRATEGY OVERVIEW PAGE
# ==============================================

elif page == "Strategy Overview":
    # call the function from strategy_overview
    strategy_overview.display_strategy_overview()

elif page == "Ratios Zones Indicator":
    display_ratios_zones_indicator()

elif page == "About":
    st.title("About")
    st.write("This app allows you to backtest trading strategies on YMAX and YMAG assets using historical data. Features include:")
    st.write("- **Asset Selection**: Choose YMAX, YMAG, or both.")
    st.write("- **Strategy Selection**: Test three strategies with customizable parameters.")
    st.write("- **Interactive Plots**: Visualize performance, entry/exit points, strategy distribution, and drawdowns using Plotly.")
    st.write("- **Performance Metrics**: View Total Return, CAGR, Volatility, Sharpe Ratio, Max Drawdown, and Calmar Ratio.")
    st.write("- **Export**: Save results to an Excel file.")
    
    
    
