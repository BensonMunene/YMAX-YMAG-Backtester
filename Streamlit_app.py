import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import strategy_overview
from ratios_zones_indicator import display_ratios_zones_indicator
from ratios_zones_backtest import display_ratios_zones_backtest

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
if "prices_and_stats_df" not in st.session_state:
    st.session_state["prices_and_stats_df"] = None
if "backtest_done" not in st.session_state:
    st.session_state["backtest_done"] = False

# ==============================================
# Setting the Data Directory
# ==============================================
data_dir = "data"
# data_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data\TradingView Data"

# ==============================================
# STRATEGIES
# ==============================================
def backtest_strategy_1(df, asset="YMAX", initial_investment=10_000):
    """
    Strategy 1:
      1) If VIX < 20 and VVIX < 100 => Long (No Hedge)
      2) If VIX >= 20 or VVIX >= 100 => Long + Short QQQ
      3) If correlation < -0.3 => No Investment
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
    Strategy 2:
      1) In market if vix_lower <= VIX <= vix_upper, vvix_lower <= VVIX < vvix_upper
      2) Re-enter if VIX in [vix_lower, vix_upper], VVIX in [vvix_lower, vvix_reentry_upper]
      3) Exit otherwise
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

    def exit_condition(vix, vvix):
        return (vix < vix_lower) or (vix > vix_upper) or (vvix < vvix_lower) or (vvix >= vvix_upper)

    def reentry_condition(vix, vvix):
        return (vix_lower <= vix <= vix_upper) and (vvix_lower <= vvix <= vvix_reentry_upper)

    temp_df["Portfolio_Value"] = np.nan
    temp_df.loc[0, "Portfolio_Value"] = initial_investment
    temp_df["In_Market"] = False
    temp_df.loc[0, "In_Market"] = False
    temp_df["Shares_Held"] = 0.0
    temp_df.loc[0, "Shares_Held"] = 0.0
    temp_df["Strategy"] = "No Investment"

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
            if reentry_condition(vix_today, vvix_today):
                cash_available = temp_df.loc[i, "Portfolio_Value"]
                if price_today > 0:
                    shares_bought = cash_available / price_today
                else:
                    shares_bought = 0
                temp_df.loc[i, "Shares_Held"] = shares_bought
                temp_df.loc[i, "In_Market"] = True
                temp_df.loc[i, "Strategy"] = strategy_label_long
                new_val = shares_bought * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
            else:
                temp_df.loc[i, "Strategy"] = "No Investment"

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    temp_df["Portfolio_Value"] = temp_df["Portfolio_Value"].round(2)
    return temp_df


def backtest_strategy_3(df, asset="YMAX", initial_investment=10_000):
    """
    Strategy 3:
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

def backtest_strategy_4(df, asset="YMAX", initial_investment=10_000, zones_params=None):
    """
    Strategy 4: 
      - You can define multiple zones.
      - Each zone may include a VIX range, a VVIX range, both, or neither.
      - If a zone has no indicator ticked, it is skipped (ignored).
      - If only VIX is ticked, that single range is used.
      - If only VVIX is ticked, that single range is used.
      - If both are ticked, both must be satisfied to be in that zone.
      - The strategy goes (or stays) long if the current VIX/VVIX is in ANY of the defined zones.
    """
    temp_df = df.copy()
    temp_df.sort_values("Date", inplace=True)
    temp_df.reset_index(drop=True, inplace=True)

    if asset == "YMAX":
        price_col = "YMAX"
        div_col = "YMAX Dividends"
    else:
        price_col = "YMAG"
        div_col = "YMAG Dividends"

    temp_df["Portfolio_Value"] = np.nan
    temp_df.loc[0, "Portfolio_Value"] = initial_investment
    temp_df["In_Market"] = False
    temp_df.loc[0, "In_Market"] = False
    temp_df["Shares_Held"] = 0.0
    temp_df.loc[0, "Shares_Held"] = 0.0
    temp_df["Strategy"] = "No Investment"

    def in_any_zone(vix, vvix):
        if not zones_params:
            return False

        for zone in zones_params:
            vix_lower = zone["vix_lower"]
            vix_upper = zone["vix_upper"]
            vvix_lower = zone["vvix_lower"]
            vvix_upper = zone["vvix_upper"]

            # Check which indicators are active
            vix_active = (vix_lower is not None and vix_upper is not None)
            vvix_active = (vvix_lower is not None and vvix_upper is not None)

            # If no indicator is active in this zone, skip it
            if not vix_active and not vvix_active:
                continue

            # Check VIX condition if active
            if vix_active and not (vix_lower <= vix <= vix_upper):
                # This zone fails if VIX is active but out of range
                continue

            # Check VVIX condition if active
            if vvix_active and not (vvix_lower <= vvix <= vvix_upper):
                # This zone fails if VVIX is active but out of range
                continue

            # If we reach here, it means all active indicators in this zone are satisfied
            return True

        return False

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
            # If we're in the market and still satisfy at least one zone, remain long
            if in_any_zone(vix_today, vvix_today):
                shares_held = temp_df.loc[i, "Shares_Held"]
                new_val = shares_held * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
                temp_df.loc[i, "Strategy"] = "Long"
            else:
                # Otherwise, exit
                temp_df.loc[i, "In_Market"] = False
                temp_df.loc[i, "Shares_Held"] = 0.0
                temp_df.loc[i, "Strategy"] = "No Investment"
        else:
            # If we're out of the market and we now satisfy at least one zone, enter
            if in_any_zone(vix_today, vvix_today):
                cash_available = temp_df.loc[i, "Portfolio_Value"]
                if price_today > 0:
                    shares_bought = cash_available / price_today
                else:
                    shares_bought = 0
                temp_df.loc[i, "Shares_Held"] = shares_bought
                temp_df.loc[i, "In_Market"] = True
                temp_df.loc[i, "Strategy"] = "Long"
                new_val = shares_bought * (price_today + div_today)
                temp_df.loc[i, "Portfolio_Value"] = new_val
            else:
                temp_df.loc[i, "Strategy"] = "No Investment"

    temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
    return temp_df

# ==============================================
# Rolling Correlations + Performance
# ==============================================
def compute_rolling_correlations(df, window):
    returns = df.loc[:, ~df.columns.str.contains("Dividends")].pct_change()
    corr_df = pd.DataFrame(index=df.index)

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
    """
    Computes overall performance metrics (Total Return, CAGR, Volatility, Sharpe, Max DD, Calmar).
    """
    df = portfolio_df.dropna(subset=["Portfolio_Value"]).copy()
    if len(df) < 2:
        return {}
    start_val = df.iloc[0]["Portfolio_Value"]
    end_val = df.iloc[-1]["Portfolio_Value"]
    total_return = (end_val / start_val - 1) * 100
    num_days = (df.iloc[-1]["Date"] - df.iloc[0]["Date"]).days
    years = num_days / 365 if num_days > 0 else 1
    cagr = ((end_val / start_val) ** (1 / years) - 1) * 100 if years > 0 else np.nan
    # This expects the column "Portfolio_Return" to still exist:
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
    
    fig.update_layout(yaxis=dict(tickformat=".2f"))
    
    return fig

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

# Additional Helpers for Yearly CAGR
def calculate_cagr_for_period(df, start_date, end_date):
    sub = df[(df["Date"] >= start_date) & (df["Date"] <= end_date)].dropna(subset=["Portfolio_Value"])
    if len(sub) < 2:
        return np.nan
    start_val = sub.iloc[0]["Portfolio_Value"]
    end_val = sub.iloc[-1]["Portfolio_Value"]
    total_return = end_val / start_val - 1
    num_days = (sub.iloc[-1]["Date"] - sub.iloc[0]["Date"]).days
    if num_days <= 0:
        return np.nan
    years = num_days / 365
    cagr = (1 + total_return) ** (1 / years) - 1
    return cagr * 100

def get_yearly_cagr(df, year=None, overall=False):
    if overall or year is None:
        start_date = df["Date"].min()
        end_date = df["Date"].max()
        return calculate_cagr_for_period(df, start_date, end_date)
    else:
        start_date = pd.to_datetime(f"{year}-01-01")
        end_date = pd.to_datetime(f"{year}-12-31")
        return calculate_cagr_for_period(df, start_date, end_date)

# ==============================================
# Sidebar Navigation
# ==============================================
page = st.sidebar.radio("Navigation", ["Backtester", "Strategy Overview", "Ratios Zones Indicator", "Ratio-Zones Backtest", "About"])

# ==============================================
# BACKTESTER PAGE
# ==============================================
if page == "Backtester":
    st.title("YMAX YMAG Backtester")

    # Callback to clear previous backtest results when any parameter changes
    def reset_backtest():
        st.session_state["backtest_done"] = False
        st.session_state["ymax_df_final"] = None
        st.session_state["ymag_df_final"] = None

    st.header("Strategy Summaries")
    st.markdown(
        """
**Strategy 1:** Uses VIX/VVIX thresholds and correlation with market volatility 
to determine long positions in YMAX/YMAG—with hedging via QQQ when volatility is high.

**Strategy 2:** Invests only when VIX is between 15 and 20 and VVIX is between 90 and 100, 
exiting if volatility moves outside these ranges. Its default sliders are VIX Lower=15, 
VIX Upper=20, VVIX Lower=90, VVIX Upper=100, and VVIX Re-Entry=95.

**Strategy 3:** Enters whenever VIX < 20 and VVIX < 95, exiting if VIX > 20 or VVIX > 100. 
By default, it uses VIX=20 and VVIX=95, both adjustable via sliders.

**Strategy 4:** Allows you to specify multiple zones for VIX and/or VVIX. For each zone, two columns are displayed.
In the left column the VIX checkbox appears; if ticked, the "Lower" and "Upper" inputs for VIX appear in a sub-row.
Similarly, in the right column the VVIX checkbox appears; if ticked, its "Lower" and "Upper" inputs are shown.
        """
    )
    
    st.subheader("Data Preview")
    st.markdown("This is a preview of the data (complete data) that we are using to run the backtests below.")
    try:
        csv_data = pd.read_csv("All Assets and Dividends.csv")         
        csv_data["Date"] = pd.to_datetime(csv_data["Date"])
        csv_data.set_index("Date", inplace=True)
        st.dataframe(csv_data, height=200)
    except FileNotFoundError:
        st.error("Could not find 'All Assets and Dividends.csv'. Make sure it's in the same folder.")

    st.markdown("""--- 
Below, you can choose which asset(s) to backtest and which strategy to apply. 
Your selections will determine the logic used in the subsequent backtest calculations.
    """)

    col_sel1, col_date, col_freq, col_sel2 = st.columns([1, 1, 1, 1], gap="medium")
    with col_sel1:
        st.subheader("Asset Selection")
        asset_choice = st.radio(
            "Select the asset(s) to backtest:",
            options=["YMAX", "YMAG", "Both"],
            index=2,
            on_change=reset_backtest
        )
        
    with col_date:
        st.subheader("Date Range")
        start_date = st.date_input("Start Date", value=datetime.date(2024, 1, 1), on_change=reset_backtest)
        end_date = st.date_input("End Date", value=datetime.date(2025, 2, 15), on_change=reset_backtest)
        
    with col_freq:
        st.subheader("Frequency")
        frequency = st.selectbox(
            "Select data frequency:",
            options=["Daily", "4 hours", "Hourly", "30 Minutes"],
            index=0,
            on_change=reset_backtest
        )
        
    with col_sel2:
        st.subheader("Strategy Selection")
        strategy_choice = st.radio(
            "Select strategy to backtest:",
            options=["Strategy 1", "Strategy 2", "Strategy 3", "Strategy 4"],
            on_change=reset_backtest
        )
        
    st.markdown("---")
    
    # Strategy 1 parameters
    if strategy_choice == "Strategy 1":
        st.subheader("Parameters for Strategy 1")
        corr_window = st.slider("Select correlation window (days):", min_value=1, max_value=30, value=14, on_change=reset_backtest)
    else:
        corr_window = 14  # default if not Strategy 1
        
    # Strategy 2 parameters
    if strategy_choice == "Strategy 2":
        st.subheader("Parameters for Strategy 2")
        vix_lower = st.slider("VIX Lower", min_value=0, max_value=50, value=15, step=1, on_change=reset_backtest)
        vix_upper = st.slider("VIX Upper", min_value=0, max_value=50, value=20, step=1, on_change=reset_backtest)
        vvix_lower = st.slider("VVIX Lower", min_value=0, max_value=200, value=90, step=1, on_change=reset_backtest)
        vvix_upper = st.slider("VVIX Upper", min_value=0, max_value=200, value=100, step=1, on_change=reset_backtest)
        vvix_reentry_upper = st.slider("VVIX Re-Entry Upper", min_value=0, max_value=200, value=95, step=1, on_change=reset_backtest)
    else:
        vix_lower = 15
        vix_upper = 20
        vvix_lower = 90
        vvix_upper = 100
        vvix_reentry_upper = 95

    # Strategy 3 parameters
    if strategy_choice == "Strategy 3":
        st.subheader("Parameters for Strategy 3")
        vix_threshold = st.slider("Select VIX threshold:", min_value=1, max_value=40, value=20, on_change=reset_backtest)
        vvix_threshold = st.slider("Select VVIX threshold:", min_value=1, max_value=120, value=95, on_change=reset_backtest)
    else:
        vix_threshold = 20
        vvix_threshold = 95

    # Strategy 4 parameters with updated layout (one row for the 4 inputs)
    zones_params = None
    if strategy_choice == "Strategy 4":
        st.subheader("Parameters for Strategy 4")
        zones_count = st.number_input("**Number of Zones in VIX and VVIX to go long:**", min_value=1, max_value=5, 
                                      value=2, step=1, on_change=reset_backtest)
        zones_params = []
        for i in range(1, int(zones_count) + 1):
            st.markdown(f"<p style='font-size:20px; font-weight:bold;'>Zone {i}</p>", unsafe_allow_html=True)
            st.markdown("Select the indicators to use:")
            # Set default values based on zone number
            if i == 1:
                default_vix_lower = 5
                default_vix_upper = 10
                default_vvix_lower = 60
                default_vvix_upper = 70
            elif i == 2:
                default_vix_lower = 15
                default_vix_upper = 20
                default_vvix_lower = 80
                default_vvix_upper = 100
            else:
                default_vix_lower = 0
                default_vix_upper = 0
                default_vvix_lower = 0
                default_vvix_upper = 0

            # Create two columns: one for VIX and one for VVIX
            col_vix, col_vvix = st.columns(2)
            with col_vix:
                use_vix = st.checkbox("VIX", key=f"use_vix_zone_{i}", on_change=reset_backtest)
                if use_vix:
                    subcol1, subcol2 = st.columns(2)
                    with subcol1:
                        zone_vix_lower = st.number_input("Lower", value=default_vix_lower, key=f"vix_lower_zone_{i}", on_change=reset_backtest)
                    with subcol2:
                        zone_vix_upper = st.number_input("Upper", value=default_vix_upper, key=f"vix_upper_zone_{i}", on_change=reset_backtest)
                else:
                    zone_vix_lower = None
                    zone_vix_upper = None
            with col_vvix:
                use_vvix = st.checkbox("VVIX", key=f"use_vvix_zone_{i}", on_change=reset_backtest)
                if use_vvix:
                    subcol3, subcol4 = st.columns(2)
                    with subcol3:
                        zone_vvix_lower = st.number_input("Lower", value=default_vvix_lower, key=f"vvix_lower_zone_{i}", on_change=reset_backtest)
                    with subcol4:
                        zone_vvix_upper = st.number_input("Upper", value=default_vvix_upper, key=f"vvix_upper_zone_{i}", on_change=reset_backtest)
                else:
                    zone_vvix_lower = None
                    zone_vvix_upper = None

            zones_params.append({
                "vix_lower": zone_vix_lower,
                "vix_upper": zone_vix_upper,
                "vvix_lower": zone_vvix_lower,
                "vvix_upper": zone_vvix_upper
            })

    # Run Backtest Button
    run_backtest = st.button("Run Backtest for Selected Strategy")

    if run_backtest:
        # 1) Load CSV file(s) based on asset selection and frequency
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
            df_ymag = df_ymag.drop(columns=["VIX", "VVIX", "QQQ"], errors="ignore")
            all_assets = df_ymax.join(df_ymag, how="inner")

        # 2) Filter by chosen date range
        all_assets = all_assets.loc[str(start_date):str(end_date)]

        # 3) Compute rolling correlations
        ps_df = compute_rolling_correlations(all_assets, corr_window)
        ps_df.reset_index(inplace=True)
        st.session_state["prices_and_stats_df"] = ps_df.copy()

        # 4) Backtest function
        def process_asset(asset_name, strategy):
            if strategy == "Strategy 1":
                return backtest_strategy_1(ps_df, asset=asset_name)
            elif strategy == "Strategy 2":
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
            elif strategy == "Strategy 4":
                return backtest_strategy_4(ps_df, asset=asset_name, zones_params=zones_params)
            else:
                return None

        chosen_strat = strategy_choice

        if asset_choice == "Both":
            ymax_res = process_asset("YMAX", chosen_strat)
            ymag_res = process_asset("YMAG", chosen_strat)
            st.session_state["ymax_df_final"] = ymax_res
            st.session_state["ymag_df_final"] = ymag_res
        else:
            res = process_asset(asset_choice, chosen_strat)
            if asset_choice == "YMAX":
                st.session_state["ymax_df_final"] = res
                st.session_state["ymag_df_final"] = None
            else:
                st.session_state["ymag_df_final"] = res
                st.session_state["ymax_df_final"] = None

        st.session_state["backtest_done"] = True

    # ===========================
    # Display Results if done
    # ===========================
    if st.session_state["backtest_done"]:
        # ---------- YMAX ----------
        if st.session_state["ymax_df_final"] is not None:
            ymax_res = st.session_state["ymax_df_final"]

            st.markdown("## Backtest Results - YMAX")
            c1, c2 = st.columns(2)
            with c1:
                fig_val_ymax = plot_portfolio_value(ymax_res, asset_label="YMAX Strategy")
                st.plotly_chart(fig_val_ymax, use_container_width=True)
            with c2:
                fig_entry_exit_ymax = plot_entry_exit(ymax_res, asset_name="YMAX")
                st.plotly_chart(fig_entry_exit_ymax, use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                fig_dd_ymax = plot_drawdown(ymax_res)
                st.plotly_chart(fig_dd_ymax, use_container_width=True)
            with c4:
                fig_strat_ymax = plot_strategy_distribution(ymax_res)
                st.plotly_chart(fig_strat_ymax, use_container_width=True)

            # Calculate performance metrics
            ymax_metrics = calculate_performance_metrics(ymax_res)
            if ymax_metrics:
                # Remove default CAGR
                if "CAGR (%)" in ymax_metrics:
                    del ymax_metrics["CAGR (%)"]

                # Yearly CAGR dropdown
                earliest_year = ymax_res["Date"].dt.year.min()
                latest_year = ymax_res["Date"].dt.year.max()
                year_options = [str(y) for y in range(earliest_year, latest_year + 1)]
                year_options.append("Overall")

                selected_year_ymax = st.selectbox("Select CAGR Year (YMAX)", year_options, key="ymax_cagr_year")
                if selected_year_ymax == "Overall":
                    cagr_val = get_yearly_cagr(ymax_res, overall=True)
                    cagr_label = "Overall CAGR (%)"
                else:
                    cagr_val = get_yearly_cagr(ymax_res, int(selected_year_ymax), overall=False)
                    cagr_label = f"{selected_year_ymax} CAGR (%)"

                ymax_metrics[cagr_label] = cagr_val
                df_ymax_perf = pd.DataFrame([ymax_metrics], index=["YMAX Strategy"]).round(2)
                st.dataframe(df_ymax_perf)
            else:
                st.info("Not enough data points for YMAX metrics.")

            # Create a copy for display, adding Portfolio_Return (%)
            ymax_display = ymax_res.copy()
            ymax_display["Portfolio_Return (%)"] = (ymax_display["Portfolio_Return"] * 100).round(2)

            st.subheader("YMAX Trading Results (Data)")
            st.dataframe(ymax_display, height=300)

        # ---------- YMAG ----------
        if st.session_state["ymag_df_final"] is not None:
            ymag_res = st.session_state["ymag_df_final"]

            st.markdown("## Backtest Results - YMAG")
            c1, c2 = st.columns(2)
            with c1:
                fig_val_ymag = plot_portfolio_value(ymag_res, asset_label="YMAG Strategy")
                st.plotly_chart(fig_val_ymag, use_container_width=True, key="fig_val_ymag")
            with c2:
                fig_entry_exit_ymag = plot_entry_exit(ymag_res, asset_name="YMAG")
                st.plotly_chart(fig_entry_exit_ymag, use_container_width=True, key="fig_entry_exit_ymag")

            c3, c4 = st.columns(2)
            with c3:
                fig_dd_ymag = plot_drawdown(ymag_res)
                st.plotly_chart(fig_dd_ymag, use_container_width=True, key="fig_dd_ymag")
            with c4:
                fig_strat_ymag = plot_strategy_distribution(ymag_res)
                st.plotly_chart(fig_strat_ymag, use_container_width=True, key="fig_strat_ymag")

            # Calculate performance metrics
            ymag_metrics = calculate_performance_metrics(ymag_res)
            if ymag_metrics:
                if "CAGR (%)" in ymag_metrics:
                    del ymag_metrics["CAGR (%)"]

                earliest_year = ymag_res["Date"].dt.year.min()
                latest_year = ymag_res["Date"].dt.year.max()
                year_options = [str(y) for y in range(earliest_year, latest_year + 1)]
                year_options.append("Overall")

                selected_year_ymag = st.selectbox("Select CAGR Year (YMAG)", year_options, key="ymag_cagr_year")
                if selected_year_ymag == "Overall":
                    cagr_val = get_yearly_cagr(ymag_res, overall=True)
                    cagr_label = "Overall CAGR (%)"
                else:
                    cagr_val = get_yearly_cagr(ymag_res, int(selected_year_ymag), overall=False)
                    cagr_label = f"{selected_year_ymag} CAGR (%)"

                ymag_metrics[cagr_label] = cagr_val
                df_ymag_perf = pd.DataFrame([ymag_metrics], index=["YMAG Strategy"]).round(2)
                st.dataframe(df_ymag_perf)
            else:
                st.info("Not enough data points for YMAG metrics.")

            # Create a copy for display, adding Portfolio_Return (%)
            ymag_display = ymag_res.copy()
            ymag_display["Portfolio_Return (%)"] = (ymag_display["Portfolio_Return"] * 100).round(2)

            st.subheader("YMAG Trading Results (Data)")
            st.dataframe(ymag_display, height=300)

# ==============================================
# STRATEGY OVERVIEW PAGE
# ==============================================
elif page == "Strategy Overview":
    strategy_overview.display_strategy_overview()

# ==============================================
# RATIOS ZONES INDICATOR
# ==============================================
elif page == "Ratios Zones Indicator":
    display_ratios_zones_indicator()

# ==============================================
# RATIO-ZONES BACKTEST
# ==============================================
elif page == "Ratio-Zones Backtest":
    display_ratios_zones_backtest()

# ==============================================
# ABOUT PAGE
# ==============================================
elif page == "About":
    st.title("About")
    st.write("This app allows you to backtest trading strategies on YMAX and YMAG assets using historical data. Features include:")
    st.write("- **Asset Selection**: Choose YMAX, YMAG, or both.")
    st.write("- **Strategy Selection**: Test four strategies with customizable parameters.")
    st.write("- **Interactive Plots**: Visualize performance, entry/exit points, strategy distribution, and drawdowns using Plotly.")
    st.write("- **Performance Metrics**: View Total Return, CAGR, Volatility, Sharpe Ratio, Max Drawdown, and Calmar Ratio.")

    st.markdown("""
        <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            text-align: center;
            padding: 10px;
            border-top: 1px solid #e3e3e3;
        }
        </style>
        <div class="footer">
            Developed by: <a href="https://www.upwork.com/freelancers/~01c4c80cfb55ca8c0d?mp_source=share">Benson</a>
        </div>
        """, unsafe_allow_html=True
    )
