import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objs as go

# ==============================================================================
# Session-State Setup for Ratio-Zones Backtest
# ==============================================================================
# We store the backtest results in session state so that changing the CAGR year
# doesn't force the user to re-run the entire backtest.
if "zones_res" not in st.session_state:
    st.session_state["zones_res"] = None
if "zones_done" not in st.session_state:
    st.session_state["zones_done"] = None

# ==============================================================================
# Main Function: display_ratios_zones_backtest()
# ==============================================================================
def display_ratios_zones_backtest():
    st.title("Ratio-Zones Backtest")
    st.markdown("""
    **Instructions**:
    1. Choose your **Asset**, **Date Range**, **Frequency**, and **Sub-Strategy** below.
    2. Define your ratio zones for taking long (and short, if applicable) positions.
    3. For sub-strategy 4.2 (Zones + MA Filter), specify the moving average windows and choose whether to exit when the filter flips.
    4. Click **Run Backtest** to view results, charts, and performance metrics.
    ---
    """)

    # ----------------------------
    # 1) Data Directory & Helpers
    # ----------------------------
    # data_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data\TradingView Data"
    data_dir = "data"


    def get_filename(asset, timeframe):
        tf_map = {
            "Daily": "Daily",
            "4H": "4H",
            "1H": "Hourly",
            "30M": "30Mins"
        }
        tf_suffix = tf_map.get(timeframe, "Daily")
        if asset in ["YMAX", "YMAG"]:
            return f"{asset}_VIX_VVIX_QQQ_{timeframe}.csv"
        elif asset == "QQQ":
            return f"QQQ_VIX_VVIX_{tf_suffix}.csv"
        elif asset == "GLD":
            return f"GLD_VIX_VVIX_{tf_suffix}.csv"

    @st.cache_data
    def load_data(filepath):
        df_temp = pd.read_csv(filepath)
        if "Dates" in df_temp.columns and "Date" not in df_temp.columns:
            df_temp.rename(columns={"Dates": "Date"}, inplace=True)
        df_temp["Date"] = pd.to_datetime(df_temp["Date"], errors="coerce")
        df_temp.sort_values("Date", inplace=True)
        return df_temp

    def in_any_zone(ratio, zones):
        if not zones:
            return False
        for (low, high) in zones:
            if low <= ratio <= high:
                return True
        return False

    # -------------------------------------------
    # 2) Backtest Strategy 4 (with or w/o MA filter)
    # -------------------------------------------
    def backtest_strategy_4(
        df,
        asset="YMAX",
        initial_investment=10_000,
        long_entry_zones=None,
        short_entry_zones=None,
        variant="4.1",  # "4.1" = zones-only; "4.2" = zones + MA filter
        ma_short_window=None,
        ma_long_window=None,
        exit_on_filter_flip=True
    ):
        """
        variant="4.2": ratio-based zones + moving average filter
        short_entry_zones: relevant if asset can be shorted (e.g., QQQ, GLD)
        """
        temp_df = df.copy()
        temp_df.sort_values("Date", inplace=True)
        temp_df.reset_index(drop=True, inplace=True)

        if long_entry_zones is None:
            long_entry_zones = []
        if short_entry_zones is None:
            short_entry_zones = []

        price_col = asset
        div_col = f"{asset} Dividends" if f"{asset} Dividends" in temp_df.columns else None

        # Compute VVIX/VIX ratio
        temp_df["VVIX/VIX Ratio"] = (temp_df["VVIX"] / temp_df["VIX"]).round(4)

        # If variant 4.2, add moving average filter
        if variant == "4.2" and ma_short_window and ma_long_window:
            temp_df["MA_short"] = temp_df[price_col].rolling(window=ma_short_window, min_periods=1).mean()
            temp_df["MA_long"] = temp_df[price_col].rolling(window=ma_long_window, min_periods=1).mean()

            def get_filter(ma_s, ma_l):
                if ma_s > ma_l:
                    return "Bullish"
                elif ma_s < ma_l:
                    return "Bearish"
                else:
                    return "Neutral"

            temp_df["Filter_Signal"] = temp_df.apply(
                lambda row: get_filter(row["MA_short"], row["MA_long"]), axis=1
            )

        # Build ratio-based signals
        ratio_signals = []
        for _, row in temp_df.iterrows():
            ratio_val = row["VVIX/VIX Ratio"] if not np.isnan(row["VVIX/VIX Ratio"]) else 0
            if asset in ["YMAX", "YMAG"]:
                # Only "Long" or "No Investment" for YMAX/YMAG
                if in_any_zone(ratio_val, long_entry_zones):
                    ratio_signals.append("Long")
                else:
                    ratio_signals.append("No Investment")
            else:
                # For QQQ, GLD, etc. we can do "Long", "Short", or "No Investment"
                if in_any_zone(ratio_val, long_entry_zones):
                    ratio_signals.append("Long")
                elif in_any_zone(ratio_val, short_entry_zones):
                    ratio_signals.append("Short")
                else:
                    ratio_signals.append("No Investment")

        temp_df["Ratio_Signal"] = ratio_signals

        # Combine ratio signals with MA filter if variant 4.2
        final_signals = []
        for i, row in temp_df.iterrows():
            ratio_sig = row["Ratio_Signal"]
            if variant == "4.1":
                final_signals.append(ratio_sig)
            else:
                filter_sig = row.get("Filter_Signal", "Neutral")
                if i == 0:
                    final_signals.append("No Investment")
                else:
                    prev_final = final_signals[i - 1]
                    if ratio_sig == "Long":
                        if exit_on_filter_flip:
                            final_signals.append("Long" if filter_sig == "Bullish" else "No Investment")
                        else:
                            final_signals.append("Long" if (prev_final == "Long" or filter_sig == "Bullish") else "No Investment")
                    elif ratio_sig == "Short" and asset not in ["YMAX", "YMAG"]:
                        if exit_on_filter_flip:
                            final_signals.append("Short" if filter_sig == "Bearish" else "No Investment")
                        else:
                            final_signals.append("Short" if (prev_final == "Short" or filter_sig == "Bearish") else "No Investment")
                    else:
                        final_signals.append("No Investment")

        temp_df["Desired_Strategy"] = final_signals

        # Initialize portfolio
        temp_df["Strategy"] = "No Investment"
        temp_df["Portfolio_Value"] = np.nan
        temp_df.loc[0, "Portfolio_Value"] = initial_investment
        temp_df["Shares_Held"] = 0.0
        temp_df["Shares_Short"] = 0.0

        for i in range(1, len(temp_df)):
            prev_val = temp_df.loc[i - 1, "Portfolio_Value"]
            temp_df.loc[i, "Portfolio_Value"] = prev_val
            current_desired = temp_df.loc[i, "Desired_Strategy"]
            prev_strategy = temp_df.loc[i - 1, "Strategy"]

            y_price_yest = temp_df.loc[i - 1, price_col]
            y_price_today = temp_df.loc[i, price_col]
            div_today = temp_df.loc[i, div_col] if div_col else 0.0

            prev_shares_held = temp_df.loc[i - 1, "Shares_Held"]
            prev_shares_short = temp_df.loc[i - 1, "Shares_Short"]

            temp_df.loc[i, "Shares_Held"] = prev_shares_held
            temp_df.loc[i, "Shares_Short"] = prev_shares_short

            if current_desired == "Long":
                if prev_strategy != "Long":
                    # Enter Long
                    shares_bought = prev_val / y_price_yest if y_price_yest > 0 else 0
                    temp_df.loc[i, "Shares_Held"] = shares_bought
                    temp_df.loc[i, "Shares_Short"] = 0.0
                    temp_df.loc[i, "Strategy"] = "Long"
                    temp_df.loc[i, "Portfolio_Value"] = shares_bought * (y_price_today + div_today)
                else:
                    # Remain Long
                    shares_held = prev_shares_held
                    temp_df.loc[i, "Portfolio_Value"] = shares_held * (y_price_today + div_today)
                    temp_df.loc[i, "Strategy"] = "Long"

            elif current_desired == "Short" and asset not in ["YMAX", "YMAG"]:
                if prev_strategy != "Short":
                    # Enter Short
                    shares_short = prev_val / y_price_yest if y_price_yest > 0 else 0
                    temp_df.loc[i, "Shares_Short"] = shares_short
                    temp_df.loc[i, "Shares_Held"] = 0.0
                    temp_df.loc[i, "Strategy"] = "Short"
                    short_proceeds = shares_short * y_price_yest
                    daily_pnl = shares_short * (y_price_yest - y_price_today)
                    temp_df.loc[i, "Portfolio_Value"] = short_proceeds + daily_pnl
                else:
                    # Remain Short
                    shares_short = prev_shares_short
                    short_proceeds = shares_short * y_price_yest
                    daily_pnl = shares_short * (y_price_yest - y_price_today)
                    temp_df.loc[i, "Portfolio_Value"] = short_proceeds + daily_pnl
                    temp_df.loc[i, "Strategy"] = "Short"

            else:
                # No Investment
                if prev_strategy == "Long":
                    # Exit from Long
                    shares_held = prev_shares_held
                    exit_val = shares_held * y_price_today
                    temp_df.loc[i, "Portfolio_Value"] = exit_val
                    temp_df.loc[i, "Shares_Held"] = 0.0
                    temp_df.loc[i, "Strategy"] = "No Investment"
                elif prev_strategy == "Short":
                    # Exit from Short
                    shares_short = prev_shares_short
                    short_proceeds = shares_short * y_price_yest
                    daily_pnl = shares_short * (y_price_yest - y_price_today)
                    exit_val = short_proceeds + daily_pnl
                    temp_df.loc[i, "Portfolio_Value"] = exit_val
                    temp_df.loc[i, "Shares_Short"] = 0.0
                    temp_df.loc[i, "Strategy"] = "No Investment"
                else:
                    temp_df.loc[i, "Strategy"] = "No Investment"

        temp_df["Portfolio_Return"] = temp_df["Portfolio_Value"].pct_change()
        return temp_df

    # ---------------------------------------
    # 3) Performance & Plotting Functions
    # ---------------------------------------
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
            "CAGR (%)": cagr,  # We'll remove or replace with year-based
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
            title=f"Ratios Strategy - {asset_label}",
            labels={"Portfolio_Value": "Portfolio Value ($)"},
        )
        fig.update_traces(
            line=dict(width=2),
            hovertemplate="Date=%{x}<br>Portfolio Value=$%{y:.2f}"
        )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Value ($)",
            yaxis=dict(tickformat=".2f")
        )
        return fig

    def plot_strategy_distribution(df):
        if "Strategy" not in df.columns:
            df["Strategy"] = "No Investment"
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
            hovertemplate="Date=%{x}<br>Drawdown=%{y:.2f}%<extra></extra>"
        )
        fig.update_layout(yaxis=dict(tickformat=".2f"))
        return fig

    def plot_entry_exit(df, asset_name="YMAX"):
        temp_df = df.copy()
        if "In_Market" not in temp_df.columns:
            temp_df["In_Market"] = temp_df["Strategy"].apply(lambda x: x.startswith("Long"))
        temp_df["Entry"] = (temp_df["In_Market"].shift(1) == False) & (temp_df["In_Market"] == True)
        temp_df["Exit"] = (temp_df["In_Market"].shift(1) == True) & (temp_df["In_Market"] == False)
        entry_days = temp_df[temp_df["Entry"] == True]
        exit_days = temp_df[temp_df["Exit"] == True]

        fig = go.Figure()
        # Asset price line
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
        # Entry markers
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
        # Exit markers
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
        # Portfolio value line
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
        # If we have MAs
        if "MA_short" in temp_df.columns and "MA_long" in temp_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=temp_df["Date"],
                    y=temp_df["MA_short"],
                    mode="lines",
                    line=dict(color="magenta", width=1, dash="dash"),
                    name="Short MA"
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=temp_df["Date"],
                    y=temp_df["MA_long"],
                    mode="lines",
                    line=dict(color="cyan", width=1, dash="dot"),
                    name="Long MA"
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

    # ---------------------------------------
    # Year-based CAGR Helpers
    # ---------------------------------------
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

    # ---------------------------------------
    # 4) UI Layout
    # ---------------------------------------
    col_sel1, col_sel2, col_sel3, col_sel4 = st.columns([1, 2, 1, 1], gap="large")

    with col_sel1:
        st.subheader("Asset")
        asset_option = st.radio("Select the asset:", ["YMAX", "YMAG", "QQQ", "GLD"], index=2)

    with col_sel2:
        st.subheader("Date Range")
        # Get earliest & latest from the dataset after loading
        # We'll load after user picks asset/timeframe below
        # But let's hold placeholders
        from_date_placeholder = st.empty()
        to_date_placeholder = st.empty()

    with col_sel3:
        st.subheader("Frequency")
        timeframe_option = st.selectbox("Data frequency:", ["Daily", "4H", "1H", "30M"], index=0)

    with col_sel4:
        st.subheader("Sub-Strategy")
        sub_strategy = st.radio("Choose Strategy", ["4.1 - Use Zones Only", "4.2 - Zones + MA Filter"])

    # Load data based on chosen asset/timeframe
    filename = get_filename(asset_option, timeframe_option)
    file_path = os.path.join(data_dir, filename)
    try:
        df_raw = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading data for {asset_option}/{timeframe_option}: {e}")
        return  # Stop if error

    if df_raw.empty:
        st.warning("No data found in the CSV.")
        return

    earliest_date = df_raw["Date"].iloc[0].date()
    latest_date = df_raw["Date"].iloc[-1].date()

    with col_sel2:
        from_date = from_date_placeholder.date_input(
            "Start Date",
            value=earliest_date,
            min_value=earliest_date,
            max_value=latest_date
        )
        to_date = to_date_placeholder.date_input(
            "End Date",
            value=latest_date,
            min_value=earliest_date,
            max_value=latest_date
        )

    st.subheader("Define Ratio Zones")
    st.markdown("""
    Specify the zones in which you want to take long (and short, if applicable) positions.
    Positions are closed when the ratio moves outside these intervals.
    """)

    def get_zones_input(label_prefix):
        num_zones = st.number_input(
            f"Number of {label_prefix} Zones",
            min_value=0,
            max_value=5,
            value=1,
            step=1
        )
        zones = []
        for i in range(num_zones):
            c1, c2 = st.columns(2)
            with c1:
                low = st.number_input(
                    f"{label_prefix} Zone {i+1} - Lower",
                    value=1.0000,
                    step=0.0001,
                    format="%.4f",
                    key=f"{label_prefix}_low_{i}"
                )
            with c2:
                high = st.number_input(
                    f"{label_prefix} Zone {i+1} - Upper",
                    value=2.0000,
                    step=0.0001,
                    format="%.4f",
                    key=f"{label_prefix}_high_{i}"
                )
            zones.append((low, high))
        return zones

    col_zone1, col_zone2 = st.columns(2)
    with col_zone1:
        st.markdown("#### Long Entry Zones")
        long_zones = get_zones_input("Long Entry")

    short_zones = []
    if asset_option in ["QQQ", "GLD"]:
        with col_zone2:
            st.markdown("#### Short Entry Zones")
            short_zones = get_zones_input("Short Entry")

    # Additional Inputs for MA Filter (if Sub-Strategy 4.2)
    if sub_strategy.startswith("4.2"):
        st.subheader("Moving Average Filter Parameters")
        ma_short_window = st.number_input("Short MA Window", min_value=1, max_value=200, value=20, step=1)
        ma_long_window = st.number_input("Long MA Window", min_value=1, max_value=500, value=50, step=1)
        exit_filter = st.radio("Exit if filter flips (when ratio remains in zone)?", ["Yes", "No"], index=0)
        exit_on_filter_flip = (exit_filter == "Yes")
    else:
        ma_short_window = None
        ma_long_window = None
        exit_on_filter_flip = True

    initial_cap = st.number_input("Initial Investment ($)", min_value=1000, max_value=1_000_000, value=10000, step=1000)

    # Button to run the backtest
    run_btn = st.button("Run Backtest")

    if run_btn:
        # Filter data by chosen date range
        mask = (df_raw["Date"] >= pd.to_datetime(from_date)) & (df_raw["Date"] <= pd.to_datetime(to_date))
        df_filtered = df_raw.loc[mask].copy()

        if df_filtered.empty:
            st.warning("No data in the selected date range.")
            return

        # Backtest
        variant = "4.1" if sub_strategy.startswith("4.1") else "4.2"
        res_df = backtest_strategy_4(
            df_filtered,
            asset=asset_option,
            initial_investment=initial_cap,
            long_entry_zones=long_zones,
            short_entry_zones=short_zones,
            variant=variant,
            ma_short_window=ma_short_window,
            ma_long_window=ma_long_window,
            exit_on_filter_flip=exit_on_filter_flip
        )

        # If results are empty
        if res_df.empty:
            st.warning("No backtest results returned.")
            return

        # Store results in session state
        st.session_state["zones_res"] = res_df
        st.session_state["zones_done"] = True

    # ------------------------------------------------
    # 5) Display Results if we have them in session
    # ------------------------------------------------
    if st.session_state["zones_done"] and st.session_state["zones_res"] is not None:
        res_df = st.session_state["zones_res"]  # Retrieve from session

        st.subheader("Backtest Results")
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig_val = plot_portfolio_value(res_df, asset_label=asset_option)
            st.plotly_chart(fig_val, use_container_width=True)
        with col_chart2:
            fig_ex = plot_entry_exit(res_df, asset_name=asset_option)
            st.plotly_chart(fig_ex, use_container_width=True)

        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            fig_dd = plot_drawdown(res_df)
            st.plotly_chart(fig_dd, use_container_width=True)
        with col_chart4:
            fig_strat = plot_strategy_distribution(res_df)
            st.plotly_chart(fig_strat, use_container_width=True)

        # Compute performance metrics
        metrics = calculate_performance_metrics(res_df)
        if metrics:
            # Remove default CAGR (%) so we can replace with year-based
            if "CAGR (%)" in metrics:
                del metrics["CAGR (%)"]

            st.subheader("Performance Metrics")
            earliest_year = res_df["Date"].dt.year.min()
            latest_year = res_df["Date"].dt.year.max()
            year_options = [str(y) for y in range(earliest_year, latest_year + 1)]
            year_options.append("Overall")

            selected_year = st.selectbox("Select CAGR Year", year_options, key="zones_cagr_year")
            if selected_year == "Overall":
                cagr_val = get_yearly_cagr(res_df, overall=True)
                cagr_label = "Overall CAGR (%)"
            else:
                cagr_val = get_yearly_cagr(res_df, int(selected_year), overall=False)
                cagr_label = f"{selected_year} CAGR (%)"

            metrics[cagr_label] = cagr_val
            perf_df = pd.DataFrame([metrics]).round(2)
            st.dataframe(perf_df)
        else:
            st.info("Not enough data to compute performance metrics.")

        # Detailed Results
        st.subheader("Detailed Results")
        show_df = res_df.copy()
        # Create a separate column for percentage returns
        show_df["Portfolio_Return (%)"] = (show_df["Portfolio_Return"] * 100).round(2)
        st.dataframe(show_df, height=400, use_container_width=True)

    else:
        st.info("Configure your parameters and click 'Run Backtest' to see results.")
