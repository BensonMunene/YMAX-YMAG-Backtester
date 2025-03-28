def display_ratios_zones_indicator():
    import warnings
    import logging

    # We wrap set_page_config in a try/except so it doesn't crash
    # if it's already been called in another page or earlier in the script.
    import streamlit as st
    from streamlit.errors import StreamlitAPIException
    try:
        st.set_page_config(
            page_title="Multi-Asset Price Analysis by VVIX/VIX Ratio",
            layout="wide"
        )
    except StreamlitAPIException:
        pass

    # Suppress specific Streamlit warnings regarding missing ScriptRunContext and session state
    warnings.filterwarnings("ignore", message="Thread 'MainThread': missing ScriptRunContext!")
    warnings.filterwarnings("ignore", message="Session state does not function when running a script without `streamlit run`")

    # Set the logging level for Streamlit's scriptrunner to ERROR to reduce warnings
    logging.getLogger("streamlit.runtime.scriptrunner").setLevel(logging.ERROR)

    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from matplotlib.dates import DateFormatter

    # -----------------------------------------------------------------------------
    # 2. Seaborn Theme for Aesthetics
    # -----------------------------------------------------------------------------
    sns.set_theme(style="whitegrid", context="talk")

    # -----------------------------------------------------------------------------
    # 3. Main Title & Brief Description
    # -----------------------------------------------------------------------------
    st.title("Multi-Asset Price Analysis by VVIX/VIX Ratio")
    st.markdown("""
    This app analyzes price data for multiple assets by computing the **VVIX/VIX** ratio.  
    The ratio is **floored** to create integer segments.
    """)

    # -----------------------------------------------------------------------------
    # 4. Asset and Timeframe Selection
    # -----------------------------------------------------------------------------
    st.markdown("#### Select Asset")
    st.markdown("Choose from **YMAX**, **YMAG**, **QQQ**, **QYLD**, **LQD**, **TLT**, or **GLD**.")
    asset_option = st.selectbox("Select Asset", ["YMAX", "YMAG", "QQQ", "QYLD", "LQD", "TLT", "GLD"])

    st.markdown("#### Select Time Frequency of the Asset Data")
    st.markdown("""
    Choose the desired frequency of the data: 
    - **Daily** Frequency Data
    - **4H** (4 hours frequency)
    - **1H** (1 hour frequency)
    - **30M** (30 minutes frequency)
    """)
    timeframe_option = st.selectbox("Select Timeframe", ["Daily", "4H", "1H", "30M"])

# -----------------------------------------------------------------------------
# 5. Custom Zone Settings
# -----------------------------------------------------------------------------
    st.markdown("### Custom Zone Settings")
    st.markdown("""
    If you want to group the computed (floored or raw) ratios into custom zones and assign your own colors, 
    please enable the option below. For example, you might want to combine ratios 1.0–2.0 as one zone 
    (with a chosen color), 3.0–5.0 as another, etc.
    """)
    use_custom = st.checkbox("Use custom zone grouping", value=True)

    if use_custom:
        # Keep this as an integer input for the number of zones
        num_zones = st.number_input(
            "Number of custom zones",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="How many distinct custom zones you want to define"
        )

        custom_zones = []
        st.markdown("#### Define each custom zone:")

        for i in range(num_zones):
            col1, col2, col3 = st.columns(3)

            # Convert default bounds to floats
            default_lower = float(1) if i == 0 else (3.0 if i == 1 else (6.0 if i == 2 else 0.0))
            default_upper = float(2) if i == 0 else (5.0 if i == 1 else (8.0 if i == 2 else 10.0))
            default_color = (
                "#32CD32" if i == 0 else
                ("#DC143C" if i == 1 else
                 ("#1E90FF" if i == 2 else "#AAAAAA"))
            )

            with col1:
                lower_bound = st.number_input(
                    f"Zone {i+1} lower bound",
                    min_value = 0.0000,
                    max_value = 10.0000,
                    value=default_lower,
                    step=0.0001,
                    format="%.4f",
                    key=f"zone_{i}_lower"
                )

            with col2:
                upper_bound = st.number_input(
                    f"Zone {i+1} upper bound",
                    min_value=lower_bound,
                    max_value=10.0000,
                    value=default_upper,
                    step=0.0001,
                    format="%.4f",
                    key=f"zone_{i}_upper"
                )

            with col3:
                zone_color = st.color_picker(
                    f"Zone {i+1} color",
                    value=default_color,
                    key=f"zone_{i}_color"
                )

            custom_zones.append({
                "lower": lower_bound,
                "upper": upper_bound,
                "color": zone_color
            })
    else:
        # Define a default color mapping if custom grouping is not used
        color_map = {
            0: 'grey',
            1: 'limegreen',
            2: 'crimson',
            3: 'dodgerblue',
            4: 'gold',
            5: 'mediumpurple',
            6: 'sienna',
            7: 'black',
            8: 'deeppink',
            9: 'darkolivegreen',
            10: 'cyan'
        }

    # -----------------------------------------------------------------------------
    # 6. Filename Logic (Assuming CSV files are in the same directory as the script)
    # -----------------------------------------------------------------------------
    def get_filename(asset, timeframe):
        """
        Return the CSV filename for the chosen asset & timeframe.
        
        - YMAX / YMAG: old naming => {asset}_VIX_VVIX_QQQ_{timeframe}.csv
        - QQQ: new naming => QQQ_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
        - GLD: new naming => GLD_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
        - QYLD: new naming => QYLD_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
        - LQD: new naming => LQD_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
        - TLT: new naming => TLT_VIX_VVIX_{Daily|4H|Hourly|30Mins}.csv
        """
        if asset in ["YMAX", "YMAG"]:
            return f"{asset}_VIX_VVIX_QQQ_{timeframe}.csv"
        elif asset == "QQQ":
            if timeframe == "Daily":
                return "QQQ_VIX_VVIX_Daily.csv"
            elif timeframe == "4H":
                return "QQQ_VIX_VVIX_4H.csv"
            elif timeframe == "1H":
                return "QQQ_VIX_VVIX_Hourly.csv"
            elif timeframe == "30M":
                return "QQQ_VIX_VVIX_30Mins.csv"
        elif asset == "GLD":
            if timeframe == "Daily":
                return "GLD_VIX_VVIX_Daily.csv"
            elif timeframe == "4H":
                return "GLD_VIX_VVIX_4H.csv"
            elif timeframe == "1H":
                return "GLD_VIX_VVIX_Hourly.csv"
            elif timeframe == "30M":
                return "GLD_VIX_VVIX_30Mins.csv"
        elif asset == "QYLD":
            if timeframe == "Daily":
                return "QYLD_VIX_VVIX_Daily.csv"
            elif timeframe == "4H":
                return "QYLD_VIX_VVIX_4H.csv"
            elif timeframe == "1H":
                return "QYLD_VIX_VVIX_Hourly.csv"
            elif timeframe == "30M":
                return "QYLD_VIX_VVIX_30Mins.csv"
        elif asset == "LQD":
            if timeframe == "Daily":
                return "LQD_VIX_VVIX_Daily.csv"
            elif timeframe == "4H":
                return "LQD_VIX_VVIX_4H.csv"
            elif timeframe == "1H":
                return "LQD_VIX_VVIX_Hourly.csv"
            elif timeframe == "30M":
                return "LQD_VIX_VVIX_30Mins.csv"
        elif asset == "TLT":
            if timeframe == "Daily":
                return "TLT_VIX_VVIX_Daily.csv"
            elif timeframe == "4H":
                return "TLT_VIX_VVIX_4H.csv"
            elif timeframe == "1H":
                return "TLT_VIX_VVIX_Hourly.csv"
            elif timeframe == "30M":
                return "TLT_VIX_VVIX_30Mins.csv"    

    filename = get_filename(asset_option, timeframe_option)
    
    # -------------------------------------------------------------------------
    # 6a. Local Path vs. GitHub
    # -------------------------------------------------------------------------
    # FOR LOCAL USAGE (Benson's laptop):
    data_dir = r"D:\Benson\aUpWork\Douglas Backtester Algo\Backtester Algorithm\Data\TradingView Data"
    file_path = os.path.join(data_dir, filename)

    # FOR GITHUB USAGE: 
    # - Comment out the two lines above 
    # - Uncomment the lines below:
    # data_dir = "data"
    # file_path = os.path.join(data_dir, filename)

    @st.cache_data
    def load_data(path):
        df_temp = pd.read_csv(path)
        # If the CSV has "Dates" instead of "Date", rename it
        if 'Dates' in df_temp.columns and 'Date' not in df_temp.columns:
            df_temp.rename(columns={'Dates': 'Date'}, inplace=True)
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')
        df_temp.sort_values('Date', inplace=True)
        return df_temp

    try:
        df = load_data(file_path)
    except Exception as e:
        st.error(f"Error loading data from {file_path}: {e}")
        st.stop()

    # -----------------------------------------------------------------------------
    # 7. Determine Default Date Range Based on Data
    # -----------------------------------------------------------------------------
    if df.empty:
        st.warning("No data available for this asset/timeframe combination.")
        st.stop()

    price_column = asset_option
    if price_column not in df.columns:
        st.error(f"Column '{price_column}' not found in the dataset. Columns found: {list(df.columns)}")
        st.stop()

    earliest_date = df['Date'].iloc[0].date()
    latest_date   = df['Date'].iloc[-1].date()

    # -----------------------------------------------------------------------------
    # 8. Date Range Inputs
    # -----------------------------------------------------------------------------
    st.markdown("### Specify a Date Range for the Zones Indicator")
    col1, col2 = st.columns(2)

    with col1:
        from_date = st.date_input(
            "From Date",
            value=earliest_date,
            min_value=earliest_date,
            max_value=latest_date,
            help="Select the start date (automatically set to the earliest date in the dataset)."
        )
    with col2:
        to_date = st.date_input(
            "To Date",
            value=latest_date,
            min_value=earliest_date,
            max_value=latest_date,
            help="Select the end date (automatically set to the latest date in the dataset)."
        )

    # -----------------------------------------------------------------------------
    # 9. Button to Generate Zones Indicator
    # -----------------------------------------------------------------------------
    st.markdown("""
    You have successfully specified your asset, timeframe, date range, and (if enabled) your custom zone settings.  
    Please click the button below to generate the zones indicator plot **using your chosen options**.
    """)
    generate_button = st.button("Generate the zones indicator")

    if generate_button:
        # Filter data to the chosen date range
        mask = (df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))
        df_filtered = df.loc[mask].copy()

        if df_filtered.empty:
            st.warning("No data available in the specified date range.")
            st.stop()

        # Compute ratio (VVIX / VIX), handle infinities and NaNs
        df_filtered['ratio'] = df_filtered['VVIX'] / df_filtered['VIX']
        df_filtered['ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        df_filtered['ratio'].fillna(0, inplace=True)

        # Create floored ratio column
        df_filtered['ratio_int'] = np.floor(df_filtered['ratio']).astype(int)

        if use_custom:
            # Define a function to get the custom zone label and color for a given ratio value
            def get_custom_zone_value(x, zones):
                for zone in zones:
                    if zone["lower"] <= x <= zone["upper"]:
                        return f"{zone['lower']}-{zone['upper']}", zone["color"]
                return ("No Zone", "grey")
            
            zone_info = df_filtered['ratio'].apply(lambda x: get_custom_zone_value(x, custom_zones))
            df_filtered['custom_zone'] = zone_info.apply(lambda x: x[0])
            df_filtered['custom_color'] = zone_info.apply(lambda x: x[1])
            # Group by contiguous changes in the custom zone label
            df_filtered['zone_group'] = (df_filtered['custom_zone'] != df_filtered['custom_zone'].shift(1)).cumsum()
        else:
            # Use the default grouping based on ratio_int changes
            df_filtered['change_id'] = (df_filtered['ratio_int'] != df_filtered['ratio_int'].shift(1)).cumsum()

        min_price = df_filtered[price_column].min()
        max_price = df_filtered[price_column].max()

        # -----------------------------------------------------------------------------
        # 10. Plotly Date/Time Formatting
        # -----------------------------------------------------------------------------
        if timeframe_option == "Daily":
            xaxis_dtick = "M1"
            xaxis_tickformat = "%b %Y"
            xaxis_hoverformat = "%b %d, %Y"
        else:
            xaxis_dtick = None
            xaxis_tickformat = "%b %d %H:%M"
            xaxis_hoverformat = "%b %d, %Y %H:%M"

        # -----------------------------------------------------------------------------
        # 11. Create the Plotly Figure
        # -----------------------------------------------------------------------------
        def create_plotly_figure(df_data):
            plotly_title = f"{asset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio"
            fig = make_subplots(rows=1, cols=1)

            if use_custom:
                groups = list(df_data.groupby('zone_group'))
            else:
                groups = list(df_data.groupby('change_id'))

            added_legends = set()

            for i, (grp_id, group_original) in enumerate(groups):
                group_original = group_original.copy()
                start_date = group_original['Date'].iloc[0]
                end_date   = group_original['Date'].iloc[-1]
                
                # For contiguous segments, skip the first row for boundary duplication (except for the first group)
                if i == 0:
                    line_data = group_original
                else:
                    line_data = group_original.iloc[1:]
                
                if use_custom:
                    label_val = group_original['custom_zone'].iloc[0]
                    color = group_original['custom_color'].iloc[0]
                else:
                    ratio_val = group_original['ratio_int'].iloc[0]
                    label_val = f"Ratio = {ratio_val}"
                    color = color_map.get(ratio_val, 'black')
                
                show_legend = label_val not in added_legends
                if show_legend:
                    added_legends.add(label_val)

                # Background rectangle for the segment
                fig.add_shape(
                    type='rect',
                    x0=start_date,
                    x1=end_date,
                    y0=min_price,
                    y1=max_price,
                    xref='x',
                    yref='y',
                    fillcolor=color,
                    opacity=0.3,
                    line_width=0,
                    layer='below'
                )
                
                # Price line for the segment
                fig.add_trace(
                    go.Scatter(
                        x=line_data['Date'],
                        y=line_data[price_column],
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=label_val,
                        showlegend=show_legend
                    )
                )

            fig.update_xaxes(
                dtick=xaxis_dtick,
                tickformat=xaxis_tickformat,
                hoverformat=xaxis_hoverformat,
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikethickness=1,
                spikecolor='grey'
            )
            fig.update_yaxes(
                title_text='Price',
                showspikes=True,
                spikemode='across',
                spikethickness=1,
                spikecolor='grey'
            )
            fig.update_layout(
                title={
                    'text': plotly_title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                hovermode="closest",
                showlegend=True,
                legend_title='Custom Zones' if use_custom else 'Floored Ratio',
                height=800,
                plot_bgcolor='white',
                paper_bgcolor='white'
            )
            return fig

        # -----------------------------------------------------------------------------
        # 12. Create the Matplotlib/Seaborn Figure (Continuous Plot)
        # -----------------------------------------------------------------------------
        def create_matplotlib_figure(df_data):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare legend placeholders for each unique group label
            if use_custom:
                unique_labels = sorted(df_data['custom_zone'].unique())
            else:
                unique_labels = sorted(df_data['ratio_int'].unique())
            
            for label in unique_labels:
                if use_custom:
                    subset = df_data[df_data['custom_zone'] == label]
                    if not subset.empty:
                        color = subset['custom_color'].iloc[0]
                    else:
                        color = 'black'
                    legend_label = f"Zone {label}"
                else:
                    color = color_map.get(label, 'black')
                    legend_label = f"Ratio = {label}"
                ax.plot([], [], color=color, label=legend_label)

            # Plot line segments for each consecutive pair of data points
            for i in range(len(df_data) - 1):
                x1 = df_data.iloc[i]['Date']
                y1 = df_data.iloc[i][price_column]
                if use_custom:
                    color = df_data.iloc[i]['custom_color']
                else:
                    ratio_val = df_data.iloc[i]['ratio_int']
                    color = color_map.get(ratio_val, 'black')
                x2 = df_data.iloc[i+1]['Date']
                y2 = df_data.iloc[i+1][price_column]
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)

            # Background rectangles for each segment
            if use_custom:
                for _, grp in df_data.groupby('zone_group'):
                    label_val = grp['custom_zone'].iloc[0]
                    color = grp['custom_color'].iloc[0]
                    start_date = grp['Date'].iloc[0]
                    end_date = grp['Date'].iloc[-1]
                    ax.axvspan(start_date, end_date, facecolor=color, alpha=0.3)
            else:
                for _, grp in df_data.groupby('change_id'):
                    ratio_val = grp['ratio_int'].iloc[0]
                    color = color_map.get(ratio_val, 'black')
                    start_date = grp['Date'].iloc[0]
                    end_date = grp['Date'].iloc[-1]
                    ax.axvspan(start_date, end_date, facecolor=color, alpha=0.3)

            ax.set_xlabel("Date", fontsize=10)
            ax.set_ylabel("Price", fontsize=10)
            ax.set_ylim(min_price, max_price)
            date_formatter = DateFormatter("%b %d, %Y")
            ax.xaxis.set_major_formatter(date_formatter)
            ax.tick_params(axis='x', labelrotation=20, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, which='major', axis='both', alpha=0.5)
            ax.legend(
                title="Custom Zones" if use_custom else "Floored Ratio",
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=8,
                title_fontsize=10
            )
            ax.set_title(
                f"{asset_option} Price Over Time ({timeframe_option}) by VVIX/VIX Ratio",
                fontsize=12
            )
            fig.tight_layout()
            return fig

        # -----------------------------------------------------------------------------
        # 13. Create Tabs for Plotly vs. Matplotlib
        # -----------------------------------------------------------------------------
        tab1, tab2 = st.tabs(["Plotly Plot", "Matplotlib Plot"])

        with tab1:
            plotly_fig = create_plotly_figure(df_filtered)
            st.plotly_chart(plotly_fig, use_container_width=True)

        with tab2:
            matplotlib_fig = create_matplotlib_figure(df_filtered)
            st.pyplot(matplotlib_fig)

        # -----------------------------------------------------------------------------
        # 14. Display Processed Data
        # -----------------------------------------------------------------------------
        st.markdown("## Processed Data")
        st.markdown("""
        Below is the processed DataFrame used to generate the above plots.
        It includes columns (`Date`, `VIX`, `VVIX`, the asset price column),
        the computed **ratio**, the floored ratio (**ratio_int**), and either the custom zone labels
        (**custom_zone**) with colors (**custom_color**) or the default segmentation (**change_id**).
        """)
        st.dataframe(df_filtered, height=300, use_container_width=True)





