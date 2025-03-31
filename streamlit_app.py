import streamlit as st
import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go # No longer explicitly needed with current plots
import requests
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import time # Import time for date checking

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(layout="wide", page_title="U.S. Job Trends Dashboard")

# Add a title and a short description
st.title("\U0001F4CA U.S. Job Trends Dashboard")
st.caption("Visualizing Total Nonfarm Employment trends and forecasts by state using BLS data. Forecasts are simple linear projections.")


# Load API Key - Make sure you have this in your Streamlit secrets!
# Example: Create a secrets.toml file with:
# BLS_API_KEY = "YOUR_ACTUAL_API_KEY"
try:
    api_key = st.secrets["BLS_API_KEY"]
except FileNotFoundError:
    st.error("ERROR: `secrets.toml` file not found. Please create it with your BLS_API_KEY.")
    st.stop()
except KeyError:
    st.error("ERROR: `BLS_API_KEY` not found in `secrets.toml`. Please add it.")
    st.stop()


# Series maps
state_series_map = {
    "SMU01000000000000001": "Alabama - Total Nonfarm",
    "SMU02000000000000001": "Alaska - Total Nonfarm",
    "SMU04000000000000001": "Arizona - Total Nonfarm",
    "SMU05000000000000001": "Arkansas - Total Nonfarm",
    "SMU06000000000000001": "California - Total Nonfarm",
    "SMU08000000000000001": "Colorado - Total Nonfarm",
    "SMU09000000000000001": "Connecticut - Total Nonfarm",
    "SMU10000000000000001": "Delaware - Total Nonfarm",
    "SMU11000000000000001": "District of Columbia - Total Nonfarm",
    "SMU12000000000000001": "Florida - Total Nonfarm",
    "SMU13000000000000001": "Georgia - Total Nonfarm",
    "SMU15000000000000001": "Hawaii - Total Nonfarm",
    "SMU16000000000000001": "Idaho - Total Nonfarm",
    "SMU17000000000000001": "Illinois - Total Nonfarm",
    "SMU18000000000000001": "Indiana - Total Nonfarm",
    "SMU19000000000000001": "Iowa - Total Nonfarm",
    "SMU20000000000000001": "Kansas - Total Nonfarm",
    "SMU21000000000000001": "Kentucky - Total Nonfarm",
    "SMU22000000000000001": "Louisiana - Total Nonfarm",
    "SMU23000000000000001": "Maine - Total Nonfarm",
    "SMU24000000000000001": "Maryland - Total Nonfarm",
    "SMU25000000000000001": "Massachusetts - Total Nonfarm",
    "SMU26000000000000001": "Michigan - Total Nonfarm",
    "SMU27000000000000001": "Minnesota - Total Nonfarm",
    "SMU28000000000000001": "Mississippi - Total Nonfarm",
    "SMU29000000000000001": "Missouri - Total Nonfarm",
    "SMU30000000000000001": "Montana - Total Nonfarm",
    "SMU31000000000000001": "Nebraska - Total Nonfarm",
    "SMU32000000000000001": "Nevada - Total Nonfarm",
    "SMU33000000000000001": "New Hampshire - Total Nonfarm",
    "SMU34000000000000001": "New Jersey - Total Nonfarm",
    "SMU35000000000000001": "New Mexico - Total Nonfarm",
    "SMU36000000000000001": "New York - Total Nonfarm",
    "SMU37000000000000001": "North Carolina - Total Nonfarm",
    "SMU38000000000000001": "North Dakota - Total Nonfarm",
    "SMU39000000000000001": "Ohio - Total Nonfarm",
    "SMU40000000000000001": "Oklahoma - Total Nonfarm",
    "SMU41000000000000001": "Oregon - Total Nonfarm",
    "SMU42000000000000001": "Pennsylvania - Total Nonfarm",
    "SMU44000000000000001": "Rhode Island - Total Nonfarm",
    "SMU45000000000000001": "South Carolina - Total Nonfarm",
    "SMU46000000000000001": "South Dakota - Total Nonfarm",
    "SMU47000000000000001": "Tennessee - Total Nonfarm",
    "SMU48000000000000001": "Texas - Total Nonfarm",
    "SMU49000000000000001": "Utah - Total Nonfarm",
    "SMU50000000000000001": "Vermont - Total Nonfarm",
    "SMU51000000000000001": "Virginia - Total Nonfarm",
    "SMU53000000000000001": "Washington - Total Nonfarm",
    "SMU54000000000000001": "West Virginia - Total Nonfarm",
    "SMU55000000000000001": "Wisconsin - Total Nonfarm",
    "SMU56000000000000001": "Wyoming - Total Nonfarm",
}

national_industry_map = {
    "CEU0500000001": "Construction - National",
    "CEU0600000001": "Manufacturing - National",
    "CEU0700000001": "Retail Trade - National",
    "CEU0800000001": "Education & Health Services - National",
    "CEU0900000001": "Leisure & Hospitality - National"
}

series_ids = list(state_series_map.keys()) + list(national_industry_map.keys())

# ---------------------------
# DATA LOADING
# ---------------------------
# Use caching to avoid hitting API too often. Cache for 6 hours (21600 seconds)
@st.cache_data(ttl=21600)
def fetch_bls(series_ids, start_year_str, end_year_str):
    """Fetches data from BLS API v2 in batches."""
    # Note: Using st.info/error inside cached function shows msg only on first run (cache miss)
    print(f"Fetching data for {len(series_ids)} series from BLS API for {start_year_str}-{end_year_str}...") # Use print for logs during cache miss
    headers = {"Content-type": "application/json"}
    all_series_data = []
    num_series = len(series_ids)
    batch_size = 50 # Max series per API call for V2

    for i in range(0, num_series, batch_size):
        batch_ids = series_ids[i:min(i + batch_size, num_series)]
        print(f"Requesting batch {i//batch_size + 1}: {len(batch_ids)} series IDs")

        data = {
            "seriesid": batch_ids,
            "startyear": start_year_str,
            "endyear": end_year_str,
            "registrationkey": api_key,
            "catalog": False # Set to True if you need series metadata
        }
        try:
            response = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", data=json.dumps(data), headers=headers, timeout=30)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            response_json = response.json()

            if response_json["status"] != "REQUEST_SUCCEEDED":
                error_msgs = response_json.get('message', [])
                print(f"BLS API Error in batch {i//batch_size + 1}: {'; '.join(error_msgs)}")
                st.warning(f"BLS API Error: {'; '.join(error_msgs)}") # Show warning in UI
                continue # Continue to next batch maybe? Or handle more gracefully

            if "Results" not in response_json or not response_json["Results"]:
                 print(f"No 'Results' key or empty results in API response for batch {i//batch_size + 1}.")
                 continue

            # Process data for the current batch
            batch_records = []
            for s in response_json["Results"]["series"]:
                sid = s["seriesID"]
                series_label = state_series_map.get(sid, national_industry_map.get(sid, sid))
                if not s.get("data"):
                     # print(f"No data points returned for series ID: {sid} ({series_label}) in this batch.")
                     continue

                for item in s["data"]:
                    try:
                        # Skip footnotes/metadata if present
                        if "footnote_codes" in item: continue

                        val = int(float(item["value"].replace(",", "")))
                        month_num_str = item["period"][1:] # Get '01' from 'M01'
                        if item["period"] == "M13": # Skip annual averages
                            continue
                        # Create date string assuming first day of month
                        date_str = f"{item['year']}-{month_num_str}-01"
                        parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')

                        if pd.isna(parsed_date):
                            # print(f"Skipping data point for {series_label} due to invalid date components: Year='{item.get('year')}', Period='{item.get('period')}'")
                            continue

                        batch_records.append({
                            "series_id": sid,
                            "label": series_label,
                            "year": item["year"],
                            "period": item["period"],
                            "periodName": item["periodName"],
                            "value": val,
                            "date": parsed_date
                        })
                    # More specific exception catching could be useful
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                         print(f"Skipping data point for {series_label} ({item.get('year')}-{item.get('periodName')}) due to parsing error: {e} - Value: '{item.get('value')}'")
                         continue
            all_series_data.extend(batch_records)
            time.sleep(0.5) # Small delay between batches

        except requests.exceptions.Timeout:
             print(f"Timeout error fetching BLS data batch {i//batch_size + 1}.")
             st.error(f"Timeout error fetching BLS data batch {i//batch_size + 1}. Some data may be missing.")
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching BLS data batch {i//batch_size + 1}: {e}")
            st.error(f"Network error fetching BLS data: {e}. Check connection or BLS API status.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from BLS API batch {i//batch_size + 1}: {e}")
            st.error("Error decoding API response. BLS API might be down or response format changed.")
        except Exception as e: # Catch-all for other unexpected errors
            print(f"An unexpected error occurred during data fetching batch {i//batch_size + 1}: {e}")
            st.error("An unexpected error occurred during data fetch.")

    if not all_series_data:
         # If API call failed entirely, this will be empty
         print("Failed to retrieve any valid data from the BLS API across all batches.")
         # Return None to indicate total failure, allows differentiating from empty results
         return None

    print(f"Successfully parsed {len(all_series_data)} records from BLS API.")
    return pd.DataFrame(all_series_data)

# --- Attempt to load from CSV first, then fetch if needed ---
csv_file = "bls_employment_data.csv" # Slightly more descriptive name
df = None
try:
    # Don't show this message every time, assume it works or fails silently until API fetch
    # st.write(f"Attempting to load data from local file: {csv_file}")
    df = pd.read_csv(csv_file, parse_dates=["date"])
    # Simple success message only if loaded from cache
    st.sidebar.success(f"Data loaded from cache ({csv_file}).") # Use sidebar for less intrusive message
    # TODO: Add logic here to check if the cached data is recent enough
    # e.g., check max date vs current date, if too old, set df = None to force refresh
    # max_date = df['date'].max()
    # if (pd.Timestamp.now() - max_date).days > 35: # Example: refresh if data is older than 35 days
    #    st.sidebar.info("Cached data is old. Fetching fresh data...")
    #    df = None
except FileNotFoundError:
    # This is expected on first run or after deleting cache
    pass # Don't show info message, just proceed to API fetch
except Exception as e:
    st.warning(f"Could not read local cache file {csv_file}: {e}. Attempting fresh API fetch.")
    df = None

# --- Trigger API Fetch if needed ---
if df is None: # Fetch from API if df is still None
    current_year = pd.Timestamp.now().year
    # Fetch data for the previous full year and the current year up to latest available
    start_year = str(current_year - 1)
    end_year = str(current_year)

    with st.spinner(f"Fetching latest BLS data ({start_year}-{end_year})... Please wait."):
        df = fetch_bls(series_ids, start_year, end_year)

    if df is not None and not df.empty:
        try:
             df.to_csv(csv_file, index=False)
             st.success(f"✅ Data loaded from API and saved to local cache: {csv_file}")
        except Exception as e:
             st.warning(f"⚠️ Failed to save data to local cache file {csv_file}: {e}")
    elif df is None: # fetch_bls indicated critical failure
         st.error("❌ Data fetching failed critically. Cannot display dashboard.")
         st.stop() # Stop execution if we couldn't get any data at all
    else: # df is an empty DataFrame, maybe API returned no results for the period
        st.error("❌ No data returned from API for the requested period. Dashboard may be empty.")
        # Don't stop, allow app to render empty state


# ---------------------------
# DATA CLEANING (only if df exists and is not empty)
# ---------------------------
if df is not None and not df.empty:
    # --- Ensure date column is datetime ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    rows_before = len(df)
    df.dropna(subset=['date'], inplace=True) # Remove rows where date conversion failed
    rows_after = len(df)
    if rows_before > rows_after:
        print(f"Removed {rows_before - rows_after} rows due to invalid dates during cleaning.")

    # --- Filter out unwanted periods/values ---
    df = df[df["period"] != "M13"] # M13 is annual average
    rows_before = len(df)
    df = df[df["value"] > 0] # Ensure positive employment values
    rows_after = len(df)
    if rows_before > rows_after:
         print(f"Removed {rows_before - rows_after} rows due to non-positive values during cleaning.")

    # --- Sort and Calculate Changes ---
    df = df.sort_values(by=["label", "date"])
    df["value_diff"] = df.groupby("label")["value"].diff()
    df["value_pct_change"] = df.groupby("label")["value"].pct_change() * 100

    # --- Create state abbreviation column ---
    state_abbrev_map = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
        "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
        "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
        "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE",
        "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
        "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
        "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
        "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
    }
    # Extract state name robustly using regex. Handles names with spaces.
    df["state_full"] = df["label"].str.extract(r"^(.+?)\s+-\s+Total Nonfarm$")[0].str.strip()
    df["state"] = df["state_full"].map(state_abbrev_map)

else:
    # If df is None or empty after load attempts, we stop earlier or show error message.
    # This block might not be strictly necessary if st.stop() is used on load failure.
    pass # st.warning("Dataframe is empty or None before cleaning step, skipping cleaning.")


# ---------------------------
# FORECASTING
# ---------------------------
# Consider caching forecast results if they take time to compute
# @st.cache_data
def add_forecast(df_subset, months=6):
    """Generates a simple linear forecast for a given data subset."""
    # df_subset should contain data for a single 'label'
    df_subset = df_subset.copy().sort_values("date").reset_index(drop=True)
    label_name = df_subset['label'].iloc[0] if not df_subset.empty else "Selected Series"

    if df_subset.empty or len(df_subset) < 2:
        # st.warning(f"Not enough data points ({len(df_subset)}) for {label_name} to generate forecast.")
        df_subset["forecast"] = False
        return df_subset # Return original data marked as not forecast

    # Create integer index for regression
    df_subset["ts_int"] = np.arange(len(df_subset))

    model = LinearRegression()
    try:
        # Fit on non-NaN data only
        subset_clean = df_subset.dropna(subset=['ts_int', 'value'])
        if len(subset_clean) < 2:
             # st.warning(f"Not enough valid data points ({len(subset_clean)}) for {label_name} to forecast.")
             df_subset["forecast"] = False
             return df_subset

        X = subset_clean[["ts_int"]]
        y = subset_clean["value"]
        model.fit(X, y)

    except ValueError as e:
        st.error(f"Error fitting linear regression model for {label_name}: {e}")
        df_subset["forecast"] = False
        return df_subset

    # Predict future values
    last_date = df_subset["date"].iloc[-1]
    # Ensure future dates start strictly after the last known date
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS')

    last_ts_int = df_subset["ts_int"].iloc[-1]
    future_ints = np.arange(last_ts_int + 1, last_ts_int + months + 1)
    future_preds = model.predict(future_ints.reshape(-1, 1))

    # Ensure forecast values are non-negative (optional, depends on context)
    future_preds[future_preds < 0] = 0

    # Create future dataframe
    df_future = pd.DataFrame({
        "date": future_dates,
        "value": future_preds,
        "label": label_name,
        "forecast": True,
        # Carry over other potentially useful columns, fill if not applicable
        "series_id": df_subset["series_id"].iloc[0] if 'series_id' in df_subset.columns else None,
        "state_full": df_subset["state_full"].iloc[0] if 'state_full' in df_subset.columns else None,
        "state": df_subset["state"].iloc[0] if 'state' in df_subset.columns else None,
    })

    df_subset["forecast"] = False # Mark historical data

    # Concatenate historical and forecast data
    return pd.concat([df_subset, df_future], ignore_index=True)

# ---------------------------
# UI AND VISUALS
# ---------------------------

# Proceed only if data loading and cleaning were successful
if df is not None and not df.empty:

    st.header("State Employment Analysis")
    # State-level selector
    state_labels_available = sorted(df[df["label"].str.contains("Total Nonfarm", na=False)]["label"].unique())

    if not state_labels_available:
        st.warning("No state-level 'Total Nonfarm' data found in the loaded dataset.")
    else:
        state_label = st.selectbox("Select a State to Analyze:", state_labels_available, help="Choose a state to see its detailed employment trends.")

        if state_label: # Proceed only if a state is selected
            state_df = df[df["label"] == state_label].copy()

            if not state_df.empty:
                # Generate forecast data for the selected state
                state_df_forecast = add_forecast(state_df, months=6) # Forecast 6 months ahead

                # Use columns for layout: Trend chart | YoY chart
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("\U0001F4C8 Employment Trend & Forecast")
                    if not state_df_forecast.empty:
                        fig_trend = px.line(state_df_forecast, x="date", y="value",
                                      color="forecast", # Color lines based on forecast status
                                      line_dash="forecast", # Dash forecast lines
                                      markers=False, # Cleaner look without markers? You can turn back on.
                                      title=f"{state_label}: Trend",
                                      labels={"value": "Employment (Thousands)", "date": "Date", "forecast": "Is Forecast?"},
                                      template="plotly_white") # Use a clean template
                        # Explicitly style forecast vs historical lines
                        fig_trend.for_each_trace(lambda t: t.update(line=dict(dash='dash')) if t.name == 'True' else ())
                        fig_trend.for_each_trace(lambda t: t.update(line=dict(dash='solid')) if t.name == 'False' else ())
                        fig_trend.update_layout(legend_title_text='Data Type') # Better legend title
                        st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        st.warning(f"Could not generate forecast plot for {state_label}.")

                with col2:
                    # --- YoY Change ---
                    st.subheader("\U0001F4C5 Monthly YoY Change")
                    df_yoy = state_df.dropna(subset=['date', 'value']).copy() # Use data with valid dates/values
                    df_yoy["year"] = df_yoy["date"].dt.year
                    df_yoy["month"] = df_yoy["date"].dt.month
                    df_yoy["Month"] = df_yoy["date"].dt.strftime("%b") # Month abbreviation
                    df_yoy = df_yoy.sort_values(by=['year', 'month'])

                    try:
                        df_pivot = df_yoy.pivot_table(index=["month", "Month"], columns="year", values="value", aggfunc='first')

                        if df_pivot.shape[1] >= 2: # Need at least two years of data
                            years = sorted(df_pivot.columns)
                            current_year_data = df_pivot[years[-1]]
                            previous_year_data = df_pivot[years[-2]]
                            # Calculate YoY change robustly
                            df_pivot["yoy_change"] = np.where(
                                (previous_year_data != 0) & previous_year_data.notna() & current_year_data.notna(),
                                ((current_year_data - previous_year_data) / previous_year_data) * 100,
                                np.nan
                            )
                            df_pivot = df_pivot.reset_index().sort_values(by="month")
                        else:
                            # Don't show warning if only one year, just indicate no plot
                            # st.warning(f"Not enough yearly data ({df_pivot.shape[1]} years found) for {state_label} to calculate YoY change.")
                            df_pivot = df_pivot.reset_index()
                            df_pivot["yoy_change"] = np.nan

                        # Plot if yoy_change column exists and has valid data
                        if "yoy_change" in df_pivot.columns and not df_pivot["yoy_change"].isnull().all():
                             fig_yoy = px.bar(df_pivot.dropna(subset=["yoy_change"]),
                                          x="Month", y="yoy_change",
                                          title=f"{state_label}: YoY % Change",
                                          labels={"yoy_change": "YoY Change (%)"},
                                          template="plotly_white")
                             # Ensure bars are sorted chronologically by month
                             month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                             fig_yoy.update_layout(xaxis={'categoryorder':'array', 'categoryarray': [m for m in month_order if m in df_pivot['Month'].unique()]})
                             st.plotly_chart(fig_yoy, use_container_width=True)
                        else:
                             st.info(f"Not enough data (requires >1 year) to calculate Year-over-Year change for {state_label}.")

                    except Exception as e:
                        st.error(f"Error generating YoY plot for {state_label}: {e}")
            else:
                st.warning(f"No data available for the selected state: {state_label}")
        else:
            st.info("Select a state from the dropdown above to view its trends.")

    st.divider() # Add a visual separator

    # --- Choropleth Map ---
    st.header("\U0001F5FA National Employment Map")
    if 'state' in df.columns:
         # Get months available where state data exists for Nonfarm jobs
         map_months_df = df[df['state'].notna() & df['label'].str.contains("Total Nonfarm", na=False)].copy()
         map_months_df['year_month'] = map_months_df['date'].dt.strftime("%Y-%m")
         available_months = sorted(map_months_df['year_month'].unique(), reverse=True)
    else:
         available_months = []
         st.warning("Cannot create map: 'state' column missing after data processing.")

    if available_months:
        map_selected_month = st.selectbox("Select Month for Map View:", available_months, help="Choose a month to see the national employment snapshot.")

        # Filter data for the selected month
        map_df = df[
            (df["date"].dt.strftime("%Y-%m") == map_selected_month) &
            (df["label"].str.contains("Total Nonfarm", na=False)) &
            (df["state"].notna())
        ].copy()

        # Check for missing states for the map - less aggressive warning
        expected_states = set(state_abbrev_map.keys()) # Use keys from the map as reference
        state_full_names_in_map = set(map_df["state_full"].unique())
        missing_state_names = expected_states - state_full_names_in_map
        if missing_state_names:
            # Only show caption if states are actually missing
             st.caption(f"Note: Data for {len(missing_state_names)} state(s) may be missing or pending release for {map_selected_month}.")


        # Plot map if map_df is not empty
        if not map_df.empty:
             fig_map = px.choropleth(map_df,
                                    locations="state",         # Use state abbreviation column
                                    locationmode="USA-states", # Specify location mode
                                    color="value",             # Color by employment value
                                    hover_name="state_full",   # Show full state name on hover
                                    hover_data={'state':False, 'value':':,'}, # Show value formatted, hide abbreviation
                                    scope="usa",               # Limit map scope to USA
                                    color_continuous_scale="Blues", # Example: Use Blues color scale
                                    title=f"Total Nonfarm Employment – {map_selected_month}",
                                    labels={'value':'Employment (Thousands)'}) # Improve label
             fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}) # Adjust map margins
             st.plotly_chart(fig_map, use_container_width=True)
             st.caption("Data: U.S. Bureau of Labor Statistics (CES). Color intensity represents employment level (thousands of persons).") # Added units clarification
        else:
             st.info(f"No state data available to display on the map for {map_selected_month}.")

    else:
        st.warning("No months with valid state data available for map view.")

    st.divider()

    # --- Download Button ---
    st.download_button(
        label="\U0001F4C2 Download Cleaned Data (CSV)",
        data=df.to_csv(index=False).encode('utf-8'), # Encode to bytes
        # Use date range from actual data for filename
        file_name=f"bls_nonfarm_employment_{df['date'].min().strftime('%Y%m')}_{df['date'].max().strftime('%Y%m')}.csv",
        mime="text/csv",
        help="Download the cleaned dataset used for this dashboard as a CSV file." # Add tooltip
    )

    # --- EXECUTIVE SUMMARY SECTION ---
    st.divider()
    st.header("Executive Summary & Chart Guide")
    st.markdown("""
        This dashboard provides insights into U.S. employment trends, focusing on **Total Nonfarm jobs** at the state level, using data directly from the **U.S. Bureau of Labor Statistics (BLS)** Current Employment Statistics (CES) program.

        **How to Use:**
        1.  Select a state from the **'Select a State to Analyze'** dropdown to view detailed historical trends, a simple forecast, and year-over-year changes for that specific state.
        2.  Use the **'Select Month for Map View'** dropdown to visualize the distribution of Total Nonfarm jobs across all states for a specific month.

        **Chart Explanations:**

        * **\U0001F4C8 Employment Trend & Forecast (Line Chart):**
            * Shows the historical monthly employment level (in thousands of persons) for the selected state (solid line).
            * Includes a 6-month future projection based on a simple linear trend (dashed line). This forecast indicates the general direction but does not account for seasonality or economic shocks.

        * **\U0001F4C5 Monthly YoY Change (Bar Chart):**
            * Displays the percentage change in employment for each month compared to the same month in the previous year (Year-over-Year).
            * Positive bars indicate job growth compared to the previous year, while negative bars indicate contraction. This helps identify acceleration or deceleration trends while mitigating seasonal effects. Requires at least two years of data to calculate.

        * **\U0001F5FA National Employment Map (Choropleth Map):**
            * Provides a geographical snapshot of total nonfarm employment (thousands of persons) across the U.S. for the selected month.
            * States are colored based on their employment level, with darker shades typically indicating higher employment. Hover over a state to see its name and exact employment figure.

        * **\U0001F4C2 Download Cleaned Data (Button):**
            * Allows you to download the processed data used in these visualizations as a CSV file for offline analysis.

        *Data is sourced from the BLS and updated based on their release schedule. Data for the most recent months may be preliminary and subject to revision.*
    """)
    # --- END EXECUTIVE SUMMARY SECTION ---

else:
    # This message appears if data loading failed critically earlier
    st.error("❌ Dashboard cannot be displayed due to data loading issues.")
