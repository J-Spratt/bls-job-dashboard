import os
import traceback
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(layout="wide", page_title="U.S. Job Trends Dashboard")

# Add a title and a short description
st.markdown("<h1 style='text-align: center;'>U.S. Job Trends Dashboard</h1>", unsafe_allow_html=True)
st.caption("Visualizing Total Nonfarm Employment trends and forecasts by state and selected national industries using BLS data. "
           "Forecasts are based on simple linear projections.")
st.info("ℹ️ **Data Source Note**: This dashboard uses BLS API data, which may occasionally differ from published state employment reports due to different update schedules and methodologies. Both sources are official BLS data.")

# --- Load BLS API Key ---
api_key = None
try:
    api_key = st.secrets["BLS_API_KEY"]
    if not api_key:
        st.error("ERROR: `BLS_API_KEY` found in `secrets.toml` but it is empty.")
        st.stop()
except FileNotFoundError:
    st.error("ERROR: `secrets.toml` file not found. Create `.streamlit/secrets.toml` with your BLS_API_KEY.")
    st.stop()
except KeyError:
    st.error("ERROR: `BLS_API_KEY` not found in `secrets.toml`. Add `BLS_API_KEY = \"YourKey\"`.")
    st.stop()
except Exception as e:
    st.error(f"Error loading secrets: {e}")
    st.stop()

# --- Series Definitions ---
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
    "CEU0000000001": "Total Nonfarm - National",
    "CEU0500000001": "Total Private - National",
    "CEU0600000001": "Goods-Producing - National",
    "CEU0700000001": "Service-Providing - National",
    "CEU0800000001": "Private Service-Providing - National",
    "CEU1000000001": "Mining and Logging - National",
    "CEU2000000001": "Construction - National",
    "CEU3000000001": "Manufacturing - National",
    "CEU4000000001": "Trade, Transportation, and Utilities - National",
    "CEU4100000001": "Wholesale Trade - National",
    "CEU4200000001": "Retail Trade - National",
    "CEU4300000001": "Transportation and Warehousing - National",
    "CEU4400000001": "Utilities - National",
    "CEU5000000001": "Information - National",
    "CEU5500000001": "Financial Activities - National",
    "CEU6000000001": "Professional and Business Services - National",
    "CEU6500000001": "Education and Health Services - National",
    "CEU7000000001": "Leisure and Hospitality - National",
    "CEU8000000001": "Other Services - National",
    "CEU9000000001": "Government - National"
}
series_ids = list(state_series_map.keys()) + list(national_industry_map.keys())
all_series_map = {**state_series_map, **national_industry_map}

# ---------------------------
# DATA LOADING FUNCTION (fetch_bls)
# ---------------------------
@st.cache_data(ttl=3600)  # Reduced cache time to 1 hour for more frequent updates
def fetch_bls(series_ids_func, start_year_str_func, end_year_str_func, api_key_func):
    """Fetches data from BLS API v2 in batches."""
    logging.info(f"--- Running fetch_bls ---")
    logging.info(f"Fetching data for {len(series_ids_func)} series from BLS API for {start_year_str_func}-{end_year_str_func}...")
    headers = {"Content-type": "application/json"}
    all_series_data = []
    num_series = len(series_ids_func)
    batch_size = 50
    max_batches = (num_series + batch_size - 1) // batch_size

    for i in range(0, num_series, batch_size):
        current_batch_num = i // batch_size + 1
        batch_ids = series_ids_func[i:min(i + batch_size, num_series)]
        logging.info(f"Requesting batch {current_batch_num}/{max_batches}: {len(batch_ids)} series IDs")
        data_payload = {
            "seriesid": batch_ids,
            "startyear": start_year_str_func,
            "endyear": end_year_str_func,
            "registrationkey": api_key_func,
            "catalog": False
        }
        data_json_str = json.dumps(data_payload)

        try:
            response = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data_json_str, headers=headers, timeout=45)
            logging.info(f"Batch {current_batch_num}: Response Status Code: {response.status_code}")

            if response.status_code != 200:
                logging.warning(f"Batch {current_batch_num}: Received non-200 status code: {response.status_code}")
                continue

            response_json = response.json()

            if response_json.get("status") != "REQUEST_SUCCEEDED":
                error_msgs = response_json.get('message', ['No message provided by API.'])
                error_msgs_str = [str(m) for m in error_msgs if m]
                error_string = '; '.join(error_msgs_str) if error_msgs_str else "Unknown API Error Status"
                logging.warning(f"BLS API Error in batch {current_batch_num}: {error_string}")
                st.warning(f"BLS API Error (batch {current_batch_num}): {error_string}")
                continue

            if "Results" not in response_json or not response_json.get("Results") or "series" not in response_json["Results"]:
                logging.warning(f"No valid 'Results' or 'series' key found for batch {current_batch_num}.")
                continue

            batch_records = []
            for s in response_json["Results"]["series"]:
                sid = s.get("seriesID")
                if not sid:
                    logging.warning("Skipping series entry with no seriesID.")
                    continue
                series_label = all_series_map.get(sid, sid)

                if not s.get("data"):
                    logging.info(f"Series {sid} ({series_label}) has no data points in this response.")
                    continue

                # DEBUG: Log Alabama data specifically
                if sid == "SMU01000000000000001":
                    logging.info(f"Alabama API data received: {len(s.get('data', []))} points")
                    if s.get("data"):
                        latest_point = s["data"][0]  # BLS returns newest first
                        logging.info(f"Alabama latest point: {latest_point}")

                for item in s["data"]:
                    try:
                        # REVISED: More lenient preliminary data handling
                        # Only skip if explicitly marked as preliminary AND there's newer non-preliminary data
                        footnote_codes = item.get("footnote_codes", "")
                        is_preliminary = "P" in footnote_codes if footnote_codes else False
                        
                        val_str = item.get("value")
                        period = item.get("period")
                        year = item.get("year")
                        periodName = item.get("periodName")

                        # Basic validation for essential fields
                        if not all([val_str, period, year, periodName]):
                            logging.warning(f"Skipping data point with missing fields: {item} for Series {sid}")
                            continue
                        
                        # Ensure it's monthly data (M01-M12)
                        if not period.startswith('M') or period == "M13":
                             continue

                        val = float(val_str.replace(",", ""))
                        month_num_str = period[1:]
                        date_str = f"{year}-{month_num_str}-01"
                        parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
                        if pd.isna(parsed_date):
                            logging.warning(f"Skipping data point with unparseable date: {item} for Series {sid}")
                            continue

                        batch_records.append({
                            "series_id": sid, "label": series_label, "year": int(year),
                            "period": period, "periodName": periodName, "value": val,
                            "date": parsed_date, "is_preliminary": is_preliminary,
                            "footnote_codes": footnote_codes
                        })
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                        logging.warning(f"Skipping data point for series {sid} due to parsing error: {e} - Item: '{item}'")

            all_series_data.extend(batch_records)
            logging.info(f"Processed {len(batch_records)} valid monthly records for batch {current_batch_num}.")
            time.sleep(0.5)

        except Exception as e:
            logging.error(f"Error processing batch {current_batch_num}: {e}")
            continue

    if not all_series_data:
        logging.warning("--- fetch_bls finished: Failed to retrieve any valid data. ---")
        st.warning("No data was successfully retrieved from the BLS API for the requested series and time period.")
        return None
    else:
        logging.info(f"--- fetch_bls finished: Successfully parsed {len(all_series_data)} valid monthly records. ---")
        return pd.DataFrame(all_series_data)

# --- Enhanced Cache Management ---
st.sidebar.subheader("Data Management")

# Cache status and controls
csv_file = "bls_employment_data.csv"
cache_expiry_hours = 1  # Cache expires hourly for more frequent updates

# Add cache clearing button
if st.sidebar.button("🔄 Clear Cache & Refresh", help="Force refresh data from BLS API"):
    st.cache_data.clear()
    if os.path.exists(csv_file):
        os.remove(csv_file)
        st.sidebar.success("Cache cleared!")
    st.rerun()

# Add debug mode toggle
debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=False, help="Show detailed debugging information")

force_refresh = st.sidebar.checkbox("Force Data Refresh", value=False, help="Check this box to bypass the local cache and fetch fresh data from the BLS API.")

df = None
# Cache loading logic with better error handling
if not force_refresh and os.path.exists(csv_file):
    try:
        cache_info = os.stat(csv_file)
        cache_age_seconds = time.time() - cache_info.st_mtime
        cache_age_hours = cache_age_seconds / 3600
        
        if cache_age_hours < cache_expiry_hours:
            df_cache = pd.read_csv(csv_file, parse_dates=["date"])
            if not df_cache.empty and all(col in df_cache.columns for col in ['series_id', 'label', 'value', 'date']):
                df = df_cache
                max_cached_date = df["date"].max().date()
                st.sidebar.success(f"✅ Using cached data\nLast updated: {max_cached_date}")
                if debug_mode:
                    st.sidebar.info(f"Cache age: {cache_age_hours:.1f} hours")
            else:
                st.sidebar.warning("Cache file invalid, fetching fresh data")
                if os.path.exists(csv_file):
                    os.remove(csv_file)
                df = None
        else:
            st.sidebar.warning(f"Cache expired ({cache_age_hours:.1f} hours old)")
            df = None
    except Exception as e:
        st.sidebar.error(f"Cache error: {e}")
        df = None
else:
    if force_refresh:
        st.sidebar.info("🔄 Force refresh enabled")
    else:
        st.sidebar.info("📥 No cache found")

# API Data Fetching with improved error handling
if df is None:
    st.sidebar.info("🌐 Fetching from BLS API...")
    
    fetch_years_history = 5
    current_year = pd.Timestamp.now().year
    start_year = str(current_year - fetch_years_history)
    end_year = str(current_year)

    with st.spinner(f"Fetching BLS data ({start_year}-{end_year})..."):
        try:
            df_fetched = fetch_bls(series_ids, start_year, end_year, api_key)
            
            if isinstance(df_fetched, pd.DataFrame) and not df_fetched.empty:
                df = df_fetched
                # Save to cache
                df.to_csv(csv_file, index=False)
                st.sidebar.success("✅ Fresh data fetched and cached")
                logging.info(f"Cached {len(df)} rows to {csv_file}")
            else:
                st.error("Failed to fetch valid data from BLS API")
                st.stop()
        except Exception as e:
            st.error(f"Critical error during API fetch: {e}")
            logging.error(f"API fetch error: {e}")
            st.stop()

# ---------------------------
# DATA CLEANING & PREPARATION
# ---------------------------
if df is not None and not df.empty:
    logging.info("--- Starting Data Cleaning & Preparation ---")
    
    # Show data info in debug mode
    if debug_mode:
        with st.expander("🔍 Debug: Raw Data Info"):
            st.write(f"**Total records:** {len(df)}")
            st.write(f"**Date range:** {df['date'].min().date()} to {df['date'].max().date()}")
            st.write(f"**Series count:** {df['series_id'].nunique()}")
            if 'is_preliminary' in df.columns:
                st.write(f"**Preliminary records:** {df['is_preliminary'].sum()}")
    
    try:
        # Data type conversion with validation
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce', downcast='integer')

        # Remove invalid records
        initial_rows = len(df)
        df.dropna(subset=['date', 'value', 'year', 'label', 'series_id'], inplace=True)
        df = df[df["value"] > 0].copy()
        
        if debug_mode and len(df) < initial_rows:
            st.sidebar.info(f"Cleaned: Removed {initial_rows - len(df)} invalid records")

        # Check if DataFrame is empty after cleaning
        if df.empty:
            st.error("No valid data remaining after cleaning. Cannot proceed.")
            st.stop()

        # Sort for proper calculations
        df = df.sort_values(by=["label", "date"]).reset_index(drop=True)

        # IMPROVED: Handle duplicate dates per series (keep most recent, prefer non-preliminary)
        if 'is_preliminary' in df.columns:
            # For each series-date combination, prefer non-preliminary data
            df = df.sort_values(['label', 'date', 'is_preliminary']).groupby(['label', 'date']).first().reset_index()

        # Calculate changes with validation
        df["value_diff"] = df.groupby("label")["value"].diff()
        df["value_pct_change"] = df.groupby("label")["value"].pct_change() * 100

        # Calculate YoY change with better error handling
        df['value_yoy_lag'] = df.groupby('label')['value'].shift(12)
        df['value_yoy_diff'] = df['value'] - df['value_yoy_lag']
        df['value_yoy_pct_change'] = np.where(
            (df['value_yoy_lag'] != 0) & (df['value_yoy_lag'].notna()),
            (df['value_yoy_diff'] / df['value_yoy_lag']) * 100,
            np.nan
        )
        df['value_yoy_pct_change'] = df['value_yoy_pct_change'].replace([np.inf, -np.inf], np.nan)

        # State mapping
        state_abbrev_map = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
            "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
            "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
            "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
            "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
            "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
            "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
            "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
            "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
            "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"
        }
        
        df["state_full"] = df["label"].str.extract(r"^([a-zA-Z\s\.]+?)\s+-\s+Total Nonfarm$", expand=False).str.strip()
        df["state_abbrev"] = df["state_full"].map(state_abbrev_map)
        df['is_national'] = df['series_id'].isin(national_industry_map.keys())

        # DEBUG: Alabama-specific validation
        if debug_mode:
            alabama_data = df[df["label"] == "Alabama - Total Nonfarm"].tail(3)
            if not alabama_data.empty:
                with st.expander("🏛️ Debug: Alabama Latest Data"):
                    st.dataframe(alabama_data[['date', 'value', 'value_diff', 'value_pct_change', 'is_preliminary', 'footnote_codes']])
                    
                    # Compare with expected values
                    latest_alabama = alabama_data.iloc[-1]
                    expected_value = 2210.7  # From official BLS data
                    expected_mom_change = -4.9
                    
                    st.write("**Validation vs Official BLS:**")
                    st.write(f"Latest Employment: {latest_alabama['value']:.1f}K (Expected: {expected_value}K)")
                    if abs(latest_alabama['value'] - expected_value) > 1:
                        st.error("⚠️ Employment value doesn't match official BLS data!")
                    
                    st.write(f"MoM Change: {latest_alabama['value_diff']:.1f}K (Expected: {expected_mom_change}K)")
                    if abs(latest_alabama['value_diff'] - expected_mom_change) > 1:
                        st.error("⚠️ MoM change doesn't match official BLS data!")

        logging.info("--- Data Cleaning & Preparation Complete ---")

    except Exception as clean_e:
        st.error(f"Error during data cleaning: {clean_e}")
        logging.error(f"Data cleaning error: {clean_e}")
        st.stop()

# ---------------------------
# FORECASTING FUNCTION
# ---------------------------
def add_forecast(df_subset, months=6):
    """Adds a simple linear forecast to a dataframe subset."""
    df_subset = df_subset.copy().sort_values("date").reset_index(drop=True)
    label_name = df_subset['label'].iloc[0] if not df_subset.empty else "Selected Series"

    if df_subset.empty or len(df_subset) < 2:
        logging.warning(f"Skipping forecast for '{label_name}': Not enough data points ({len(df_subset)}).")
        df_subset["forecast"] = False
        return df_subset

    df_subset["ts_int"] = np.arange(len(df_subset))
    subset_clean = df_subset.dropna(subset=['ts_int', 'value'])
    if len(subset_clean) < 2:
        logging.warning(f"Skipping forecast for '{label_name}': Not enough clean data points for model ({len(subset_clean)}).")
        df_subset["forecast"] = False
        return df_subset

    model = LinearRegression()
    try:
        X = subset_clean[["ts_int"]]
        y = subset_clean["value"]
        model.fit(X, y)
        logging.info(f"Forecast model fitted for '{label_name}' using {len(subset_clean)} points.")
    except Exception as forecast_fit_e:
        st.error(f"Error fitting forecast model for '{label_name}': {forecast_fit_e}")
        df_subset["forecast"] = False
        return df_subset

    try:
        last_actual_date = df_subset["date"].iloc[-1]
        future_dates = pd.date_range(start=last_actual_date + pd.DateOffset(months=1), periods=months, freq='MS')
        last_ts_int = df_subset["ts_int"].iloc[-1]
        future_ints = np.arange(last_ts_int + 1, last_ts_int + 1 + months)

        future_preds = model.predict(future_ints.reshape(-1, 1))
        future_preds[future_preds < 0] = 0

        future_data = {
            "date": future_dates, "value": future_preds, "label": label_name,
            "forecast": True,
            "series_id": df_subset["series_id"].iloc[0] if 'series_id' in df_subset.columns else None,
            "state_full": df_subset["state_full"].iloc[0] if 'state_full' in df_subset.columns and pd.notna(df_subset["state_full"].iloc[0]) else None,
            "state_abbrev": df_subset["state_abbrev"].iloc[0] if 'state_abbrev' in df_subset.columns and pd.notna(df_subset["state_abbrev"].iloc[0]) else None,
            "is_national": df_subset["is_national"].iloc[0] if 'is_national' in df_subset.columns else None,
            "year": future_dates.year, "periodName": future_dates.strftime('%B'),
            "period": future_dates.strftime('M%m')
        }
        df_future = pd.DataFrame(future_data)
        df_subset["forecast"] = False
        df_combined = pd.concat([df_subset, df_future], ignore_index=True)
        return df_combined

    except Exception as forecast_pred_e:
        st.error(f"Error predicting forecast for '{label_name}': {forecast_pred_e}")
        df_subset["forecast"] = False
        return df_subset

# ---------------------------
# === MAIN DASHBOARD UI ===
# ---------------------------
if df is not None and not df.empty:
    logging.info("--- Starting UI and Visuals ---")
    
    # Add data quality indicator
    latest_date_overall = df['date'].max()
    data_freshness = (pd.Timestamp.now() - latest_date_overall).days
    
    if data_freshness <= 7:
        st.success(f"📊 Data is current (latest: {latest_date_overall.strftime('%B %Y')})")
    elif data_freshness <= 30:
        st.warning(f"📊 Data is {data_freshness} days old (latest: {latest_date_overall.strftime('%B %Y')})")
    else:
        st.error(f"📊 Data is {data_freshness} days old (latest: {latest_date_overall.strftime('%B %Y')}) - Consider refreshing")

    try:
        # --- State Analysis Section ---
        st.header("State Employment Analysis")
        state_labels_available = sorted(df[df["state_abbrev"].notna()]["label"].unique())

        if not state_labels_available:
            st.warning("No state-level 'Total Nonfarm' data found to display.")
        else:
            state_label = st.selectbox(
                "Select a State:", state_labels_available, key="state_selector",
                help="Choose a state to view detailed trends and forecasts."
            )

            if state_label:
                state_df_orig = df[df["label"] == state_label].copy()
                if not state_df_orig.empty:
                    state_df_forecast = add_forecast(state_df_orig, months=6)

                    st.subheader(f"Key Indicators: {state_label}")
                    
                    # Get latest actual data
                    actual_rows = state_df_forecast[(state_df_forecast['forecast'] == False) & pd.notna(state_df_forecast['value'])]
                    if not actual_rows.empty:
                        latest_actual_data = actual_rows.sort_values(by='date').iloc[-1]

                        latest_date_str = latest_actual_data['date'].strftime('%B %Y')
                        latest_val = latest_actual_data.get('value', 0)
                        mom_diff_val = latest_actual_data.get('value_diff', np.nan)
                        mom_pct_val = latest_actual_data.get('value_pct_change', np.nan)
                        yoy_diff_val = latest_actual_data.get('value_yoy_diff', np.nan)
                        yoy_pct_val = latest_actual_data.get('value_yoy_pct_change', np.nan)

                        # Get forecast
                        forecast_rows = state_df_forecast[state_df_forecast['forecast'] == True]
                        if not forecast_rows.empty:
                            first_forecast_data = forecast_rows.sort_values(by='date').iloc[0]
                            forecast_val = first_forecast_data.get('value', np.nan)
                            forecast_date_str = first_forecast_data['date'].strftime('%B %Y')
                        else:
                            forecast_val = np.nan
                            forecast_date_str = "N/A"

                        # Display metrics with better formatting
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Latest Employment", f"{latest_val:,.1f}K",
                                      help=f"Latest data for {latest_date_str}")
                        
                        mom_delta = f"{mom_pct_val:+.2f}%" if not pd.isna(mom_pct_val) else ""
                        col_m2.metric("MoM Change",
                                      f"{mom_diff_val:+,.1f}K" if not pd.isna(mom_diff_val) else "N/A",
                                      mom_delta,
                                      help="Month-over-Month change")
                        
                        yoy_delta = f"{yoy_pct_val:+.2f}%" if not pd.isna(yoy_pct_val) else ""
                        col_m3.metric("YoY Change",
                                      f"{yoy_diff_val:+,.1f}K" if not pd.isna(yoy_diff_val) else "N/A",
                                      yoy_delta,
                                      help="Year-over-Year change")
                        
                        col_m4.metric("Next Month Forecast",
                                      f"{forecast_val:,.1f}K" if not pd.isna(forecast_val) else "N/A",
                                      help=f"Forecast for {forecast_date_str}")

                        # Add preliminary data warning if applicable
                        if 'is_preliminary' in latest_actual_data and latest_actual_data.get('is_preliminary', False):
                            st.info("ℹ️ Latest data point is preliminary and subject to revision.")

                    else:
                        st.warning("Could not retrieve latest data points for key indicators.")

                    # Charts
                    st.subheader("Employment Trend & Forecast")
                    fig_state_line = px.line(
                        state_df_forecast, x='date', y='value', color='forecast',
                        labels={'date': 'Date', 'value': 'Employment (Thousands)', 'forecast': 'Data Type'},
                        title=f"Employment Trend for {state_label}"
                    )
                    fig_state_line.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%B %Y}<br>Employment: %{y:,.1f}K<extra></extra>"
                    )
                    fig_state_line.update_layout(legend_title_text='Data Type', hovermode="x unified")
                    st.plotly_chart(fig_state_line, use_container_width=True)

                    st.subheader("Monthly % Change")
                    state_df_actual_mom = state_df_forecast[(state_df_forecast['forecast'] == False) & state_df_forecast['value_pct_change'].notna()].copy()
                    if not state_df_actual_mom.empty:
                        fig_state_bar = px.bar(
                            state_df_actual_mom, x='date', y='value_pct_change',
                            labels={'date': 'Date', 'value_pct_change': 'MoM % Change'},
                            title=f"Month-over-Month % Change for {state_label}"
                        )
                        fig_state_bar.update_traces(hovertemplate="Date: %{x|%B %Y}<br>MoM Change: %{y:.2f}%<extra></extra>")
                        st.plotly_chart(fig_state_bar, use_container_width=True)
                    else:
                        st.info("Not enough data to display Monthly % Change chart.")

                else:
                    st.warning(f"No data found for the selected state: {state_label}")

        st.divider()

        # --- State Comparison Map Section ---
        st.header("State Comparison Map")
        st.caption("Latest Month-over-Month Percentage Change")
        
        try:
            df_states_only = df[df['state_abbrev'].notna()].copy()
            
            if not df_states_only.empty:
                latest_state_indices = df_states_only.groupby('label')['date'].idxmax()
                latest_state_data = df_states_only.loc[latest_state_indices]
                latest_state_data = latest_state_data.dropna(subset=['state_abbrev', 'value_pct_change'])
                
                if not latest_state_data.empty:
                    map_latest_date = latest_state_data['date'].max()
                    map_latest_date_str = map_latest_date.strftime('%B %Y')
                    st.markdown(f"*(Latest available data: {map_latest_date_str})*")

                    fig_map = px.choropleth(
                        latest_state_data, locations='state_abbrev', locationmode='USA-states',
                        color='value_pct_change', scope='usa', color_continuous_scale="RdYlGn",
                        range_color=[-2, 2],
                        hover_name='state_full',
                        hover_data={'state_abbrev': False, 'value_pct_change': ':.2f%'},
                        title="Latest MoM % Change by State"
                    )
                    fig_map.update_layout(
                        coloraxis_colorbar=dict(title="MoM % Change"),
                        geo=dict(lakecolor='rgba(0,0,0,0)')
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                else:
                    st.warning("Could not retrieve valid MoM % change data for the map.")
        except Exception as map_e:
            st.error(f"Error generating state comparison map: {map_e}")
            logging.error(f"Map error: {map_e}")

        st.divider()

        # --- National Industry Analysis Section ---
        st.header("National Industry Analysis")
        national_labels_available = sorted(df[df["is_national"] == True]["label"].unique())

        if not national_labels_available:
            st.warning("No national industry data found to display.")
        else:
            national_label = st.selectbox(
                "Select a National Series:", national_labels_available, key="national_selector",
                help="Choose a national industry or aggregate series."
            )

            if national_label:
                nat_df_orig = df[df["label"] == national_label].copy()
                if not nat_df_orig.empty:
                    nat_df_forecast = add_forecast(nat_df_orig, months=6)

                    st.subheader(f"Key Indicators: {national_label}")
                    
                    actual_rows_nat = nat_df_forecast[(nat_df_forecast['forecast'] == False) & pd.notna(nat_df_forecast['value'])]
                    if not actual_rows_nat.empty:
                        latest_actual_data_nat = actual_rows_nat.sort_values(by='date').iloc[-1]

                        latest_date_str_nat = latest_actual_data_nat['date'].strftime('%B %Y')
                        latest_val_nat = latest_actual_data_nat.get('value', 0)
                        mom_diff_nat_val = latest_actual_data_nat.get('value_diff', np.nan)
                        mom_pct_nat_val = latest_actual_data_nat.get('value_pct_change', np.nan)
                        yoy_diff_nat_val = latest_actual_data_nat.get('value_yoy_diff', np.nan)
                        yoy_pct_nat_val = latest_actual_data_nat.get('value_yoy_pct_change', np.nan)

                        forecast_rows_nat = nat_df_forecast[nat_df_forecast['forecast'] == True]
                        if not forecast_rows_nat.empty:
                            first_forecast_data_nat = forecast_rows_nat.sort_values(by='date').iloc[0]
                            forecast_val_nat = first_forecast_data_nat.get('value', np.nan)
                            forecast_date_str_nat = first_forecast_data_nat['date'].strftime('%B %Y')
                        else:
                            forecast_val_nat = np.nan
                            forecast_date_str_nat = "N/A"

                        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                        col_n1.metric("Latest Employment", f"{latest_val_nat:,.1f}K",
                                      help=f"Latest data for {latest_date_str_nat}")
                        col_n2.metric("MoM Change",
                                      f"{mom_diff_nat_val:+,.1f}K" if not pd.isna(mom_diff_nat_val) else "N/A",
                                      f"{mom_pct_nat_val:+.2f}%" if not pd.isna(mom_pct_nat_val) else "",
                                      help="Month-over-Month change")
                        col_n3.metric("YoY Change",
                                      f"{yoy_diff_nat_val:+,.1f}K" if not pd.isna(yoy_diff_nat_val) else "N/A",
                                      f"{yoy_pct_nat_val:+.2f}%" if not pd.isna(yoy_pct_nat_val) else "",
                                      help="Year-over-Year change")
                        col_n4.metric("Next Month Forecast",
                                      f"{forecast_val_nat:,.1f}K" if not pd.isna(forecast_val_nat) else "N/A",
                                      help=f"Forecast for {forecast_date_str_nat}")
                    else:
                        st.warning("Could not retrieve latest data points for national key indicators.")

                    # National Charts
                    st.subheader("Trend & Forecast")
                    fig_nat_line = px.line(
                        nat_df_forecast, x='date', y='value', color='forecast',
                        labels={'date': 'Date', 'value': 'Employment (Thousands)', 'forecast': 'Data Type'},
                        title=f"Employment Trend for {national_label}"
                    )
                    fig_nat_line.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%B %Y}<br>Employment: %{y:,.1f}K<extra></extra>"
                    )
                    fig_nat_line.update_layout(legend_title_text='Data Type', hovermode="x unified")
                    st.plotly_chart(fig_nat_line, use_container_width=True)

                    st.subheader("Monthly % Change")
                    nat_df_actual_mom = nat_df_forecast[(nat_df_forecast['forecast'] == False) & nat_df_forecast['value_pct_change'].notna()].copy()
                    if not nat_df_actual_mom.empty:
                        fig_nat_bar = px.bar(
                            nat_df_actual_mom, x='date', y='value_pct_change',
                            labels={'date': 'Date', 'value_pct_change': 'MoM % Change'},
                            title=f"Month-over-Month % Change for {national_label}"
                        )
                        fig_nat_bar.update_traces(hovertemplate="Date: %{x|%B %Y}<br>MoM Change: %{y:.2f}%<extra></extra>")
                        st.plotly_chart(fig_nat_bar, use_container_width=True)
                    else:
                        st.info("Not enough data to display Monthly % Change chart.")

                else:
                    st.warning(f"No data found for the selected national series: {national_label}")

        st.divider()

        # --- Latest National Industry Data Section ---
        st.header("Latest National Industry Data")
        try:
            latest_nat_data = df[(df['is_national'] == True) & (df['date'] == latest_date_overall)].copy()
            latest_nat_data = latest_nat_data.sort_values(by='label')

            if not latest_nat_data.empty:
                latest_date_nat_str = latest_date_overall.strftime('%B %Y')
                st.caption(f"Latest available data: {latest_date_nat_str}")

                display_cols = ['label', 'value', 'value_diff', 'value_pct_change', 'value_yoy_diff', 'value_yoy_pct_change']
                latest_nat_display = latest_nat_data[display_cols].rename(columns={
                    'label': 'Industry', 'value': 'Employment (K)',
                    'value_diff': 'MoM Diff (K)', 'value_pct_change': 'MoM % Change',
                    'value_yoy_diff': 'YoY Diff (K)', 'value_yoy_pct_change': 'YoY % Change'
                })

                st.dataframe(latest_nat_display, use_container_width=True, hide_index=True,
                             column_config={
                                 "Employment (K)": st.column_config.NumberColumn(format="%.1f"),
                                 "MoM Diff (K)": st.column_config.NumberColumn(format="%+.1f"),
                                 "MoM % Change": st.column_config.NumberColumn(format="%+.2f%%"),
                                 "YoY Diff (K)": st.column_config.NumberColumn(format="%+.1f"),
                                 "YoY % Change": st.column_config.NumberColumn(format="%+.2f%%"),
                             })
            else:
                st.warning(f"No national industry data found for {latest_date_overall.strftime('%B %Y')}")

        except Exception as table_e:
            st.error(f"Error displaying national data table: {table_e}")

        # --- Data Quality and Source Information ---
        with st.expander("📊 Data Quality & Source Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Data Freshness:**")
                st.write(f"• Latest data: {latest_date_overall.strftime('%B %Y')}")
                st.write(f"• Age: {data_freshness} days")
                st.write(f"• Total records: {len(df):,}")
                
                if 'is_preliminary' in df.columns:
                    prelim_count = df['is_preliminary'].sum()
                    st.write(f"• Preliminary records: {prelim_count:,}")
            
            with col2:
                st.markdown("**Source & Methodology:**")
                st.write("• **Source:** U.S. Bureau of Labor Statistics")
                st.write("• **Survey:** Current Employment Statistics (CES)")
                st.write("• **Frequency:** Monthly")
                st.write("• **Geographic Coverage:** All 50 states + DC")
                st.write("• **Industry Coverage:** Total Nonfarm Employment")

        # --- Interpretation Section ---
        with st.expander("📖 Interpreting the Dashboard", expanded=False):
            st.markdown("""
            This dashboard presents U.S. employment data from the Bureau of Labor Statistics (BLS). Here's how to interpret the different components:

            **🔢 Key Indicators:**
            * **Latest Employment:** Total nonfarm employment (thousands) for the most recent month
            * **MoM Change:** Month-over-Month change vs. previous month (absolute and percentage)
            * **YoY Change:** Year-over-Year change vs. same month previous year (smoother trend indicator)
            * **Next Month Forecast:** Simple linear projection (directional indicator only)

            **📈 Charts:**
            * **Trend Line:** Historical employment levels with forecasted extension
            * **Bar Chart:** Monthly percentage changes showing volatility and patterns
            * **State Map:** Geographic comparison of latest monthly changes

            **⚠️ Important Notes:**
            * Forecasts are simple linear projections and should not be used for policy decisions
            * Preliminary data is subject to revision
            * State totals may not sum to national totals due to different survey methodologies
            * All employment figures are seasonally adjusted unless noted

            **📊 Data Quality Indicators:**
            * Green status: Data is current (≤7 days old)
            * Yellow status: Data is 8-30 days old
            * Red status: Data is >30 days old (refresh recommended)
            """)

    except Exception as ui_error:
        st.error(f"Error in dashboard UI: {ui_error}")
        logging.error(f"UI error: {ui_error}")
        traceback.print_exc()

elif df is None:
    st.error("❌ Failed to load employment data. Please check your API key and try refreshing.")
    logging.error("Dashboard failed: df is None")
else:
    st.warning("⚠️ No employment data available after processing.")
    logging.warning("Dashboard failed: df is empty")

logging.info("--- Streamlit script execution finished ---")
