import os
import traceback
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go # Needed for more customization if required
import requests
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# --- Current Time Context ---
# Current time is Sunday, April 6, 2025 at 12:48:45 AM CDT.

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Optional Cache Deletion (Comment out if not needed on every run) ---
# if os.path.exists("bls_employment_data.csv"):
#     try:
#         os.remove("bls_employment_data.csv")
#         logging.info("Cache file 'bls_employment_data.csv' deleted.")
#     except Exception as e:
#         logging.error(f"Error deleting cache file: {e}")
# else:
#     logging.info("Cache file 'bls_employment_data.csv' not found, nothing to delete.")

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(layout="wide", page_title="U.S. Job Trends Dashboard")

# Add a title and a short description
st.markdown("<h1 style='text-align: center;'>U.S. Job Trends Dashboard</h1>", unsafe_allow_html=True)
st.caption("Visualizing Total Nonfarm Employment trends and forecasts by state and selected national industries using BLS data. "
           "Forecasts are based on simple linear projections.")

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
@st.cache_data(ttl=21600)  # Cache data for 6 hours
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
                error_detail_msg = f"API request failed for batch {current_batch_num}. Status: {response.status_code}."
                try:
                    error_detail = response.json()
                    logging.warning(f"Batch {current_batch_num}: Error Response Body: {error_detail}")
                    api_messages = error_detail.get('message', [])
                    if api_messages:
                        error_detail_msg += f" Detail: {'; '.join(m for m in api_messages if m)}"
                    elif error_detail:
                         error_detail_msg += f" Response: {error_detail}"
                    else:
                         error_detail_msg += f" Body: {response.text[:200]}"
                except json.JSONDecodeError:
                    logging.warning(f"Batch {current_batch_num}: Couldn't decode JSON from error response. Body: {response.text[:500]}")
                    error_detail_msg += " Non-JSON response received."
                st.warning(error_detail_msg)
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

                for item in s["data"]:
                    try:
                        if "footnote_codes" in item and "P" in item["footnote_codes"]:
                            continue

                        val_str = item.get("value")
                        period = item.get("period")
                        year = item.get("year")
                        periodName = item.get("periodName")

                        if not all([val_str, period, year, periodName]):
                            logging.warning(f"Skipping data point with missing fields: {item} for Series {sid}")
                            continue
                        if period.startswith('Q') or period == "M13":
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
                            "date": parsed_date
                        })
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                        logging.warning(f"Skipping data point for series {sid} due to parsing error: {e} - Item: '{item}'")

            all_series_data.extend(batch_records)
            logging.info(f"Processed {len(batch_records)} valid monthly records for batch {current_batch_num}.")
            time.sleep(0.5)

        except requests.exceptions.Timeout:
            logging.error(f"Timeout error occurred fetching batch {current_batch_num}.")
            st.error(f"Network Timeout fetching data batch {current_batch_num}. Results may be incomplete.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred fetching batch {current_batch_num}: {e}")
            st.error(f"Network error fetching data: {e}. Results may be incomplete.")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response for batch {current_batch_num}: {e}")
            response_text = ""
            try: response_text = response.text[:500]
            except NameError: pass
            logging.error(f"Response Text (start): {response_text}")
            st.error("Error decoding API response. Data fetching failed.")
            return None
        except Exception as e:
            logging.error(f"An unexpected error occurred processing batch {current_batch_num}: {e}")
            traceback.print_exc()
            st.error("An unexpected error occurred during data fetch. Results may be incomplete.")

    if not all_series_data:
        logging.warning("--- fetch_bls finished: Failed to retrieve any valid data. ---")
        st.warning("No data was successfully retrieved from the BLS API for the requested series and time period.")
        return None
    else:
        logging.info(f"--- fetch_bls finished: Successfully parsed {len(all_series_data)} valid monthly records. ---")
        return pd.DataFrame(all_series_data)

# --- Cache Loading Logic ---
st.sidebar.subheader("Data Cache Status")
csv_file = "bls_employment_data.csv"
cache_expiry_days = 1 # Cache expires daily

df = None
try:
    if os.path.exists(csv_file):
        cache_info = os.stat(csv_file)
        cache_age_seconds = time.time() - cache_info.st_mtime
        cache_age_days = cache_age_seconds / (60 * 60 * 24)
        logging.info(f"Cache file found: {csv_file}. Age: {cache_age_days:.1f} days.")

        if cache_age_days < cache_expiry_days:
            logging.info("Cache is within expiry, attempting to load.")
            df = pd.read_csv(csv_file, parse_dates=["date"])
            if df.empty:
                logging.warning(f"Cache file '{csv_file}' is empty. Deleting.")
                st.sidebar.warning(f"Cache file was empty. Fetching fresh data.")
                os.remove(csv_file)
                df = None
            else:
                required_cols = ['series_id', 'label', 'year', 'period', 'periodName', 'value', 'date']
                if not all(col in df.columns for col in required_cols):
                     logging.warning(f"Cache file '{csv_file}' is missing required columns. Deleting.")
                     st.sidebar.warning(f"Cache file missing columns. Fetching fresh data.")
                     os.remove(csv_file)
                     df = None
                else:
                    max_cached_date = df["date"].max().date()
                    logging.info(f"Cache loaded successfully. Last data point: {max_cached_date}")
                    st.sidebar.success(f"Using cached data.\nLast data: {max_cached_date}")
        else:
            logging.warning(f"Cache is older than {cache_expiry_days} days ({cache_age_days:.1f} days). Fetching fresh data.")
            st.sidebar.warning(f"Cache expired ({cache_age_days:.1f} days old).\nFetching fresh data.")
            df = None
    else:
        logging.info(f"Cache file '{csv_file}' not found.")
        st.sidebar.info(f"Cache file not found.\nFetching fresh data.")
        df = None

except pd.errors.EmptyDataError:
    logging.warning(f"Cache file '{csv_file}' could not be read (EmptyDataError).")
    st.sidebar.warning(f"Cache file was invalid.\nFetching fresh data.")
    if os.path.exists(csv_file): os.remove(csv_file)
    df = None
except FileNotFoundError:
    logging.info(f"Cache file '{csv_file}' not found during load attempt.")
    st.sidebar.info(f"Cache file not found.\nFetching fresh data.")
    df = None
except Exception as e:
    logging.error(f"Error reading cache file '{csv_file}': {e}. Trying fresh fetch.")
    st.sidebar.error(f"Error reading cache: {e}.\nTrying fresh fetch.")
    if os.path.exists(csv_file):
        try:
            os.remove(csv_file)
            logging.info(f"Removed potentially corrupted cache file: {csv_file}")
        except OSError as rm_e:
            logging.error(f"Cache Error: Could not remove corrupted {csv_file}: {rm_e}")
    df = None


# --- API Data Fetching Logic ---
if df is None:
    logging.info("df is None, proceeding with API fetch.")
    st.sidebar.markdown("---")
    st.sidebar.info("Fetching data from BLS API...")

    fetch_years_history = 5
    current_year = pd.Timestamp.now().year
    start_year = str(current_year - fetch_years_history)
    end_year = str(current_year)

    with st.spinner(f"Fetching BLS data ({start_year}-{end_year})... This may take a moment."):
        fetched_data_successfully = False
        try:
            if 'api_key' in locals() or 'api_key' in globals():
                if 'series_ids' in locals() and series_ids:
                    logging.info(f"Calling fetch_bls for years {start_year}-{end_year}...")
                    df_fetched = fetch_bls(series_ids, start_year, end_year, api_key)

                    logging.info(f"fetch_bls returned type: {type(df_fetched)}")
                    if isinstance(df_fetched, pd.DataFrame) and not df_fetched.empty:
                        logging.info(f"fetch_bls returned DataFrame with {len(df_fetched)} rows.")
                        df = df_fetched
                        fetched_data_successfully = True
                    elif isinstance(df_fetched, pd.DataFrame) and df_fetched.empty:
                        logging.warning("fetch_bls returned an empty DataFrame.")
                        st.warning("API returned no data for the requested period/series.")
                    else:
                        logging.warning("fetch_bls did not return a valid DataFrame.")
                else:
                    st.error("Series IDs list ('series_ids') is not defined or empty. Cannot fetch.")
                    logging.error("Error: series_ids not defined or empty before calling fetch_bls.")
            else:
                st.error("API Key variable ('api_key') not found. Check secrets loading.")
                logging.error("API Key variable not found.")

        except Exception as api_fetch_error:
            st.sidebar.error(f"Critical error during API fetch process: {api_fetch_error}")
            logging.error(f"--- Critical Error calling/processing fetch_bls ---")
            traceback.print_exc()

    logging.info("--- Processing API fetch results ---")
    if fetched_data_successfully:
        try:
            logging.info(f"API fetch successful. Saving {len(df)} rows to cache: {csv_file}")
            df['date'] = pd.to_datetime(df['date'])
            df.to_csv(csv_file, index=False)
            st.sidebar.success(f"Fresh data saved to cache.")
            logging.info(f"Cache save successful: {csv_file}")
        except Exception as e:
            st.sidebar.warning(f"Failed to save fetched data to cache {csv_file}: {e}")
            logging.error(f"Cache save failed: {e}")
    elif df is None:
        st.error("Data fetching failed. Cannot display dashboard. Please check logs or try again later.")
        logging.error("API fetch resulted in no usable data (df is None or empty). Stopping.")
        st.stop()


# ---------------------------
# DATA CLEANING & PREPARATION (with YoY)
# ---------------------------
if df is not None and not df.empty:
    logging.info("--- Starting Data Cleaning & Preparation ---")
    try:
        # Convert types
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce', downcast='integer')

        initial_rows = len(df)
        df.dropna(subset=['date', 'value', 'year', 'label', 'series_id'], inplace=True)
        if len(df) < initial_rows:
             logging.info(f"Removed {initial_rows - len(df)} rows due to NaN values in essential columns.")

        initial_rows = len(df)
        df = df[df["value"] > 0].copy()
        if len(df) < initial_rows:
             logging.info(f"Removed {initial_rows - len(df)} rows with non-positive employment values.")

        if df.empty:
             st.error("No valid data remaining after initial cleaning. Cannot proceed.")
             logging.error("DataFrame became empty after initial cleaning steps.")
             st.stop()

        logging.info(f"Dataframe shape after initial cleaning: {df.shape}")

        # Sort for calculations
        df = df.sort_values(by=["label", "date"]).reset_index(drop=True)

        # Calculate MoM change
        df["value_diff"] = df.groupby("label")["value"].diff()
        df["value_pct_change"] = df.groupby("label")["value"].pct_change() * 100
        logging.info("Calculated MoM value_diff and value_pct_change.")

        # Calculate YoY change
        df['value_yoy_lag'] = df.groupby('label')['value'].shift(12)
        df['value_yoy_diff'] = df['value'] - df['value_yoy_lag']
        df['value_yoy_pct_change'] = (df['value_yoy_diff'] / df['value_yoy_lag']) * 100
        df['value_yoy_pct_change'] = df['value_yoy_pct_change'].replace([np.inf, -np.inf], np.nan)
        logging.info("Calculated YoY value_yoy_diff and value_yoy_pct_change.")

        # --- State Abbreviation Mapping ---
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
        logging.info("Attempted to extract state abbreviations.")
        state_rows_mask = df['state_full'].notna()
        logging.info(f"Total rows matching state pattern: {state_rows_mask.sum()}")
        logging.info(f"State rows successfully mapped to abbreviation: {df.loc[state_rows_mask, 'state_abbrev'].notna().sum()}")
        logging.info(f"State rows UNABLE to map to abbreviation: {df.loc[state_rows_mask, 'state_abbrev'].isnull().sum()}")

        # Identify National Series
        df['is_national'] = df['series_id'].isin(national_industry_map.keys())
        logging.info(f"Identified {df['is_national'].sum()} national series rows.")

        logging.info("--- Data Cleaning & Preparation Complete ---")

    except Exception as clean_e:
        logging.error(f"--- Error during Data Cleaning ---")
        st.error(f"Error during data cleaning/preparation: {clean_e}")
        traceback.print_exc()
        df = None
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
        logging.error(f"Error fitting forecast model for {label_name}: {forecast_fit_e}")
        traceback.print_exc()
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
        logging.info(f"Forecast generated for '{label_name}' for {months} months.")
        return df_combined

    except Exception as forecast_pred_e:
        st.error(f"Error predicting forecast for '{label_name}': {forecast_pred_e}")
        logging.error(f"Error predicting forecast for {label_name}: {forecast_pred_e}")
        traceback.print_exc()
        df_subset["forecast"] = False
        return df_subset


# ---------------------------
# === MAIN DASHBOARD UI ===
# ---------------------------
if df is not None and not df.empty:
    logging.info("--- Starting UI and Visuals ---")
    try: # Main try block for UI
        latest_date_overall = df['date'].max()

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
                    latest_actual_data = state_df_forecast[(state_df_forecast['forecast'] == False) & pd.notna(state_df_forecast['value'])].sort_values(by='date').iloc[-1] if not state_df_forecast[state_df_forecast['forecast'] == False].empty else None

                    if latest_actual_data is not None:
                        latest_date_str = latest_actual_data['date'].strftime('%B %Y')
                        latest_val = latest_actual_data.get('value', 0)
                        mom_diff = latest_actual_data.get('value_diff', None)
                        mom_pct = latest_actual_data.get('value_pct_change', None)
                        yoy_diff = latest_actual_data.get('value_yoy_diff', None)
                        yoy_pct = latest_actual_data.get('value_yoy_pct_change', None)

                        first_forecast_data = state_df_forecast[state_df_forecast['forecast'] == True].sort_values(by='date').iloc[0] if not state_df_forecast[state_df_forecast['forecast'] == True].empty else None
                        forecast_val = first_forecast_data.get('value', None) if first_forecast_data is not None else None
                        forecast_date_str = first_forecast_data['date'].strftime('%B %Y') if first_forecast_data is not None else "N/A"

                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        col_m1.metric("Latest Employment", f"{latest_val:,.1f}K", help=f"Latest actual data ({latest_date_str})")
                        col_m2.metric("MoM Change", f"{mom_diff:+,.1f}K" if mom_diff is not None else "N/A", f"{mom_pct:+.2f}%" if mom_pct is not None else "", help="Month-over-Month Change vs Previous Month")
                        col_m3.metric("YoY Change", f"{yoy_diff:+,.1f}K" if yoy_diff is not None else "N/A", f"{yoy_pct:+.2f}%" if yoy_pct is not None else "", help="Year-over-Year Change vs Same Month Previous Year")
                        col_m4.metric("Next Month Forecast", f"{forecast_val:,.1f}K" if forecast_val is not None else "N/A", help=f"Simple linear forecast for {forecast_date_str}")
                    else:
                        st.warning("Could not retrieve latest data points for key indicators.")

                    st.subheader(f"Employment Trend & Forecast")
                    fig_state_line = px.line(
                        state_df_forecast, x='date', y='value', color='forecast',
                        labels={'date': 'Date', 'value': 'Employment (Thousands)', 'forecast': 'Data Type'}, markers=False
                    )
                    fig_state_line.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%B %Y}<br>Employment: %{y:,.1f}K<extra></extra>", connectgaps=False
                    )
                    fig_state_line.update_layout(legend_title_text='Data Type', hovermode="x unified")
                    st.plotly_chart(fig_state_line, use_container_width=True)

                    st.subheader(f"Monthly % Change")
                    state_df_actual_mom = state_df_forecast[(state_df_forecast['forecast'] == False) & state_df_forecast['value_pct_change'].notna()].copy()
                    fig_state_bar = px.bar(
                        state_df_actual_mom, x='date', y='value_pct_change',
                        labels={'date': 'Date', 'value_pct_change': 'MoM % Change'}
                    )
                    fig_state_bar.update_traces(hovertemplate="Date: %{x|%B %Y}<br>MoM Change: %{y:.2f}%<extra></extra>")
                    fig_state_bar.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_state_bar, use_container_width=True)

                else:
                    st.warning(f"No data found for the selected state: {state_label}")
            else:
                st.info("Select a state from the dropdown above to see details.")

        st.divider()

        # --- State Comparison Map Section ---
        st.header("State Comparison Map")
        st.caption("Latest Month-over-Month Percentage Change")
        try:
            # === CORRECTED MAP DATA SELECTION ===
            # 1. Filter the DataFrame for state rows FIRST
            df_states_only = df[df['state_abbrev'].notna()].copy() # Use .copy()

            if not df_states_only.empty:
                # 2. Find the index of the latest date *within the filtered state DataFrame*
                latest_state_indices = df_states_only.groupby('label')['date'].idxmax()
                # 3. Select the latest rows using the valid indices
                latest_state_data = df_states_only.loc[latest_state_indices]
            else:
                latest_state_data = pd.DataFrame() # Handle empty case
            # === END OF FIX ===

            # Ensure we have the MoM % change calculated for these latest points
            latest_state_data = latest_state_data.dropna(subset=['state_abbrev', 'value_pct_change'])

            if not latest_state_data.empty:
                map_latest_date = latest_state_data['date'].max()
                map_latest_date_str = map_latest_date.strftime('%B %Y')
                st.markdown(f"*(Showing data for: {map_latest_date_str})*")

                fig_map = px.choropleth(
                   latest_state_data, locations='state_abbrev', locationmode='USA-states',
                   color='value_pct_change', scope='usa', color_continuous_scale="RdYlGn",
                   range_color=[-2, 2], hover_name='state_full',
                   hover_data={'state_abbrev': False, 'value_pct_change': ':.2f%'}
                )
                fig_map.update_layout(coloraxis_colorbar=dict(title="MoM % Change"))
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("Could not retrieve latest valid MoM % change data for the map.")
                logging.warning("Map generation: No valid latest state data points found after processing.")
        except Exception as map_e:
            st.error(f"An error occurred while generating the state comparison map: {map_e}")
            logging.error(f"Error generating map: {map_e}")
            traceback.print_exc() # Log the full error for debugging

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
                    latest_actual_data_nat = nat_df_forecast[(nat_df_forecast['forecast'] == False) & pd.notna(nat_df_forecast['value'])].sort_values(by='date').iloc[-1] if not nat_df_forecast[nat_df_forecast['forecast'] == False].empty else None

                    if latest_actual_data_nat is not None:
                        latest_date_str_nat = latest_actual_data_nat['date'].strftime('%B %Y')
                        latest_val_nat = latest_actual_data_nat.get('value', 0)
                        mom_diff_nat = latest_actual_data_nat.get('value_diff', None)
                        mom_pct_nat = latest_actual_data_nat.get('value_pct_change', None)
                        yoy_diff_nat = latest_actual_data_nat.get('value_yoy_diff', None)
                        yoy_pct_nat = latest_actual_data_nat.get('value_yoy_pct_change', None)

                        first_forecast_data_nat = nat_df_forecast[nat_df_forecast['forecast'] == True].sort_values(by='date').iloc[0] if not nat_df_forecast[nat_df_forecast['forecast'] == True].empty else None
                        forecast_val_nat = first_forecast_data_nat.get('value', None) if first_forecast_data_nat is not None else None
                        forecast_date_str_nat = first_forecast_data_nat['date'].strftime('%B %Y') if first_forecast_data_nat is not None else "N/A"

                        col_n1, col_n2, col_n3, col_n4 = st.columns(4)
                        col_n1.metric("Latest Employment", f"{latest_val_nat:,.1f}K", help=f"Latest actual data ({latest_date_str_nat})")
                        col_n2.metric("MoM Change", f"{mom_diff_nat:+,.1f}K" if mom_diff_nat is not None else "N/A", f"{mom_pct_nat:+.2f}%" if mom_pct_nat is not None else "", help="Month-over-Month Change vs Previous Month")
                        col_n3.metric("YoY Change", f"{yoy_diff_nat:+,.1f}K" if yoy_diff_nat is not None else "N/A", f"{yoy_pct_nat:+.2f}%" if yoy_pct_nat is not None else "", help="Year-over-Year Change vs Same Month Previous Year")
                        col_n4.metric("Next Month Forecast", f"{forecast_val_nat:,.1f}K" if forecast_val_nat is not None else "N/A", help=f"Simple linear forecast for {forecast_date_str_nat}")
                    else:
                        st.warning("Could not retrieve latest data points for national key indicators.")

                    st.subheader(f"Trend & Forecast")
                    fig_nat_line = px.line(
                        nat_df_forecast, x='date', y='value', color='forecast',
                        labels={'date': 'Date', 'value': 'Employment (Thousands)', 'forecast': 'Data Type'}, markers=False
                    )
                    fig_nat_line.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>Date: %{x|%B %Y}<br>Employment: %{y:,.1f}K<extra></extra>", connectgaps=False
                    )
                    fig_nat_line.update_layout(legend_title_text='Data Type', hovermode="x unified")
                    st.plotly_chart(fig_nat_line, use_container_width=True)

                    st.subheader(f"Monthly % Change")
                    nat_df_actual_mom = nat_df_forecast[(nat_df_forecast['forecast'] == False) & nat_df_forecast['value_pct_change'].notna()].copy()
                    fig_nat_bar = px.bar(
                        nat_df_actual_mom, x='date', y='value_pct_change',
                        labels={'date': 'Date', 'value_pct_change': 'MoM % Change'}
                    )
                    fig_nat_bar.update_traces(hovertemplate="Date: %{x|%B %Y}<br>MoM Change: %{y:.2f}%<extra></extra>")
                    fig_nat_bar.update_layout(hovermode="x unified")
                    st.plotly_chart(fig_nat_bar, use_container_width=True)

                else:
                    st.warning(f"No data found for the selected national series: {national_label}")
            else:
                st.info("Select a national series from the dropdown above to see details.")

        st.divider()

        # --- Interpretation Section ---
        with st.expander("Interpreting the Dashboard", expanded=False):
            st.markdown("""
            This dashboard presents U.S. employment data from the Bureau of Labor Statistics (BLS). Here's how to interpret the different components:

            **1. Key Indicators:**
            * **Latest Employment:** The total nonfarm employment figure (in thousands) for the most recent month available in the selected series.
            * **MoM Change:** The change (Month-over-Month) in employment (absolute thousands and percentage) from the previous month. Positive values indicate growth; negative values indicate decline.
            * **YoY Change:** The change (Year-over-Year) in employment (absolute thousands and percentage) compared to the same month in the previous year. This helps identify longer-term trends, smoothing out monthly volatility.
            * **Next Month Forecast:** A simple estimate of the employment level for the next month, based on a linear regression of the historical data. **Note:** This is a basic projection and may not capture complex economic factors or seasonality accurately.

            **2. Employment Trend & Forecast Chart (Line Chart):**
            * **Y-axis:** Represents Total Employment (in thousands) for the selected series.
            * **X-axis:** Represents the date (monthly).
            * **Solid Line:** Shows the historical employment data over time. Observe the slope to see general trends (upward slope indicates growth, downward indicates decline).
            * **Dashed Line:** Shows the simple linear forecast for the next six months based on the historical trend.

            **3. Monthly % Change Chart (Bar Chart):**
            * **Y-axis:** Represents the percentage change in employment compared to the *previous* month.
            * **X-axis:** Represents the date (monthly).
            * **Bars above zero:** Indicate months where employment grew compared to the prior month.
            * **Bars below zero:** Indicate months where employment declined compared to the prior month. The height of the bar shows the magnitude of the change.

            **4. State Comparison Map (Chloropleth Map):**
            * This map visualizes the most recent Month-over-Month (MoM) percentage change in employment for all available states simultaneously.
            * **Color:** Each state is colored based on its MoM % change value. Refer to the color scale legend on the map (typically green shades indicate growth, red shades indicate decline, with intensity showing magnitude) to compare performance across states for the latest month.

            *The interpretations above apply similarly to both the "State Employment Analysis" and the "National Industry Analysis" sections.*
            """)

        st.divider()

        # --- Latest National Industry Data Section ---
        st.header("Latest National Industry Data")
        try:
            latest_nat_data = df[(df['is_national'] == True) & (df['date'] == latest_date_overall)].copy()
            latest_nat_data = latest_nat_data.sort_values(by='label')

            if not latest_nat_data.empty:
                 latest_date_nat_str = latest_date_overall.strftime('%B %Y')
                 st.caption(f"Showing national industry data for the latest available month: {latest_date_nat_str}")

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
                 st.warning(f"No national industry data found for the latest date ({latest_date_overall.strftime('%B %Y')}).")

        except Exception as table_e:
             st.error(f"Error displaying latest national data table: {table_e}")
             logging.error(f"Error generating latest national table: {table_e}")
             traceback.print_exc()

    # --- Main Exception Handler for UI ---
    except Exception as ui_error:
        st.error(f"An error occurred while building the dashboard UI: {ui_error}")
        logging.error(f"--- Error in Main Dashboard UI ---")
        traceback.print_exc()
        st.warning("Some parts of the dashboard might not be displayed correctly.")

elif df is None:
     logging.error("UI: Skipping UI build because df is None.")
else:
     logging.warning("UI: Skipping UI build because DataFrame is empty.")
     st.warning("No data available to display after loading and cleaning.")

logging.info("--- Streamlit script execution finished ---")
