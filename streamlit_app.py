import os
import traceback
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
import json
from sklearn.linear_model import LinearRegression
import numpy as np
import time
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Attempt to clear the cache file on every run ---
# (Leaving this part as is - assuming it's correct)
if os.path.exists("bls_employment_data.csv"):
    try:
        os.remove("bls_employment_data.csv")
        logging.info("Cache file 'bls_employment_data.csv' deleted.")
    except Exception as e:
        logging.error(f"Error deleting cache file: {e}")
else:
    logging.info("Cache file 'bls_employment_data.csv' not found, nothing to delete.")

# ---------------------------
# SETUP
# ---------------------------
st.set_page_config(layout="wide", page_title="U.S. Job Trends Dashboard")

# Add a title and a short description (Professional Tone)
st.markdown("<h1 style='text-align: center;'>U.S. Job Trends Dashboard</h1>", unsafe_allow_html=True)
st.caption("Visualizing Total Nonfarm Employment trends and forecasts by state and selected national industries using BLS data. "
           "Forecasts are based on simple linear projections.")

# --- Load BLS API Key from Streamlit Secrets ---
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

# Series maps (Leaving as is)
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
    "CEU0000000001": "Total Nonfarm - National",  # Added National Total
    "CEU0500000001": "Construction - National",
    "CEU0600000001": "Manufacturing - National",
    "CEU0700000001": "Retail Trade - National",
    "CEU0800000001": "Education & Health Services - National",
    "CEU0900000001": "Leisure & Hospitality - National"
}
series_ids = list(state_series_map.keys()) + list(national_industry_map.keys())


# ---------------------------
# DATA LOADING FUNCTION
# ---------------------------
@st.cache_data(ttl=21600)  # Cache data for 6 hours
def fetch_bls(series_ids_func, start_year_str_func, end_year_str_func, api_key_func):
    """Fetches data from BLS API v2 in batches."""
    logging.info(f"--- Running fetch_bls ---")
    logging.info(f"Fetching data for {len(series_ids_func)} series from BLS API for {start_year_str_func}-{end_year_str_func}...")
    headers = {"Content-type": "application/json"}
    all_series_data = []
    num_series = len(series_ids_func)
    batch_size = 50  # BLS API v2 limit
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
            response = requests.post("https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data_json_str, headers=headers, timeout=30)  # 30 sec timeout
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
                    else:
                        error_detail_msg += f" Response: {error_detail}"
                except json.JSONDecodeError:
                    logging.warning(f"Batch {current_batch_num}: Couldn't decode JSON from error response. Body: {response.text[:500]}")
                    error_detail_msg += " Non-JSON response received."
                st.warning(error_detail_msg)
                continue

            response_json = response.json()

            if response_json.get("status") != "REQUEST_SUCCEEDED":
                error_msgs = response_json.get('message', ['No message provided by API.'])
                error_msgs_str = [str(m) for m in error_msgs if m]
                error_string = '; '.join(error_msgs_str)
                logging.warning(f"BLS API Error in batch {current_batch_num}: {error_string}")
                st.warning(f"BLS API Error (batch {current_batch_num}): {error_string}")
                continue

            if "Results" not in response_json or not response_json.get("Results") or "series" not in response_json["Results"]:
                logging.warning(f"No valid 'Results' or 'series' key found for batch {current_batch_num}.")
                st.info(f"No data returned in batch {current_batch_num}.")
                continue

            batch_records = []
            for s in response_json["Results"]["series"]:
                sid = s.get("seriesID")
                if not sid:
                    logging.warning("Skipping series entry with no seriesID.")
                    continue
                series_label = state_series_map.get(sid, national_industry_map.get(sid, sid))

                if not s.get("data"):
                    logging.warning(f"Series {sid} ({series_label}) has no data points in this response.")
                    continue

                for item in s["data"]:
                    try:
                        if "footnote_codes" in item and item["footnote_codes"]:
                            continue
                        val_str = item.get("value")
                        period = item.get("period")
                        year = item.get("year")
                        periodName = item.get("periodName")
                        if not all([val_str, period, year, periodName]):
                            logging.warning(f"Skipping data point with missing fields: {item}")
                            continue
                        val = float(val_str.replace(",", ""))
                        if period == "M13":
                            continue
                        month_num_str = period[1:]
                        date_str = f"{year}-{month_num_str}-01"
                        parsed_date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
                        if pd.isna(parsed_date):
                            logging.warning(f"Skipping data point with unparseable date: {item}")
                            continue
                        batch_records.append({
                            "series_id": sid, "label": series_label, "year": int(year),
                            "period": period, "periodName": periodName, "value": val,
                            "date": parsed_date
                        })
                    except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                        logging.warning(f"Skipping data point due to parsing error: {e} - Item: '{item}'")

            all_series_data.extend(batch_records)
            logging.info(f"Processed {len(batch_records)} records for batch {current_batch_num}.")
            time.sleep(0.5) # Be nice to the API

        except requests.exceptions.Timeout:
            logging.error(f"Timeout error occurred fetching batch {current_batch_num}.")
            st.error(f"Network Timeout fetching data batch {current_batch_num}. Results may be incomplete.")
            # Continue to next batch if possible, or return partial results
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error occurred fetching batch {current_batch_num}: {e}")
            st.error(f"Network error fetching data: {e}. Results may be incomplete.")
            return None # Critical network error, stop processing
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON response for batch {current_batch_num}: {e}")
            response_text = ""
            try:
                response_text = response.text[:500] # Get first 500 chars of response
            except NameError: # response might not be defined if request failed early
                pass
            logging.error(f"Response Text (start): {response_text}")
            st.error("Error decoding API response. Data fetching failed.")
            return None # Critical JSON error, stop processing
        except Exception as e: # Catch other unexpected errors
            logging.error(f"An unexpected error occurred processing batch {current_batch_num}: {e}")
            traceback.print_exc() # Print full traceback to logs
            st.error("An unexpected error occurred during data fetch. Results may be incomplete.")
            # Depending on severity, you might want to 'continue' or 'return None'

    if not all_series_data:
        logging.warning("--- fetch_bls finished: Failed to retrieve any valid data. ---")
        st.warning("No data was successfully retrieved from the BLS API for the requested series and time period.")
        return None
    else:
        logging.info(f"--- fetch_bls finished: Successfully parsed {len(all_series_data)} records. ---")
        return pd.DataFrame(all_series_data)

# --- Cache Loading Logic ---
st.sidebar.subheader("Data Cache Status")
csv_file = "bls_employment_data.csv"
cache_expiry_days = 45 # Set cache expiry (e.g., 45 days for BLS monthly data)

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
                logging.warning(f"Cache file '{csv_file}' is empty.")
                st.sidebar.warning(f"Cache file '{csv_file}' is empty. Will fetch fresh data.")
                os.remove(csv_file) # Remove empty cache file
                df = None
            else:
                max_cached_date = df["date"].max().date()
                logging.info(f"Cache loaded successfully. Last data: {max_cached_date}")
                st.sidebar.success(f"Using cached data.\nLast data point: {max_cached_date}")
        else:
            logging.warning(f"Cache is older than {cache_expiry_days} days ({cache_age_days:.1f} days). Fetching fresh data.")
            st.sidebar.warning(f"Cache is older than {cache_expiry_days} days ({cache_age_days:.1f} days).\nFetching fresh data.")
            df = None # Indicate cache miss due to expiry
    else:
        logging.info(f"Cache file '{csv_file}' not found.")
        st.sidebar.info(f"Cache file '{csv_file}' not found.\nWill fetch fresh data.")
        df = None # Indicate cache miss

except pd.errors.EmptyDataError:
    logging.warning(f"Cache file '{csv_file}' was empty/invalid.")
    st.sidebar.warning(f"Cache file '{csv_file}' was empty/invalid.\nWill fetch fresh data.")
    try:
        os.remove(csv_file)
        logging.info(f"Removed empty/invalid cache file: {csv_file}")
    except OSError as e:
        logging.error(f"Cache Error: Could not remove empty/invalid {csv_file}: {e}")
    df = None
except FileNotFoundError: # Should be caught by os.path.exists, but good practice
    logging.info(f"Cache file '{csv_file}' not found.")
    st.sidebar.info(f"Cache file '{csv_file}' not found.\nWill fetch fresh data.")
    df = None
except Exception as e: # Catch any other error during cache read
    logging.error(f"Error reading cache: {e}. Trying fresh fetch.")
    st.sidebar.error(f"Error reading cache: {e}.\nTrying fresh fetch.")
    if os.path.exists(csv_file): # Try to remove potentially corrupted cache
        try:
            os.remove(csv_file)
            logging.info(f"Removed potentially corrupted cache file: {csv_file}")
        except OSError as rm_e:
            logging.error(f"Cache Error: Could not remove corrupted {csv_file}: {rm_e}")
    df = None

# --- API Data Fetching Logic (if cache miss or expired) ---
if df is None:
    logging.info("df is None, attempting API fetch.")
    st.sidebar.markdown("---")
    st.sidebar.info("Fetching data from BLS API...")

    # Define years to fetch (e.g., last 3 full years + current year)
    fetch_years_history = 3
    current_year = pd.Timestamp.now().year
    start_year = str(current_year - fetch_years_history)
    end_year = str(current_year)

    with st.spinner(f"Fetching BLS data ({start_year}-{end_year})... This may take a moment."):
        fetched_data_successfully = False
        try:
            # Ensure api_key and series_ids are available
            if 'api_key' in locals() or 'api_key' in globals():
                if 'series_ids' in locals() and series_ids:
                    logging.info(f"Calling fetch_bls for years {start_year}-{end_year}...")
                    df_fetched = fetch_bls(series_ids, start_year, end_year, api_key)

                    logging.info(f"fetch_bls returned type: {type(df_fetched)}")
                    if isinstance(df_fetched, pd.DataFrame):
                        logging.info(f"fetch_bls returned DataFrame with {len(df_fetched)} rows.")
                        if not df_fetched.empty:
                            df = df_fetched
                            fetched_data_successfully = True
                        else:
                            logging.warning("fetch_bls returned an empty DataFrame.")
                            st.warning("API returned no data for the requested period/series.")
                    elif df_fetched is None:
                        logging.warning("fetch_bls returned None, indicating a failure.")
                        # Error messages should have been shown within fetch_bls
                    else:
                        # Should not happen if fetch_bls has correct return types
                        logging.error(f"fetch_bls returned unexpected type: {type(df_fetched)}")
                        st.error("Received unexpected data type from API fetch function.")
                else:
                    st.error("Series IDs list is not defined or empty. Cannot fetch.")
                    logging.error("Error: series_ids not defined or empty before calling fetch_bls.")
            else:
                st.error("API Key variable ('api_key') not found. Check secrets loading.")
                logging.error("API Key variable not found.")

        # Catch unexpected errors during the *call* to fetch_bls itself
        except Exception as api_fetch_error:
            st.sidebar.error(f"Critical error during API fetch call: {api_fetch_error}")
            logging.error(f"--- Critical Error calling fetch_bls ---")
            traceback.print_exc()
            # df remains None

    logging.info("--- Processing API fetch results ---")
    # Save to cache if fetch was successful
    if fetched_data_successfully:
        try:
            logging.info(f"API fetch successful. Attempting to save {len(df)} rows to cache: {csv_file}")
            # Ensure date column is datetime before saving
            df['date'] = pd.to_datetime(df['date'])
            df.to_csv(csv_file, index=False)
            st.sidebar.success(f"Fresh data saved to cache.")
            logging.info(f"Cache save successful: {csv_file}")
        except Exception as e:
            st.sidebar.warning(f"Failed to save fetched data to cache {csv_file}: {e}")
            logging.error(f"Cache save failed: {e}")
    # If fetch failed and df is still None, stop the app
    elif df is None:
        st.error("Data fetching failed. Cannot display dashboard. Check API key and network connection.")
        logging.error("API fetch resulted in no usable data (df is None or empty).")
        st.stop() # Stop execution if no data could be fetched


# ---------------------------
# DATA CLEANING & PREPARATION
# ---------------------------
if df is not None and not df.empty:
    logging.info("--- Starting Data Cleaning & Preparation ---")
    try:
        # Convert types and handle potential errors
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        # Use 'integer' for potentially large years, handle NaNs later if needed
        df['year'] = pd.to_numeric(df['year'], errors='coerce', downcast='integer')

        # Drop rows with NaNs in essential columns *after* type conversion
        rows_before = len(df)
        df.dropna(subset=['date', 'value', 'year', 'label', 'series_id'], inplace=True)
        rows_after = len(df)
        if rows_before > rows_after:
            logging.info(f"Removed {rows_before - rows_after} rows due to NaN values in essential columns.")

        # Filter out annual averages (M13) or other unwanted periods
        rows_before = len(df)
        df = df[df["period"] != "M13"].copy() # Use .copy() to avoid SettingWithCopyWarning
        rows_after = len(df)
        if rows_before > rows_after:
            logging.info(f"Removed {rows_before - rows_after} rows with period M13.")

        # Filter out potentially erroneous non-positive values
        rows_before = len(df)
        df = df[df["value"] > 0].copy() # Use .copy()
        rows_after = len(df)
        if rows_before > rows_after:
            logging.info(f"Removed {rows_before - rows_after} rows with non-positive employment values.")

        if df.empty:
             st.error("No valid data remaining after cleaning. Cannot proceed.")
             logging.error("DataFrame became empty after cleaning steps.")
             st.stop()

        logging.info(f"Dataframe shape after initial cleaning: {df.shape}")

        # Sort for calculations
        df = df.sort_values(by=["label", "date"]).reset_index(drop=True)

        # Calculate difference and percentage change
        df["value_diff"] = df.groupby("label")["value"].diff()
        df["value_pct_change"] = df.groupby("label")["value"].pct_change() * 100
        logging.info("Calculated value_diff and value_pct_change.")

        # Create State Abbreviation column (handle potential missing states gracefully)
        state_abbrev_map = {
            "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
            "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
            "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
            "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
            "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
            "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
            "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
            "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
            "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
            "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
            "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
            "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI",
            "Wyoming": "WY"
        }
        # Extract state name robustly, handle cases where it might not match
        df["state_full"] = df["label"].str.extract(r"^([a-zA-Z\s\.]+?)\s+-\s+Total Nonfarm$", expand=False).str.strip()
        df["state"] = df["state_full"].map(state_abbrev_map) # Map to abbreviation, results in NaN if no match
        logging.info("Attempted to extract state abbreviations.")
        state_rows = df['state_full'].notna()
        logging.info(f"Total state rows identified: {state_rows.sum()}")
        logging.info(f"Rows with missing state abbreviation (among state rows): {df.loc[state_rows, 'state'].isnull().sum()}")

        logging.info("--- Data Cleaning & Preparation Complete ---")

    except Exception as clean_e:
        logging.error(f"--- Error during Data Cleaning ---")
        st.error(f"Error during data cleaning/preparation: {clean_e}")
        traceback.print_exc()
        df = None # Set df to None to prevent UI from trying to render
        st.stop() # Stop execution


# ---------------------------
# FORECASTING FUNCTION
# ---------------------------
def add_forecast(df_subset, months=6):
    """Adds a simple linear forecast to a dataframe subset."""
    # Ensure working with a copy and sorted data
    df_subset = df_subset.copy().sort_values("date").reset_index(drop=True)
    label_name = df_subset['label'].iloc[0] if not df_subset.empty else "Selected Series"

    # Check for sufficient data points *before* calculations
    if df_subset.empty or len(df_subset) < 2:
        logging.warning(f"Skipping forecast for '{label_name}': Not enough data points ({len(df_subset)}).")
        df_subset["forecast"] = False # Add column even if no forecast
        return df_subset

    # Create time series integer index for regression
    df_subset["ts_int"] = np.arange(len(df_subset))

    # Prepare data for regression, dropping NaNs *locally* for the model
    subset_clean = df_subset.dropna(subset=['ts_int', 'value'])
    if len(subset_clean) < 2:
        logging.warning(f"Skipping forecast for '{label_name}': Not enough clean data points for model ({len(subset_clean)}).")
        df_subset["forecast"] = False
        return df_subset

    # Fit the linear regression model
    model = LinearRegression()
    try:
        X = subset_clean[["ts_int"]]
        y = subset_clean["value"]
        model.fit(X, y)
        logging.info(f"Forecast model fitted for '{label_name}'.")
    except Exception as forecast_fit_e:
        st.error(f"Error fitting forecast model for '{label_name}': {forecast_fit_e}")
        logging.error(f"Error fitting forecast model for {label_name}: {forecast_fit_e}")
        traceback.print_exc()
        df_subset["forecast"] = False # Indicate forecast failed
        return df_subset # Return original data with forecast flag

    # Generate future predictions
    try:
        last_date = df_subset["date"].iloc[-1]
        # Generate future dates starting the month *after* the last date
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=months, freq='MS') # 'MS' for Month Start
        last_ts_int = df_subset["ts_int"].iloc[-1]
        future_ints = np.arange(last_ts_int + 1, last_ts_int + 1 + months)

        future_preds = model.predict(future_ints.reshape(-1, 1))
        # Ensure forecast values are not negative (optional, depends on context)
        future_preds[future_preds < 0] = 0

        # Create DataFrame for future predictions
        future_data = {
            "date": future_dates,
            "value": future_preds,
            "label": label_name,
            "forecast": True, # Flag these rows as forecast
            # Carry over relevant identifiers if they exist
            "series_id": df_subset["series_id"].iloc[0] if 'series_id' in df_subset.columns else None,
            "state_full": df_subset["state_full"].iloc[0] if 'state_full' in df_subset.columns and pd.notna(df_subset["state_full"].iloc[0]) else None,
            "state": df_subset["state"].iloc[0] if 'state' in df_subset.columns and pd.notna(df_subset["state"].iloc[0]) else None,
            # Add date parts for consistency if needed elsewhere
            "year": future_dates.year,
            "periodName": future_dates.strftime('%B'),
            "period": future_dates.strftime('M%m')
        }
        df_future = pd.DataFrame(future_data)

        # Mark original data points as not forecast
        df_subset["forecast"] = False
        # Combine original data with forecast data
        df_combined = pd.concat([df_subset, df_future], ignore_index=True)
        logging.info(f"Forecast generated for '{label_name}' for {months} months.")
        return df_combined

    except Exception as forecast_pred_e:
        st.error(f"Error predicting forecast for '{label_name}': {forecast_pred_e}")
        logging.error(f"Error predicting forecast for {label_name}: {forecast_pred_e}")
        traceback.print_exc()
        df_subset["forecast"] = False # Indicate forecast failed
        return df_subset # Return original data


# ---------------------------
# === MAIN DASHBOARD UI ===
# ---------------------------
# Ensure df exists and is not empty before attempting to build UI
if df is not None and not df.empty:
    logging.info("--- Starting UI and Visuals ---")
    # --- THIS IS THE MAIN TRY BLOCK FOR THE UI ---
    try: # <--- START of the main UI try block (around line 514 in original)
        # --- State Analysis Section ---
        st.header("State Employment Analysis")

        # Check if 'label' column exists before using it
        if "label" in df.columns:
            # Filter labels more robustly - ensure it's state AND 'Total Nonfarm'
            state_labels_available = sorted(
                df[df["label"].str.contains("Total Nonfarm", na=False) &
                   df["state"].notna() # Check if state abbreviation was successfully mapped
                  ]["label"].unique()
            )
        else:
            state_labels_available = []
            st.warning("Column 'label' not found in the data. Cannot display state analysis.")
            logging.warning("UI: 'label' column missing.")

        # Proceed only if state labels are available
        if not state_labels_available:
            if "label" in df.columns: # Only show this if label column exists but no valid states found
                 st.warning("No state-level 'Total Nonfarm' data with valid state names found to display.")
                 logging.warning("UI: No valid state labels found for selectbox.")
        else:
            # Use a key for the selectbox for better state management
            state_label = st.selectbox(
                "Select a State:",
                state_labels_available,
                key="state_selector",
                help="Choose a state to view detailed trends and forecasts."
            )

            if state_label: # If a state is selected
                # Filter data for the selected state
                state_df = df[df["label"] == state_label].copy() # Use .copy()

                if not state_df.empty:
                    # Add forecast to the state-specific data
                    state_df_forecast = add_forecast(state_df, months=6) # Use 6 months forecast

                    # Display Key Indicators
                    st.subheader(f"Key Indicators: {state_label}")
                    # Get the latest *actual* data point (not forecast)
                    latest_actual_data = state_df_forecast[state_df_forecast['forecast'] == False].sort_values(by='date').iloc[-1]

                    # Safely format values, providing defaults if columns missing
                    latest_employment_val = latest_actual_data.get('value', 0)
                    latest_employment_str = f"{latest_employment_val:,.1f}K" # Format as thousands with 1 decimal place
                    latest_date_str = latest_actual_data['date'].strftime('%B %Y') if pd.notna(latest_actual_data.get('date')) else "N/A"

                    # Use columns for layout
                    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

                    col_metric1.metric("Latest Employment", latest_employment_str, help=f"Latest actual data ({latest_date_str})")

                    # Safely get MoM change values and format them
                    mom_change_val = latest_actual_data.get('value_diff', None)
                    mom_change_pct = latest_actual_data.get('value_pct_change', None)

                    mom_change_val_str = f"{mom_change_val:,.1f}K" if mom_change_val is not None else "N/A"
                    mom_delta_val_str = f"{mom_change_val:+,.1f}K" if mom_change_val is not None else None # Add sign for delta

                    mom_change_pct_str = f"{mom_change_pct:.2f}%" if mom_change_pct is not None else "N/A"
                    mom_delta_pct_str = f"{mom_change_pct:+.2f}%" if mom_change_pct is not None else None # Add sign for delta

                    col_metric2.metric("Month-over-Month Change (Value)", mom_change_val_str, delta=mom_delta_val_str, help="Change from the previous month.")
                    col_metric3.metric("Month-over-Month Change (%)", mom_change_pct_str, delta=mom_delta_pct_str, help="Percentage change from the previous month.")

                    # --- Add code for the 4th metric (e.g., Year-over-Year change) if desired ---
                    # (Placeholder - requires calculating YoY change in data prep)
                    col_metric4.metric("Year-over-Year Change (%)", "N/A", help="YoY Change calculation not implemented.")

                    # --- Display Line Chart ---
                    st.subheader(f"Employment Trend & Forecast: {state_label}")

                    # Create the line chart using Plotly Express
                    fig_state = px.line(
                        state_df_forecast,
                        x='date',
                        y='value',
                        color='forecast', # Color lines based on whether it's actual (False) or forecast (True)
                        title=f"Total Nonfarm Employment: {state_label}",
                        labels={'date': 'Date', 'value': 'Employment (Thousands)', 'forecast': 'Data Type'},
                        markers=True # Show markers on the points
                    )

                    # Customize hover data
                    fig_state.update_traces(
                        hovertemplate="<b>%{fullData.name}</b><br>" +
                                      "Date: %{x|%B %Y}<br>" +
                                      "Employment: %{y:,.1f}K<extra></extra>" # Format hover value
                    )

                    # Customize legend and axis titles
                    fig_state.update_layout(
                        legend_title_text='Data Type',
                        xaxis_title='Date',
                        yaxis_title='Employment (Thousands)'
                    )

                    # Adjust y-axis format if needed (e.g., show "K" for thousands)
                    # fig_state.update_yaxes(tickformat=",.0fK") # Example: If values are in actual numbers, not thousands

                    st.plotly_chart(fig_state, use_container_width=True)

                else:
                    st.warning(f"No data found for the selected state: {state_label}")
                    logging.warning(f"UI: No data found for selected state: {state_label}")
            else:
                st.info("Select a state from the dropdown above to see details.")


        # --- You would continue adding other UI elements here ---
        # For example: National Industry Analysis Section
        # st.header("National Industry Analysis")
        # ... similar logic with selectbox for industry_label ...
        # ... display metrics and chart for selected industry ...

        # --- For example: Map Visualization ---
        # st.header("State Employment Map (Latest Month)")
        # ... logic to get latest data for all states ...
        # ... create choropleth map using px.choropleth ...

        # --- END OF THE CODE WITHIN THE MAIN UI 'TRY' BLOCK ---
        # Assume all your plot generation, st.write, st.dataframe, etc. for the main page ends here.

    # --- VVV THIS IS THE FIX VVV ---
    # Add the 'except' block immediately after the end of the 'try' block's indented code.
    # Make sure this 'except' line has the SAME indentation as the 'try' line on ~514.
    except Exception as ui_error:
        st.error(f"An error occurred while building the dashboard UI: {ui_error}")
        logging.error(f"--- Error in Main Dashboard UI ---")
        # Print the full traceback to the Streamlit logs/console for debugging
        traceback.print_exc()
        st.warning("Some parts of the dashboard might not be displayed correctly.")
    # --- ^^^ END OF THE FIX ^^^ ---

# Handle the case where the DataFrame is None or empty *before* the UI block
elif df is None:
     logging.error("UI: Skipping UI build because df is None (likely due to fetch/cache/clean error).")
     # Error messages should have been displayed earlier (fetch/cache/clean sections)
     # st.error("Could not load data. Dashboard cannot be displayed.") # Optional redundant message
else: # df exists but is empty after initial check
     logging.warning("UI: Skipping UI build because DataFrame is empty.")
     st.warning("No data available to display.")

logging.info("--- Streamlit script execution finished ---")
