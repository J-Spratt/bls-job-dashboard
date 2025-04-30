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

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Streamlit Page Setup ---
st.set_page_config(layout="wide", page_title="U.S. Job Trends Dashboard")
st.markdown("<h1 style='text-align: center;'>U.S. Job Trends Dashboard</h1>", unsafe_allow_html=True)
st.caption(
    "Visualizing Total Nonfarm Employment trends and forecasts by state and selected national industries using BLS data. "
    "Forecasts are based on simple linear projections."
)

# --- Load BLS API Key ---
api_key = None
try:
    api_key = st.secrets["BLS_API_KEY"]
    if not api_key:
        st.error("ERROR: `BLS_API_KEY` found in `secrets.toml` but it is empty.")
        st.stop()
except (FileNotFoundError, KeyError, Exception) as e:
    st.error(f"Error loading secrets: {e}")
    st.stop()

# --- Series Definitions (partial placeholder) ---
state_series_map = { ... }  # Same as original
national_industry_map = { ... }  # Full dictionary omitted for brevity
series_ids = list(state_series_map.keys()) + list(national_industry_map.keys())
all_series_map = {**state_series_map, **national_industry_map}

# --- Data Loading Function ---
@st.cache_data(ttl=21600)
def fetch_bls(series_ids, start_year, end_year, api_key):
    """
    Fetch BLS time series data in batches. Returns a combined DataFrame.
    Parameters:
        series_ids (list): List of BLS series IDs
        start_year (str): Starting year (e.g., '2020')
        end_year (str): Ending year (e.g., '2025')
        api_key (str): Registered BLS API key
    Returns:
        pd.DataFrame or None
    """
    logging.info("--- Running fetch_bls ---")
    headers = {"Content-type": "application/json"}
    all_series_data = []
    num_series = len(series_ids)
    batch_size = 50
    max_batches = (num_series + batch_size - 1) // batch_size

    with st.spinner(f"Fetching BLS data from {start_year} to {end_year}..."):
        for i in range(0, num_series, batch_size):
            current_batch_num = i // batch_size + 1
            batch_ids = series_ids[i : min(i + batch_size, num_series)]
            logging.info(f"Requesting batch {current_batch_num}/{max_batches}: {len(batch_ids)} series IDs")

            data_payload = {
                "seriesid": batch_ids,
                "startyear": start_year,
                "endyear": end_year,
                "registrationkey": api_key,
                "catalog": False,
            }
            try:
                response = requests.post(
                    "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                    data=json.dumps(data_payload),
                    headers=headers,
                    timeout=45,
                )
                if response.status_code != 200:
                    logging.warning(f"Non-200 response: {response.status_code}")
                    continue
                response_json = response.json()
                if response_json.get("status") != "REQUEST_SUCCEEDED":
                    logging.warning(f"Failed status: {response_json.get('message')}")
                    continue

                for s in response_json.get("Results", {}).get("series", []):
                    sid = s.get("seriesID")
                    label = all_series_map.get(sid, sid)
                    for item in s.get("data", []):
                        try:
                            if item.get("period", "").startswith("M") and item.get("period") != "M13":
                                val = float(item["value"].replace(",", ""))
                                date = pd.to_datetime(f"{item['year']}-{item['period'][1:]}-01")
                                all_series_data.append({
                                    "series_id": sid,
                                    "label": label,
                                    "year": int(item['year']),
                                    "period": item['period'],
                                    "periodName": item['periodName'],
                                    "value": val,
                                    "date": date
                                })
                        except Exception as e:
                            logging.warning(f"Skipping data point: {e}")
                time.sleep(0.5)
            except Exception as e:
                logging.error(f"Batch fetch error: {e}")
                traceback.print_exc()

    if not all_series_data:
        st.error("No data retrieved from BLS API. Please try again later or check logs.")
        return None
    return pd.DataFrame(all_series_data)

# --- Forecast Explanation ---
st.caption("Note: Forecasts shown are based on simple linear projections using the past employment trend. These are best for short-term exploration and do not account for seasonality or macroeconomic shifts.")

# --- Data Handling Logic ---
csv_file = "bls_employment_data.csv"
df = None

if os.path.exists(csv_file):
    try:
        df = pd.read_csv(csv_file, parse_dates=["date"])
        if df.empty or 'date' not in df.columns:
            logging.warning("Cached CSV is empty or invalid, refetching data...")
            df = None
        else:
            last_updated = df['date'].max()
            st.sidebar.success(f"Using cached data. Last updated: {last_updated.strftime('%B %d, %Y')}")
    except Exception as e:
        logging.error(f"Failed to load cached data: {e}")
        df = None

# --- Fallback to API fetch only if cache missing or invalid ---
if df is None:
    logging.info("Cache missing or invalid. Pulling fresh data from BLS API.")
    fetch_years_history = 5
    current_year = pd.Timestamp.now().year
    start_year = str(current_year - fetch_years_history)
    end_year = str(current_year)
    df = fetch_bls(series_ids, start_year, end_year, api_key)
    if df is not None:
        try:
            df.to_csv(csv_file, index=False)
            logging.info("Saved fresh data to cache.")
            last_updated = df['date'].max()
            st.sidebar.success(f"Fetched and cached new data. Last updated: {last_updated.strftime('%B %d, %Y')}")
        except Exception as e:
            logging.error(f"Could not save cache: {e}")
