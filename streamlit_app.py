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
    "SMU56000000000000001": "Wyoming - Total Nonfarm"
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

# --- Data Loading Function ---
@st.cache_data(ttl=21600)
def fetch_bls(series_ids, start_year, end_year, api_key):
    ...  # No changes to the function here

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
    with st.spinner(f"Fetching BLS data from {start_year} to {end_year}..."):
        df = fetch_bls(series_ids, start_year, end_year, api_key)
    if df is not None:
        try:
            df.to_csv(csv_file, index=False)
            logging.info("Saved fresh data to cache.")
            last_updated = df['date'].max()
            st.sidebar.success(f"Fetched and cached new data. Last updated: {last_updated.strftime('%B %d, %Y')}")
        except Exception as e:
            logging.error(f"Could not save cache: {e}")
