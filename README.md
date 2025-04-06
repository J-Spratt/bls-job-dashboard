# 📊 U.S. Job Trends Dashboard

An interactive Streamlit web app that visualizes **Total Nonfarm Employment** trends and forecasts across all 50 U.S. states and selected national industries using data from the **Bureau of Labor Statistics (BLS)**.

---

## 🔍 Overview

This project uses live data from the BLS API to show:

- 📈 Employment trend lines by state
- 🔮 6-month linear forecasts
- 📉 Month-over-month % change
- 🗺️ Choropleth map of employment levels by state
- ✅ Cache fallback and freshness check
- ⚙️ Streamlit-powered UI for clean interaction

Built to support workforce analysts, economists, job seekers, and policy professionals with timely insights into labor trends.

---

## 🚀 Features

- **Interactive State Selector**: Explore employment metrics state-by-state
- **Forecasting**: Visualize projected job trends using simple linear regression
- **Dynamic Metrics**: See monthly job gains/losses and percent changes
- **National Map View**: Get a big-picture view of employment levels by region
- **BLS API Integration**: Pulls data directly from the official U.S. BLS system

---

## 📦 Tech Stack

- **Python**
- **Streamlit** – App framework
- **Pandas** – Data wrangling
- **Plotly** – Interactive visualizations
- **Scikit-learn** – Forecasting with linear regression
- **Requests** – API calls

---

## 🧪 Running the App Locally

```bash
git clone https://github.com/J-Spratt/bls-job-dashboard.git
cd bls-job-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```
