# 📊 U.S. Job Trends Dashboard (BLS API + Streamlit)

This is an interactive Streamlit dashboard that visualizes monthly job trends, gains, and losses by U.S. state and sector using data from the Bureau of Labor Statistics (BLS).

### 🚀 Features
- 🔽 Filter by state/sector
- 📈 View employment trends over time
- 📉 See monthly job change + % change (dual-axis)
- 🗺️ Animated choropleth map showing job growth/loss by state
- 🔐 API key securely managed via Streamlit secrets

---

### 📁 Folder Structure
```
bls-dashboard/
├── streamlit_app.py
├── .streamlit/
│   └── secrets.toml         # ← Your BLS API key goes here
├── requirements.txt
├── README.md
└── .gitignore
```

---

### 🧪 How to Run Locally

1. Clone this repo  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your BLS API key to `.streamlit/secrets.toml`:
   ```toml
   BLS_API_KEY = "your-real-api-key"
   ```

4. Launch the app:
   ```bash
   streamlit run streamlit_app.py
   ```

---

### 🌐 Deploy to Streamlit Cloud

1. Push to a **public GitHub repo**
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub, select the repo
4. Add your API key under **Settings > Secrets**:
   ```
   BLS_API_KEY = "your-real-api-key"
   ```

---

### 📬 Contact
Made by John • Built with Streamlit, Plotly, and BLS API
