# Trader Performance vs Market Sentiment

Analysing how the Crypto Fear & Greed Index shapes trader behaviour and daily PnL across on-chain accounts, with a predictive model and a lightweight dashboard.

---

## Project layout

```
├── historical_data.csv       # raw trade-level data (211k rows)
├── fear_greed_index.csv      # daily Fear & Greed index
├── Cleaned_Data.csv          # merged, aggregated output (2340 rows)
│
├── notebook.ipynb            # data cleaning & feature engineering
├── eda.ipynb                 # exploratory analysis
├── model.ipynb               # Random Forest + KMeans clustering
│
├── random_forest_model.pkl   # saved classifier
├── kmeans_model.pkl          # saved clustering model
├── scaler.pkl                # saved StandardScaler
│
└── app.py                    # Streamlit dashboard
```

---

## Setup

**Python 3.10+ recommended.**

```bash
# 1. clone / download the repo, then:
pip install pandas scikit-learn matplotlib seaborn streamlit joblib
```

No virtual environment is required but you can use one if you prefer.

---

## How to run

### Step 1 — Data cleaning
Open and run `notebook.ipynb` top to bottom.  
Reads `historical_data.csv` + `fear_greed_index.csv`, aggregates trades to a daily per-account level and merges in sentiment. Outputs `Cleaned_Data.csv`.

### Step 2 — EDA
Open and run `eda.ipynb`.  
Explores sentiment distribution, how trading behaviour changes across Fear/Greed regimes, and a quick correlation check.

### Step 3 — Modelling
Open and run `model.ipynb`.  
Trains a Random Forest classifier (next-day PnL bucket) and a KMeans clustering model (trader archetypes). Saves three `.pkl` files to the project root.

### Step 4 — Dashboard
```bash
streamlit run app.py
```
Opens on `http://localhost:8501`.  
Loads the pickled models and CSV directly — no uploads needed.

---

## Notes
- All four files (`Cleaned_Data.csv`, `random_forest_model.pkl`, `kmeans_model.pkl`, `scaler.pkl`) must be in the **same folder** as `app.py` when running the dashboard.
- The notebooks must be run **in order** (notebook → eda → model) since each step depends on the output of the previous one.
