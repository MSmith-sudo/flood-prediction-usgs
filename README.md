# Next-Day Streamflow Forecast (AI × Hydrology)

Predicts tomorrow’s daily mean discharge (streamflow) using USGS NWIS Daily Values (DV) data.

## What it does
- Searches for North Carolina USGS stream gages with Daily Values discharge (parameter 00060, statistic 00003)
- Downloads daily mean streamflow for 2018–2024
- Builds time-series features (lags + rolling stats + seasonality)
- Trains a Random Forest regressor to forecast next-day discharge
- Saves a prediction plot (`streamflow_pred.png`)

## Output
- `streamflow_pred.png`
- Prints test MAE in cfs

## Tech
Python, requests, pandas, scikit-learn, matplotlib

# Rainfall-Driven Turbidity Risk Detection (USGS)

A machine learning pipeline for detecting **high turbidity (water-quality risk) periods** using **USGS instantaneous (IV) turbidity data** combined with **rainfall measurements**.

This project focuses on how **precipitation events drive turbidity spikes**, a common indicator of sediment transport, runoff, and potential water-quality degradation.

---

## 🚰 What the Project Does

- Automatically discovers **North Carolina USGS stream gages** that report:
  - Turbidity (parameter `63680`)
  - Precipitation (parameter `00045`)
- Downloads **hourly instantaneous data** from the USGS NWIS API
- Aggregates and synchronizes rainfall + turbidity time series
- Engineers rainfall-driven and temporal features
- Trains a **Random Forest classifier** to flag **high turbidity risk periods**
- Evaluates performance using:
  - Confusion matrix
  - Precision, recall, and F1-score
- Produces a visualization highlighting predicted risk events

---

## 🧠 Modeling Approach

### Problem Framing
This is framed as a **binary risk classification problem**:
- `0` → Normal turbidity conditions  
- `1` → Elevated turbidity risk (above a defined threshold)

The model is designed as a **screening tool**, prioritizing **high recall** to minimize missed contamination events.

### Features
- Rolling rainfall totals (6h, 24h)
- Turbidity rolling statistics (mean, std)
- Rate-of-change features
- Time-of-day and seasonal indicators

### Model
- **RandomForestClassifier**
- Time-aware train/test split (no shuffling)
- Class balancing enabled

---

## 📊 Example Results

On a real USGS monitoring site, the model achieved:
- **High-risk recall ≈ 97%**
- **Overall accuracy ≈ 99%**

This indicates strong performance for identifying rainfall-driven turbidity events without excessive false alarms.

---

## 📈 Output

- `turbidity_risk.png`  
  Visualization of turbidity values with predicted high-risk periods highlighted.

---

## 🛠️ Tech Stack

- Python
- requests
- pandas / numpy
- scikit-learn
- matplotlib
- USGS NWIS Water Services API

---

## 📌 Notes

- Thresholds (e.g., turbidity ≥ 10) are configurable and intended as **conservative screening values**, not regulatory limits.
- Data availability varies by site; the script automatically selects a site with overlapping turbidity and precipitation records.

---

## 🔍 Why This Matters

Rainfall-driven turbidity spikes are a major concern for:
- Drinking water treatment
- Aquatic habitat health
- Watershed management
- Environmental monitoring systems

This project demonstrates how **machine learning can support real-time water-quality risk detection** using publicly available data.

