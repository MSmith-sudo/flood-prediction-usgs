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
