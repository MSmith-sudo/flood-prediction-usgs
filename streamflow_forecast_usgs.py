import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# -------------------------
# Settings
# -------------------------
STATE = "nc"
PARAM = "00060"      # Discharge (cfs)
STAT_CD = "00003"    # Daily mean
START = "2018-01-01"
END = "2024-12-31"
LOCAL_TZ = "America/New_York"

N_LAGS = 7
ROLL_1 = 7
ROLL_2 = 30

OUT_PLOT = "streamflow_pred.png"


# -----------------------------
# Robust session w/ retries
# -----------------------------
session = requests.Session()
retries = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------
# 1) Get candidate sites in NC that claim discharge DV (00060)
# ---------------------------------------------------------
site_url = "https://nwis.waterservices.usgs.gov/nwis/site/"
site_params = {
    "format": "rdb",
    "stateCd": STATE,
    "parameterCd": PARAM,
    "siteType": "ST",
    "hasDataTypeCd": "dv",
}

print("Searching for NC USGS DV sites that report discharge (00060)...")
resp = session.get(site_url, params=site_params, timeout=(5, 20))
resp.raise_for_status()
text = resp.text

candidates = []
for line in text.splitlines():
    if (
        not line
        or line.startswith("#")
        or line.startswith("agency_cd")
        or line.startswith("5s")
    ):
        continue
    cols = line.split("\t")
    if len(cols) >= 3:
        site_no = cols[1]
        station_nm = cols[2]
        candidates.append((site_no, station_nm))

if not candidates:
    raise RuntimeError("No candidate DV discharge sites found in NC.")

print(f"Found {len(candidates)} candidate sites. Probing for actual DV data...")


# ---------------------------------------------------------
# 2) Probe candidates until we find one that actually returns data
# ---------------------------------------------------------
dv_url = "https://waterservices.usgs.gov/nwis/dv/"
SITE = None
STATION_NM = None
data = None

MAX_PROBES = 50  # bumped slightly for reliability

for i, (site_no, station_nm) in enumerate(candidates[:MAX_PROBES], start=1):
    dv_params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": PARAM,
        "statCd": STAT_CD,
        "startDT": START,
        "endDT": END,
        "siteStatus": "all",
    }

    try:
        r = session.get(dv_url, params=dv_params, timeout=(5, 30))
        r.raise_for_status()
        j = r.json()

        ts = j.get("value", {}).get("timeSeries", [])
        if ts and ts[0].get("values") and ts[0]["values"][0].get("value"):
            SITE = site_no
            STATION_NM = station_nm
            data = j
            print(f"Using working DV site ({i}/{min(MAX_PROBES, len(candidates))}): {SITE} | {STATION_NM}")
            print("DEBUG final URL:", r.url)
            break
        else:
            print(f"Probe {i}: {site_no} returned no DV series in range.")
    except Exception as e:
        print(f"Probe {i}: {site_no} failed ({type(e).__name__}).")

if not SITE:
    raise ValueError(
        f"No working DV discharge site found in first {MAX_PROBES} candidates for {START}..{END}. "
        "Try increasing MAX_PROBES or widening the date range."
    )


# --------------------------------------------
# 3) Build dataframe from DV JSON (THIS WAS MISSING)
# --------------------------------------------
ts = data.get("value", {}).get("timeSeries", [])
values = ts[0]["values"][0]["value"]
df = pd.DataFrame(values)

# Parse timestamps safely: UTC -> local date
df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True)
df["date_local"] = df["dateTime"].dt.tz_convert(LOCAL_TZ).dt.date

# DV value is usually daily mean discharge in cfs
df["flow_cfs"] = pd.to_numeric(df["value"], errors="coerce")

df = df.dropna(subset=["flow_cfs"]).sort_values("date_local")

# Ensure one row per day (DV should already be daily, but keep it safe)
df = df.groupby("date_local", as_index=False)["flow_cfs"].mean()
df["date_local"] = pd.to_datetime(df["date_local"])

if df.empty or len(df) < 300:
    raise ValueError("Not enough daily data returned to train a model.")


# --------------------------------------------
# 4) Feature engineering (lags + rolling stats)
# --------------------------------------------
df = df.sort_values("date_local").reset_index(drop=True)

for k in range(1, N_LAGS + 1):
    df[f"lag_{k}"] = df["flow_cfs"].shift(k)

df[f"roll_mean_{ROLL_1}"] = df["flow_cfs"].rolling(ROLL_1).mean()
df[f"roll_std_{ROLL_1}"] = df["flow_cfs"].rolling(ROLL_1).std()
df[f"roll_mean_{ROLL_2}"] = df["flow_cfs"].rolling(ROLL_2).mean()
df[f"roll_std_{ROLL_2}"] = df["flow_cfs"].rolling(ROLL_2).std()

# Seasonality features
df["day_of_year"] = df["date_local"].dt.dayofyear
df["month"] = df["date_local"].dt.month

# Target: tomorrow's flow
df["target_next_day"] = df["flow_cfs"].shift(-1)

model_df = df.dropna().copy()

feature_cols = (
    [f"lag_{k}" for k in range(1, N_LAGS + 1)]
    + [f"roll_mean_{ROLL_1}", f"roll_std_{ROLL_1}", f"roll_mean_{ROLL_2}", f"roll_std_{ROLL_2}"]
    + ["day_of_year", "month"]
)

X = model_df[feature_cols].to_numpy()
y = model_df["target_next_day"].to_numpy()

# Time-aware split: last 20% test, no shuffle
split_idx = int(len(model_df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = model_df["date_local"].iloc[split_idx:]


# -------------------------
# 5) Train model
# -------------------------
model = RandomForestRegressor(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)

print(f"Test MAE (cfs): {mae:,.2f}")


# -------------------------
# 6) Plot predictions
# -------------------------
plt.figure()
plt.plot(dates_test, y_test, linewidth=1, label="Actual next-day flow")
plt.plot(dates_test, pred, linewidth=1, label="Predicted next-day flow")
plt.title(f"Next-Day Streamflow Forecast (USGS {SITE})\nMAE={mae:,.2f} cfs")
plt.xlabel(f"Date ({LOCAL_TZ})")
plt.ylabel("Discharge (cfs)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)

print(f"Saved plot to {OUT_PLOT}")
