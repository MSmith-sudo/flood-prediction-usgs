import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # VS Code-friendly: saves plots instead of opening a window
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

START = "2024-01-01"
END = "2024-12-31"
PARAM = "00400"   # pH
STATE = "nc"      # search within North Carolina for a pH-reporting IV site
LOCAL_TZ = "America/New_York"


# -----------------------------
# 1) Robust session w/ retries
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
# 2) Auto-discover a USGS site in NC that reports pH (IV)
# ---------------------------------------------------------
site_url = "https://nwis.waterservices.usgs.gov/nwis/site/"
site_params = {
    "format": "rdb",
    "stateCd": STATE,
    "parameterCd": PARAM,
    "siteType": "ST",       # streams
    "hasDataTypeCd": "iv",  # instantaneous values
}

print("Searching for an NC USGS IV site that reports pH (00400)...")
resp = session.get(site_url, params=site_params, timeout=(5, 20))
resp.raise_for_status()
text = resp.text

SITE = None
for line in text.splitlines():
    if (
        not line
        or line.startswith("#")
        or line.startswith("agency_cd")
        or line.startswith("5s")
    ):
        continue
    cols = line.split("\t")
    # RDB columns: agency_cd, site_no, station_nm, ...
    if len(cols) > 1:
        SITE = cols[1]
        break

if not SITE:
    raise RuntimeError("Couldn't find any NC IV sites that report pH (00400).")

print("Using discovered pH site:", SITE)


# ------------------------------------
# 3) Pull pH time series from USGS IV
# ------------------------------------
url = "https://waterservices.usgs.gov/nwis/iv/"
params = {
    "format": "json",
    "sites": SITE,
    "parameterCd": PARAM,
    "startDT": START,
    "endDT": END,
    "siteStatus": "all",
}

print("DEBUG SITE =", SITE)
print("DEBUG PARAM =", PARAM)
print("DEBUG params dict =", params)

print("About to request:", url, params)
r = session.get(url, params=params, timeout=(5, 20))
print("DEBUG final URL:", r.url)
r.raise_for_status()
data = r.json()

ts = data.get("value", {}).get("timeSeries", [])
if not ts:
    raise ValueError("No timeSeries returned even after site discovery. Try a different date range.")

values = ts[0]["values"][0]["value"]
df = pd.DataFrame(values)

# --- FIX: Parse datetimes as UTC, then convert to local time ---
df["dateTime"] = pd.to_datetime(df["dateTime"], utc=True)
df["dateTime_local"] = df["dateTime"].dt.tz_convert(LOCAL_TZ)

df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.dropna(subset=["value"]).sort_values("dateTime")

if df.empty:
    raise ValueError("Returned timeSeries had no numeric pH values after cleaning.")


# -----------------------------
# 4) Anomaly detection (ML)
# -----------------------------
df["hour"] = df["dateTime_local"].dt.hour
X = df[["value", "hour"]].to_numpy()

iso = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
)
pred = iso.fit_predict(X)
df["anomaly"] = (pred == -1)


# -----------------------------
# 5) Plot + save output
# -----------------------------
plt.figure()
plt.plot(df["dateTime_local"], df["value"], linewidth=1)
anom = df[df["anomaly"]]
plt.scatter(anom["dateTime_local"], anom["value"])
plt.title(f"pH Anomaly Detection (USGS site {SITE})")
plt.xlabel(f"Time ({LOCAL_TZ})")
plt.ylabel("pH")
plt.tight_layout()

out_file = "ph_anomalies.png"
plt.savefig(out_file, dpi=200)
print(f"Saved plot to {out_file}")

print(df["anomaly"].value_counts())
