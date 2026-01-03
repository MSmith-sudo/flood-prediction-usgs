import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# -------------------------
# Settings
# -------------------------
STATE = "nc"
LOCAL_TZ = "America/New_York"

# USGS parameter codes
TURB_PARAM = "63680"   # Turbidity (FNU/NTU-style)
RAIN_PARAM = "00045"   # Precipitation, total (often inches)

# Keep this range modest so you don't pull millions of points
START = "2024-01-01"
END   = "2024-12-31"

# Label threshold: "high turbidity risk"
HIGH_TURBIDITY = 10.0   # adjust later if you want (units depend on sensor, often FNU)

# Feature windows (hours)
RAIN_WIN_6H  = 6
RAIN_WIN_24H = 24
TURB_WIN_6H  = 6
TURB_WIN_24H = 24

OUT_PLOT = "turbidity_risk.png"


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
# Helpers
# ---------------------------------------------------------
def parse_rdb_sites(rdb_text: str):
    """Parse NWIS site RDB into a list of (site_no, station_nm)."""
    out = []
    for line in rdb_text.splitlines():
        if (
            not line
            or line.startswith("#")
            or line.startswith("agency_cd")
            or line.startswith("5s")
        ):
            continue
        cols = line.split("\t")
        if len(cols) >= 3:
            out.append((cols[1], cols[2]))
    return out


def fetch_iv_series(site_no: str, parameter_cd: str):
    """Fetch a single IV time series (JSON) for one parameter at one site."""
    iv_url = "https://waterservices.usgs.gov/nwis/iv/"
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": parameter_cd,
        "startDT": START,
        "endDT": END,
        "siteStatus": "all",
    }
    r = session.get(iv_url, params=params, timeout=(5, 30))
    r.raise_for_status()
    j = r.json()
    ts = j.get("value", {}).get("timeSeries", [])
    if not ts:
        return None, r.url

    # find first series that actually has values
    for series in ts:
        vals = series.get("values", [])
        if vals and vals[0].get("value"):
            return series, r.url
    return None, r.url


def series_to_df(series: dict, value_name: str):
    """Convert one USGS series to a dataframe with datetime + numeric value."""
    values = series["values"][0]["value"]
    df = pd.DataFrame(values)

    # Parse time as UTC then convert; keep tz-aware for resampling
    dt = pd.to_datetime(df["dateTime"], utc=True)
    dt_local = dt.dt.tz_convert(LOCAL_TZ)

    out = pd.DataFrame({
        "dt": dt_local,
        value_name: pd.to_numeric(df["value"], errors="coerce"),
    }).dropna()

    out = out.sort_values("dt")
    return out


# ---------------------------------------------------------
# 1) Find candidate sites for turbidity, and for rain, then intersect
# ---------------------------------------------------------
site_url = "https://nwis.waterservices.usgs.gov/nwis/site/"

print("Searching for NC IV sites that report turbidity (63680)...")
turb_r = session.get(site_url, params={
    "format": "rdb",
    "stateCd": STATE,
    "parameterCd": TURB_PARAM,
    "siteType": "ST",
    "hasDataTypeCd": "iv",
}, timeout=(5, 20))
turb_r.raise_for_status()
turb_sites = parse_rdb_sites(turb_r.text)
turb_set = {s for s, _ in turb_sites}
print(f"Found turbidity sites: {len(turb_sites)}")

print("Searching for NC IV sites that report precipitation (00045)...")
rain_r = session.get(site_url, params={
    "format": "rdb",
    "stateCd": STATE,
    "parameterCd": RAIN_PARAM,
    "siteType": "ST",
    "hasDataTypeCd": "iv",
}, timeout=(5, 20))
rain_r.raise_for_status()
rain_sites = parse_rdb_sites(rain_r.text)
rain_set = {s for s, _ in rain_sites}
print(f"Found precip sites: {len(rain_sites)}")

both = turb_set.intersection(rain_set)
if not both:
    raise RuntimeError(
        "Couldn't find any NC stream sites that report BOTH turbidity (63680) and precipitation (00045) as IV.\n"
        "Try expanding to nearby states, or we can switch rainfall source (NOAA) in the next step."
    )

# Keep a name lookup
name_lookup = {}
for s, nm in turb_sites:
    name_lookup[s] = nm
for s, nm in rain_sites:
    name_lookup.setdefault(s, nm)

candidates = [(s, name_lookup.get(s, "")) for s in list(both)]
print(f"Sites with BOTH turbidity + precip: {len(candidates)}. Probing for actual data in {START}..{END}...")


# ---------------------------------------------------------
# 2) Probe candidates until both series return data in the date window
# ---------------------------------------------------------
SITE = None
STATION_NM = None
turb_series = None
rain_series = None
turb_url = None
rain_url = None

MAX_PROBES = 80

for i, (site_no, station_nm) in enumerate(candidates[:MAX_PROBES], start=1):
    try:
        ts_turb, u_turb = fetch_iv_series(site_no, TURB_PARAM)
        if ts_turb is None:
            print(f"Probe {i}: {site_no} no turbidity IV series in range.")
            continue

        ts_rain, u_rain = fetch_iv_series(site_no, RAIN_PARAM)
        if ts_rain is None:
            print(f"Probe {i}: {site_no} has turbidity but no precip IV series in range.")
            continue

        SITE = site_no
        STATION_NM = station_nm
        turb_series = ts_turb
        rain_series = ts_rain
        turb_url = u_turb
        rain_url = u_rain
        print(f"Using working site ({i}/{min(MAX_PROBES, len(candidates))}): {SITE} | {STATION_NM}")
        print("DEBUG turbidity URL:", turb_url)
        print("DEBUG precip URL:", rain_url)
        break

    except Exception as e:
        print(f"Probe {i}: {site_no} failed ({type(e).__name__}).")

if SITE is None:
    raise RuntimeError(
        f"No site found in first {MAX_PROBES} candidates with BOTH series populated for {START}..{END}.\n"
        "Try increasing MAX_PROBES, changing date range, or we can pull rainfall from NOAA instead."
    )


# ---------------------------------------------------------
# 3) Build joined hourly dataset
# ---------------------------------------------------------
df_turb = series_to_df(turb_series, "turbidity")
df_rain = series_to_df(rain_series, "rain")

# Hourly aggregation (mean turbidity, sum rainfall per hour)
turb_h = df_turb.set_index("dt").resample("1H").mean()
rain_h = df_rain.set_index("dt").resample("1H").sum()

df = turb_h.join(rain_h, how="inner").dropna()
df = df.sort_index()

if len(df) < 24 * 30:
    raise ValueError("Not enough overlapping hourly data to model (need at least ~30 days).")

# ---------------------------------------------------------
# 4) Labels + features (rainfall drives turbidity)
# ---------------------------------------------------------
df["high_risk"] = (df["turbidity"] >= HIGH_TURBIDITY).astype(int)

# Rain features
df["rain_6h"]  = df["rain"].rolling(RAIN_WIN_6H).sum()
df["rain_24h"] = df["rain"].rolling(RAIN_WIN_24H).sum()

# Turbidity context features
df["turb_mean_6h"]  = df["turbidity"].rolling(TURB_WIN_6H).mean()
df["turb_std_6h"]   = df["turbidity"].rolling(TURB_WIN_6H).std()
df["turb_mean_24h"] = df["turbidity"].rolling(TURB_WIN_24H).mean()
df["turb_std_24h"]  = df["turbidity"].rolling(TURB_WIN_24H).std()

# Rate of change features
df["turb_delta_1h"] = df["turbidity"].diff(1)
df["rain_delta_1h"] = df["rain"].diff(1)

# Time features
df["hour"] = df.index.hour
df["month"] = df.index.month

# Drop rows with NaNs from rolling/diff
model_df = df.dropna().copy()

feature_cols = [
    "rain", "rain_6h", "rain_24h", "rain_delta_1h",
    "turb_mean_6h", "turb_std_6h", "turb_mean_24h", "turb_std_24h",
    "turb_delta_1h",
    "hour", "month"
]

X = model_df[feature_cols].to_numpy()
y = model_df["high_risk"].to_numpy()
t_index = model_df.index

# Time-aware split: last 20% test
split = int(len(model_df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
t_test = t_index[split:]


# ---------------------------------------------------------
# 5) Train classifier
# ---------------------------------------------------------
clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample"
)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)

print("\n=== Confusion Matrix (test) ===")
print(confusion_matrix(y_test, pred))

print("\n=== Classification Report (test) ===")
print(classification_report(y_test, pred, digits=3))

print("\nClass balance (overall):")
print(model_df["high_risk"].value_counts())


# ---------------------------------------------------------
# 6) Plot: turbidity with predicted risk markers
# ---------------------------------------------------------
# Build a plot frame aligned to test window
plot_df = model_df.loc[t_test].copy()
plot_df["pred_risk"] = pred

plt.figure()
plt.plot(plot_df.index, plot_df["turbidity"], linewidth=1, label="Turbidity")
risk_pts = plot_df[plot_df["pred_risk"] == 1]
plt.scatter(risk_pts.index, risk_pts["turbidity"], s=10, label="Predicted high-risk")
plt.axhline(HIGH_TURBIDITY, linewidth=1, linestyle="--", label=f"Threshold={HIGH_TURBIDITY}")

plt.title(f"Turbidity Risk Detection w/ Rain Features\nUSGS {SITE} {STATION_NM}")
plt.xlabel(f"Time ({LOCAL_TZ})")
plt.ylabel("Turbidity")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=200)

print(f"\nSaved plot to {OUT_PLOT}")
