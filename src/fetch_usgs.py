from __future__ import annotations

import argparse
import pandas as pd
from dataretrieval import nwis

# USGS parameter codes:
# 00065 = gage height (stage)
# 00060 = discharge
PARAM_STAGE = "00065"
PARAM_FLOW = "00060"

def fetch_iv(site: str, start: str, end: str) -> pd.DataFrame:
    df, _meta = nwis.get_record(
        sites=site,
        service="iv",
        start=start,
        end=end,
        parameterCd=f"{PARAM_STAGE},{PARAM_FLOW}",
    )

    if df.empty:
        raise RuntimeError("No data returned. Try different dates or a different site.")

    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True)

    # Keep any columns that start with the parameter codes (USGS sometimes appends qualifiers)
    stage_cols = [c for c in df.columns if str(c).startswith(PARAM_STAGE)]
    flow_cols = [c for c in df.columns if str(c).startswith(PARAM_FLOW)]

    out = pd.DataFrame(index=df.index)

    if stage_cols:
        out["stage"] = pd.to_numeric(df[stage_cols[0]], errors="coerce")
    else:
        raise RuntimeError("Stage (00065) not found in response columns.")

    if flow_cols:
        out["discharge"] = pd.to_numeric(df[flow_cols[0]], errors="coerce")
    else:
        out["discharge"] = pd.NA  # discharge not always available

    # Hourly resample to simplify modeling
    out = out.resample("1H").mean()

    # Drop rows missing stage
    out = out.dropna(subset=["stage"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="USGS site number (e.g., 02146381)")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/usgs_hourly.parquet")
    args = ap.parse_args()

    df = fetch_iv(args.site, args.start, args.end)

    print("Rows:", len(df))
    print(df.head(5))
    print(df.tail(5))

    df.to_parquet(args.out)
    print(f"Saved to {args.out}")

if __name__ == "__main__":
    main()
