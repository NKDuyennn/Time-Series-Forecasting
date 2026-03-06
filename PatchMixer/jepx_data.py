"""
jepx_data.py
------------
Fetch JEPX spot_summary data for years 2022-2026, merge with any existing
dataset/jepx/jepx2022-2026.csv, and save a single up-to-date CSV.

Output format (compatible with PatchMixer Dataset_Custom):
  - First column : 'date'  (timezone-naive, ISO format: YYYY-MM-DD HH:MM:SS)
  - Remaining    : feature columns (English names)
  - No pandas index in the CSV file

Usage:
    python jepx_data.py
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from datetime import timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
YEARS          = ["2022", "2023", "2024", "2025", "2026"]
OUTPUT_DIR     = "dataset/jepx"
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, "jepx2022-2026.csv")
API_URL        = "https://www.jepx.jp/js/csv_read.php"
REQUEST_DELAY  = 0.5   # seconds between API calls (be polite to the server)

HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.9,ja;q=0.8",
    "cache-control": "no-cache",
    "if-modified-since": "Thu, 01 Jun 1970 00:00:00 GMT",
    "pragma": "no-cache",
    "referer": "https://www.jepx.jp/en/electricpower/market-data/spot/",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# ---------------------------------------------------------------------------
# Japanese → English column rename mapping
# Keys use regex-free exact match; extend as needed when new columns appear.
# ---------------------------------------------------------------------------
COLUMN_RENAME = {
    # Core identifiers
    "受渡日":                          "delivery_date",
    "時刻コード":                        "time_code",
    # Bid volumes
    "売り入札量(MWh)":                   "sell_bid_volume_mwh",
    "買い入札量(MWh)":                   "buy_bid_volume_mwh",
    # Contract
    "約定総量(MWh)":                     "contract_volume_mwh",
    "約定量(MWh)":                       "contract_volume_mwh",
    # System price
    "システムプライス(円/kWh)":             "system_price",
    # Area prices
    "エリアプライス 北海道(円/kWh)":         "area_price_hokkaido",
    "エリアプライス 東北(円/kWh)":           "area_price_tohoku",
    "エリアプライス 東京(円/kWh)":           "area_price_tokyo",
    "エリアプライス 中部(円/kWh)":           "area_price_chubu",
    "エリアプライス 北陸(円/kWh)":           "area_price_hokuriku",
    "エリアプライス 関西(円/kWh)":           "area_price_kansai",
    "エリアプライス 中国(円/kWh)":           "area_price_chugoku",
    "エリアプライス 四国(円/kWh)":           "area_price_shikoku",
    "エリアプライス 九州(円/kWh)":           "area_price_kyushu",
    # Congestion rent (sometimes present)
    "連系線利用料 北海道-東北(円/kWh)":       "congestion_hokkaido_tohoku",
    "連系線利用料 東北-東京(円/kWh)":         "congestion_tohoku_tokyo",
    "連系線利用料 東京-中部(円/kWh)":         "congestion_tokyo_chubu",
    "連系線利用料 中部-北陸(円/kWh)":         "congestion_chubu_hokuriku",
    "連系線利用料 北陸-関西(円/kWh)":         "congestion_hokuriku_kansai",
    "連系線利用料 関西-中国(円/kWh)":         "congestion_kansai_chugoku",
    "連系線利用料 中国-四国(円/kWh)":         "congestion_chugoku_shikoku",
    "連系線利用料 四国-九州(円/kWh)":         "congestion_shikoku_kyushu",
    "連系線利用料 中部-関西(円/kWh)":         "congestion_chubu_kansai",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Combine 受渡日 (delivery_date) and 時刻コード (time_code, 1-based, 30-min slots)
    into a timezone-naive datetime Series named 'date'.
    """
    base = pd.to_datetime(df["delivery_date"], format="%Y/%m/%d", errors="coerce")
    delta = (df["time_code"].astype(int) - 1) * timedelta(minutes=30)
    return (base + delta).rename("date")


def fetch_year(year: str) -> pd.DataFrame | None:
    """
    Fetch spot_summary DataFrame for *year* directly from the API.
    Returns None on failure.
    """
    params = {"dir": "spot_summary", "file": f"spot_summary_{year}.csv"}
    try:
        resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            df = pd.read_csv(StringIO(resp.text), encoding="utf-8-sig")
            print(f"  [api] {year}: {len(df):,} rows fetched")
            time.sleep(REQUEST_DELAY)
            return df
        else:
            print(f"  [api] {year}: HTTP {resp.status_code} — skipped")
    except Exception as e:
        print(f"  [api] {year}: {e} — skipped")

    return None


def process(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Strip whitespace from column names.
    2. Rename Japanese columns to English.
    3. Build the 'date' datetime column.
    4. Drop raw identifier columns (delivery_date, time_code).
    5. Set 'date' as first column (no pandas index).
    6. Drop duplicate timestamps (keep last).
    """
    # 1. strip
    df.columns = df.columns.str.strip()

    # 2. rename – only map columns that exist
    rename_map = {k: v for k, v in COLUMN_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # 3. build date
    if "delivery_date" not in df.columns or "time_code" not in df.columns:
        raise ValueError(
            "Expected columns '受渡日'/'delivery_date' and '時刻コード'/'time_code' not found. "
            f"Available: {list(df.columns)}"
        )
    df["date"] = _build_datetime(df)

    # 4. drop raw identifiers
    df = df.drop(columns=["delivery_date", "time_code"], errors="ignore")

    # 5. put 'date' first
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    # 6. drop duplicates
    df = df.drop_duplicates(subset=["date"], keep="last")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(" JEPX spot_summary data fetcher")
    print("=" * 60)

    # -- Fetch all years --
    frames = []
    for year in YEARS:
        df_raw = fetch_year(year)
        if df_raw is not None:
            try:
                frames.append(process(df_raw))
            except Exception as e:
                print(f"  [process] {year}: {e} — skipped")

    if not frames:
        print("\nNo data fetched. Exiting.")
        return

    new_data = pd.concat(frames, ignore_index=True)
    new_data["date"] = pd.to_datetime(new_data["date"])
    new_data = new_data.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    # -- Merge with existing saved file (update / upsert) --
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            existing["date"] = pd.to_datetime(existing["date"])
            # Combine: new_data wins on duplicate timestamps
            combined = (
                pd.concat([existing, new_data], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            print(f"\nMerged with existing file: {len(existing):,} → {len(combined):,} rows")
        except Exception as e:
            print(f"\nCould not read existing file ({e}); overwriting.")
            combined = new_data
    else:
        combined = new_data

    # -- Format date column to ISO string (timezone-naive) --
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # -- Save --
    combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n✓ Saved {len(combined):,} rows  →  {OUTPUT_FILE}")
    print(f"  Date range : {combined['date'].iloc[0]}  →  {combined['date'].iloc[-1]}")
    print(f"  Columns    : {list(combined.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
