"""
00_merge_datasets.py
====================
Step 0 — Run this BEFORE 01_data_preparation.ipynb.

What it does:
  1. For every place folder in sliot_dataset/, concatenate the 4 SST part-files
     into one full SST CSV for that place (saves sliot_dataset/<place>/sst_full.csv).
  2. Same for DHW (saves sliot_dataset/<place>/dhw_full.csv).
  3. Concatenate all places' full SSTs → dataset/sst.csv  (overwrites the old one).
  4. Concatenate all places' full DHWs → dataset/dhw.csv  (overwrites the old one).

After running this script, open 01_data_preparation.ipynb and run it — it will
automatically pick up the new multi-place data (the notebook only needs the two
small fixes shown in the comments at the bottom of this file).
"""

import os
import glob
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
SLIOT_DIR  = "sliot_dataset"   # folder containing per-place subdirs
DATASET_DIR = "dataset"        # where the notebook reads sst.csv / dhw.csv

os.makedirs(DATASET_DIR, exist_ok=True)

# ── Helper: load and concat the 4 part-files for one place ─────────────────
def concat_parts(place_dir: str, kind: str) -> pd.DataFrame:
    """
    kind = 'sst' or 'dhw'
    Reads <place_dir>/<kind>/<kind>1.csv … <kind>4.csv,
    drops the units row (row 1), sorts by time, deduplicates.
    """
    pattern = os.path.join(place_dir, kind, f"{kind}*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"  WARNING: no {kind} files in {place_dir}/{kind}/")
        return pd.DataFrame()

    frames = []
    for f in files:
        df = pd.read_csv(f, skiprows=[1])   # row 0 = headers, row 1 = units
        df.columns = [c.strip() for c in df.columns]
        frames.append(df)
        print(f"    Loaded {os.path.relpath(f)} → {len(df)} rows")

    merged = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time", "latitude", "longitude"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    return merged

# ── Main loop over all place directories ────────────────────────────────────
place_dirs = sorted([
    d for d in os.listdir(SLIOT_DIR)
    if os.path.isdir(os.path.join(SLIOT_DIR, d))
])

print(f"Found {len(place_dirs)} place(s): {place_dirs}\n")

all_sst_frames = []
all_dhw_frames = []

for place in place_dirs:
    place_dir = os.path.join(SLIOT_DIR, place)
    print(f"Processing place: {place}")

    # ── SST ──────────────────────────────────────────────────────────────
    sst_full = concat_parts(place_dir, "sst")
    if sst_full.empty:
        print(f"  SKIP {place} (no SST data)\n")
        continue

    sst_out = os.path.join(place_dir, "sst_full.csv")
    sst_full.to_csv(sst_out, index=False)
    print(f"  Saved {sst_out}  ({len(sst_full)} rows, "
          f"lat={sst_full['latitude'].iloc[0]}, lon={sst_full['longitude'].iloc[0]})")

    # ── DHW ──────────────────────────────────────────────────────────────
    dhw_full = concat_parts(place_dir, "dhw")
    if dhw_full.empty:
        print(f"  SKIP {place} DHW (no data)\n")
        continue

    dhw_out = os.path.join(place_dir, "dhw_full.csv")
    dhw_full.to_csv(dhw_out, index=False)
    print(f"  Saved {dhw_out}  ({len(dhw_full)} rows)\n")

    all_sst_frames.append(sst_full)
    all_dhw_frames.append(dhw_full)

# ── Combine all places into one CSV each ────────────────────────────────────
if not all_sst_frames:
    raise RuntimeError("No SST data loaded from any place folder!")

combined_sst = (
    pd.concat(all_sst_frames, ignore_index=True)
    .drop_duplicates(subset=["time", "latitude", "longitude"])
    .sort_values(["latitude", "longitude", "time"])
    .reset_index(drop=True)
)

combined_dhw = (
    pd.concat(all_dhw_frames, ignore_index=True)
    .drop_duplicates(subset=["time", "latitude", "longitude"])
    .sort_values(["latitude", "longitude", "time"])
    .reset_index(drop=True)
)

sst_path = os.path.join(DATASET_DIR, "sst.csv")
dhw_path = os.path.join(DATASET_DIR, "dhw.csv")

combined_sst.to_csv(sst_path, index=False)
combined_dhw.to_csv(dhw_path, index=False)

print("=" * 60)
print(f"Combined SST saved → {sst_path}  ({len(combined_sst)} rows)")
print(f"Combined DHW saved → {dhw_path}  ({len(combined_dhw)} rows)")

# ── Print per-place summary ──────────────────────────────────────────────────
print("\nSST per-place location summary:")
for place, df in zip(place_dirs, all_sst_frames):
    lat = df["latitude"].iloc[0]
    lon = df["longitude"].iloc[0]
    # Rename SST col name if needed
    sst_col = [c for c in df.columns if c not in ("time","latitude","longitude")][0]
    print(f"  {place:15s}  lat={lat:7.3f}  lon={lon:7.3f}  "
          f"temp={df[sst_col].mean():.2f}°C avg  rows={len(df)}")

# ── Unique lat/lon pairs found ───────────────────────────────────────────────
unique_locs = (
    combined_sst[["latitude","longitude"]]
    .drop_duplicates()
    .sort_values(["latitude","longitude"])
    .reset_index(drop=True)
)
print(f"\n{len(unique_locs)} unique sensor location(s) in combined dataset:")
print(unique_locs.to_string())

print("\n✅ Done!  You can now run 01_data_preparation.ipynb")
print("   (apply the two small notebook fixes described below)\n")

# ── Notebook fix guide ───────────────────────────────────────────────────────
print("""
NOTEBOOK FIXES NEEDED IN 01_data_preparation.ipynb
====================================================
Only TWO cells need to change — the rest stays identical.

FIX 1 — Cell "Load SST & DHW" (currently reads dataset/sst.csv)
----------------------------------------------------------------
No change needed — the cell already reads dataset/sst.csv which
now contains all places.

FIX 2 — Cell "Define the 3 Triangle Sensor Points"
---------------------------------------------------
Replace the hardcoded triangle with real coords read from the data:

    # OLD (hardcoded triangle):
    sensor_coords = [
        (8.880, 79.525),  # Sensor 1
        (8.870, 79.530),  # Sensor 2
        (8.870, 79.520)   # Sensor 3
    ]

    # NEW (real place coordinates from data):
    sensor_coords = list(
        df_sst.groupby(["latitude","longitude"])
        .size()
        .reset_index()[["latitude","longitude"]]
        .itertuples(index=False, name=None)
    )
    print(f"Loaded {len(sensor_coords)} sensor location(s):")
    for i, (lat, lon) in enumerate(sensor_coords, 1):
        print(f"  Sensor {i}: lat={lat}, lon={lon}")

FIX 3 — Cell "Simulate data for 3 sensor points"
-------------------------------------------------
The old cell copies data and applies noise. With real multi-place
data this cell is no longer needed — each place already has its own
real lat/lon row in the merged dataframe.

Replace the whole "simulate" cell with:

    # With multi-place data, each lat/lon location IS a real sensor.
    # Assign sensor IDs based on unique (lat, lon) pairs.
    loc_to_id = {
        (row.latitude, row.longitude): i
        for i, row in enumerate(
            df.groupby(["latitude","longitude"]).size()
            .reset_index().itertuples(index=False), start=1
        )
    }
    df_triangle = df.copy()
    df_triangle["sensor_id"] = df_triangle.apply(
        lambda r: loc_to_id[(r.latitude, r.longitude)], axis=1
    )
    print(f"Total combined data shape: {df_triangle.shape}")
    df_triangle.head()

Everything after this (time conversion, normalisation, X_train/y_train
save) stays exactly the same.
""")
