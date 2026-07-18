"""
Prepare real multi-location SST/DHW training data (no synthetic sensor noise).

Uses the five Sri Lanka reef sites in sliot_dataset/ as actual spatial points.
Also builds time- and location-based hold-out splits.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

LOCATIONS = ["hikkaduwa", "kalpitiya", "passikudha", "south_east", "trinco"]

_ROOT = os.path.dirname(os.path.abspath(__file__))
_SLIOT = os.path.join(_ROOT, "sliot_dataset")


def load_location_frame(location: str) -> pd.DataFrame:
    """Load merged SST + DHW for one real satellite location (no noise)."""
    sst_path = os.path.join(_SLIOT, location, "sst_full.csv")
    dhw_path = os.path.join(_SLIOT, location, "dhw_full.csv")
    if not os.path.exists(sst_path):
        raise FileNotFoundError(sst_path)

    sst = pd.read_csv(sst_path)
    sst["time"] = pd.to_datetime(sst["time"], utc=True)

    if os.path.exists(dhw_path):
        dhw = pd.read_csv(dhw_path)
        dhw["time"] = pd.to_datetime(dhw["time"], utc=True)
        df = pd.merge(
            sst,
            dhw[["time", "degree_heating_week"]],
            on="time",
            how="left",
        )
    else:
        df = sst.copy()
        df["degree_heating_week"] = 0.0

    df["degree_heating_week"] = df["degree_heating_week"].fillna(0.0).clip(lower=0)
    df["location"] = location
    # Keep real satellite coordinates — do not invent triangle offsets
    return df.sort_values("time").reset_index(drop=True)


def load_all_locations(locations: Optional[List[str]] = None) -> pd.DataFrame:
    locations = locations or LOCATIONS
    frames = [load_location_frame(loc) for loc in locations]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["time", "location"]).reset_index(drop=True)
    t0 = df["time"].min()
    df["days"] = (df["time"] - t0).dt.total_seconds() / 86400.0
    return df


def fit_scalers(df: pd.DataFrame) -> Dict[str, MinMaxScaler]:
    scalers: Dict[str, MinMaxScaler] = {}
    for col, key in [
        ("latitude", "scaler_lat"),
        ("longitude", "scaler_lon"),
        ("days", "scaler_time"),
        ("analysed_sst", "scaler_temp"),
        ("degree_heating_week", "scaler_dhw"),
    ]:
        scaler = MinMaxScaler()
        scaler.fit(df[[col]])
        scalers[key] = scaler
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: Dict[str, MinMaxScaler]) -> pd.DataFrame:
    out = df.copy()
    out["lat_norm"] = scalers["scaler_lat"].transform(out[["latitude"]])
    out["lon_norm"] = scalers["scaler_lon"].transform(out[["longitude"]])
    out["time_norm"] = scalers["scaler_time"].transform(out[["days"]])
    out["temp_norm"] = scalers["scaler_temp"].transform(out[["analysed_sst"]])
    out["dhw_norm"] = scalers["scaler_dhw"].transform(out[["degree_heating_week"]])
    return out


def time_holdout_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by unique calendar dates (not shuffled rows).

    Oldest → train, middle → val, newest → test.
    """
    dates = np.array(sorted(df["time"].dt.normalize().unique()))
    n = len(dates)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_dates = set(dates[-n_test:])
    val_dates = set(dates[-(n_test + n_val) : -n_test])

    day = df["time"].dt.normalize()
    test = df[day.isin(test_dates)].copy()
    val = df[day.isin(val_dates)].copy()
    train = df[~day.isin(test_dates | val_dates)].copy()
    return train, val, test


def location_holdout_split(
    df: pd.DataFrame,
    holdout_location: str = "trinco",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-location-out: train on others, evaluate on holdout site."""
    holdout = df[df["location"] == holdout_location].copy()
    train = df[df["location"] != holdout_location].copy()
    return train, holdout


def arrays_from_df(df: pd.DataFrame, include_dhw: bool = False):
    X = df[["lat_norm", "lon_norm", "time_norm"]].values.astype(np.float32)
    if include_dhw:
        y = df[["temp_norm", "dhw_norm"]].values.astype(np.float32)
    else:
        y = df[["temp_norm"]].values.astype(np.float32)
    return X, y


def prepare_and_save(
    out_dir: Optional[str] = None,
    holdout_location: str = "trinco",
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Dict:
    """
    Build real-spatial dataset + hold-outs and write artefacts expected by training.
    """
    out_dir = out_dir or _ROOT
    os.makedirs(os.path.join(out_dir, "dataset"), exist_ok=True)

    df = load_all_locations()
    scalers = fit_scalers(df)
    df = apply_scalers(df, scalers)

    train_t, val_t, test_t = time_holdout_split(df, val_frac=val_frac, test_frac=test_frac)
    train_loc, holdout_loc = location_holdout_split(df, holdout_location=holdout_location)

    # Primary training arrays: time-based train split (real locations, no noise)
    # y_* = temperature only (interpolating PINN + physics)
    # y_*_sst_dhw = [temp, dhw] for multi-output / risk-aware training
    X_train, y_train = arrays_from_df(train_t, include_dhw=False)
    X_val, y_val = arrays_from_df(val_t, include_dhw=False)
    X_test, y_test = arrays_from_df(test_t, include_dhw=False)
    X_loc_holdout, y_loc_holdout = arrays_from_df(holdout_loc, include_dhw=False)

    _, y_train_multi = arrays_from_df(train_t, include_dhw=True)
    _, y_val_multi = arrays_from_df(val_t, include_dhw=True)
    _, y_test_multi = arrays_from_df(test_t, include_dhw=True)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"), X_val)
    np.save(os.path.join(out_dir, "y_val.npy"), y_val)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    np.save(os.path.join(out_dir, "X_loc_holdout.npy"), X_loc_holdout)
    np.save(os.path.join(out_dir, "y_loc_holdout.npy"), y_loc_holdout)
    np.save(os.path.join(out_dir, "y_train_sst_dhw.npy"), y_train_multi)
    np.save(os.path.join(out_dir, "y_val_sst_dhw.npy"), y_val_multi)
    np.save(os.path.join(out_dir, "y_test_sst_dhw.npy"), y_test_multi)
    csv_path = os.path.join(out_dir, "dataset", "real_locations_data.csv")
    df.to_csv(csv_path, index=False)

    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    sensor_coords = (
        df.groupby("location")[["latitude", "longitude"]].first().reset_index()
    )
    sensor_info = {
        "locations": LOCATIONS,
        "sensor_coords": [
            (float(r.latitude), float(r.longitude)) for _, r in sensor_coords.iterrows()
        ],
        "location_names": sensor_coords["location"].tolist(),
        "time_range": (str(df["time"].min()), str(df["time"].max())),
        "holdout_location": holdout_location,
        "n_train": int(len(train_t)),
        "n_val": int(len(val_t)),
        "n_test": int(len(test_t)),
        "n_loc_holdout": int(len(holdout_loc)),
        "data_source": "sliot_dataset real satellite points (no synthetic noise)",
    }
    with open(os.path.join(out_dir, "sensor_info.pkl"), "wb") as f:
        pickle.dump(sensor_info, f)

    # Also dump split frames for forecast validation
    train_t.to_csv(os.path.join(out_dir, "dataset", "split_train.csv"), index=False)
    val_t.to_csv(os.path.join(out_dir, "dataset", "split_val.csv"), index=False)
    test_t.to_csv(os.path.join(out_dir, "dataset", "split_test.csv"), index=False)

    summary = {
        "n_rows": len(df),
        "n_locations": df["location"].nunique(),
        "n_train": len(train_t),
        "n_val": len(val_t),
        "n_test": len(test_t),
        "holdout_location": holdout_location,
        "n_loc_holdout": len(holdout_loc),
        "time_range": sensor_info["time_range"],
    }
    print("Prepared real spatial dataset:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return summary


if __name__ == "__main__":
    prepare_and_save()
