"""
Build supervised forecast samples: given context at issue time, predict SST
(and DHW) at +1 / +3 / +7 days.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from prepare_data import (
    apply_scalers,
    fit_scalers,
    load_all_locations,
    time_holdout_split,
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_HORIZONS = [1, 3, 7]


def build_forecast_frame(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    For each location and horizon h, join issue rows at t with targets at t+h.
    Requires scaled columns from apply_scalers.
    """
    horizons = horizons or DEFAULT_HORIZONS
    df = df.sort_values(["location", "time"]).copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    pieces = []
    for loc, g in df.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        issue = g.rename(
            columns={
                "time": "time_issue",
                "days": "days_issue",
                "analysed_sst": "sst_issue",
                "degree_heating_week": "dhw_issue",
                "time_norm": "time_norm_issue",
                "temp_norm": "temp_norm_issue",
                "dhw_norm": "dhw_norm_issue",
            }
        )
        keep_issue = [
            "location",
            "latitude",
            "longitude",
            "lat_norm",
            "lon_norm",
            "time_issue",
            "days_issue",
            "sst_issue",
            "dhw_issue",
            "time_norm_issue",
            "temp_norm_issue",
            "dhw_norm_issue",
        ]
        issue = issue[keep_issue]

        for h in horizons:
            target = g.copy()
            target["time_issue"] = target["time"] - pd.Timedelta(days=h)
            target = target.rename(
                columns={
                    "time": "time_target",
                    "days": "days_target",
                    "analysed_sst": "sst_target",
                    "degree_heating_week": "dhw_target",
                    "time_norm": "time_norm_target",
                    "temp_norm": "temp_norm_target",
                    "dhw_norm": "dhw_norm_target",
                }
            )
            keep_target = [
                "location",
                "time_issue",
                "time_target",
                "days_target",
                "sst_target",
                "dhw_target",
                "time_norm_target",
                "temp_norm_target",
                "dhw_norm_target",
            ]
            merged = pd.merge(
                issue,
                target[keep_target],
                on=["location", "time_issue"],
                how="inner",
            )
            merged["horizon_days"] = h
            pieces.append(merged)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def arrays_for_forecast(
    fdf: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    X: [lat, lon, time_target, horizon_norm, temp_issue, dhw_issue]
    y: [temp_target, dhw_target]
    """
    horizons = horizons or DEFAULT_HORIZONS
    h_max = float(max(horizons))
    horizon_norm = fdf["horizon_days"].values.astype(np.float32) / h_max
    X = np.column_stack(
        [
            fdf["lat_norm"].values,
            fdf["lon_norm"].values,
            fdf["time_norm_target"].values,
            horizon_norm,
            fdf["temp_norm_issue"].values,
            fdf["dhw_norm_issue"].values,
        ]
    ).astype(np.float32)
    y = np.column_stack(
        [
            fdf["temp_norm_target"].values,
            fdf["dhw_norm_target"].values,
        ]
    ).astype(np.float32)
    return X, y


def prepare_forecast_dataset(
    out_dir: Optional[str] = None,
    horizons: Optional[List[int]] = None,
) -> Dict:
    out_dir = out_dir or _ROOT
    horizons = horizons or DEFAULT_HORIZONS

    df = load_all_locations()
    scalers = fit_scalers(df)
    df = apply_scalers(df, scalers)
    train_t, val_t, test_t = time_holdout_split(df)

    os.makedirs(os.path.join(out_dir, "dataset"), exist_ok=True)

    def _pack(split_df, name):
        fdf = build_forecast_frame(split_df, horizons=horizons)
        X, y = arrays_for_forecast(fdf, horizons=horizons)
        np.save(os.path.join(out_dir, f"X_forecast_{name}.npy"), X)
        np.save(os.path.join(out_dir, f"y_forecast_{name}.npy"), y)
        fdf.to_csv(
            os.path.join(out_dir, "dataset", f"forecast_{name}.csv"), index=False
        )
        return len(fdf), X.shape, y.shape

    n_tr, xs_tr, ys_tr = _pack(train_t, "train")
    n_va, xs_va, ys_va = _pack(val_t, "val")
    n_te, xs_te, ys_te = _pack(test_t, "test")

    meta = {
        "horizons": horizons,
        "input_features": [
            "lat_norm",
            "lon_norm",
            "time_norm_target",
            "horizon_norm",
            "temp_norm_issue",
            "dhw_norm_issue",
        ],
        "output_features": ["temp_norm_target", "dhw_norm_target"],
        "n_train": n_tr,
        "n_val": n_va,
        "n_test": n_te,
        "X_train_shape": list(xs_tr),
        "y_train_shape": list(ys_tr),
    }
    with open(os.path.join(out_dir, "forecast_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    with open(os.path.join(out_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)

    print("Forecast dataset prepared:")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    return meta


if __name__ == "__main__":
    prepare_forecast_dataset()
