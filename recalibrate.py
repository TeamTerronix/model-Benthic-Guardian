"""
Rolling recalibration to correct warm/cold drift on recent data.

Fits a simple additive offset per location (and a global fallback) using the
validation split (or last N days of train+val), then applies it at inference /
forecast evaluation:

    T_corrected = T_pinn_biased + offset[location]

This targets the ~+0.4°C warm bias seen on the newest test years without
full retraining.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from validate_forecast import pinn_predict_batch, load_artifacts

_ROOT = os.path.dirname(os.path.abspath(__file__))


def fit_offsets(
    df: pd.DataFrame,
    model,
    scalers,
    min_samples: int = 30,
) -> Dict[str, float]:
    """
    Per-location mean(actual - pinn_raw). Also stores 'global' mean residual.
    """
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    preds = pinn_predict_batch(
        model,
        scalers,
        df["latitude"].values,
        df["longitude"].values,
        df["days"].values,
    )
    resid = df["analysed_sst"].values - preds
    df = df.assign(_resid=resid)

    offsets: Dict[str, float] = {"global": float(np.mean(resid))}
    for loc, g in df.groupby("location"):
        if len(g) < min_samples:
            offsets[loc] = offsets["global"]
        else:
            offsets[loc] = float(g["_resid"].mean())
    return offsets


def recalibrate_and_save(
    source: str = "val",
    output_path: Optional[str] = None,
) -> Dict:
    """
    source:
      'val'  — dataset/split_val.csv (recommended; no test leakage)
      'train_tail' — last 90 days of train+val
    """
    model, scalers, sensor_info, model_path, is_forecast = load_artifacts(
        model_path=os.path.join(_ROOT, "pinn_model_best.h5")
    )
    if is_forecast:
        # Force interpolating backbone for residual fitting
        model = tf.keras.models.load_model(
            os.path.join(_ROOT, "pinn_model_best.h5"), compile=False
        )

    if source == "val":
        path = os.path.join(_ROOT, "dataset", "split_val.csv")
        df = pd.read_csv(path, parse_dates=["time"])
    elif source == "train_tail":
        train = pd.read_csv(
            os.path.join(_ROOT, "dataset", "split_train.csv"), parse_dates=["time"]
        )
        val = pd.read_csv(
            os.path.join(_ROOT, "dataset", "split_val.csv"), parse_dates=["time"]
        )
        both = pd.concat([train, val], ignore_index=True)
        both["time"] = pd.to_datetime(both["time"], utc=True)
        cutoff = both["time"].max() - pd.Timedelta(days=90)
        df = both[both["time"] >= cutoff].copy()
    else:
        raise ValueError(f"Unknown source: {source}")

    offsets = fit_offsets(df, model, scalers)
    result = {
        "offsets": offsets,
        "source": source,
        "n_rows": int(len(df)),
        "model_path": model_path,
        "note": "Add offset to PINN(+bias) predictions: T + offsets[location]",
    }

    out = output_path or os.path.join(_ROOT, "recalibration.pkl")
    with open(out, "wb") as f:
        pickle.dump(result, f)

    print("=" * 60)
    print("RECALIBRATION OFFSETS (°C)  [actual - pinn]")
    print("=" * 60)
    print(f"source : {source}  (n={len(df)})")
    print(f"model  : {model_path}")
    for k, v in sorted(offsets.items(), key=lambda kv: (kv[0] != "global", kv[0])):
        print(f"  {k:12s}: {v:+.4f} °C")
    print(f"Saved -> {out}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["val", "train_tail"],
        default="val",
        help="Data used to fit offsets (never use test)",
    )
    args = parser.parse_args()
    recalibrate_and_save(source=args.source)
