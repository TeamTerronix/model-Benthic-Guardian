"""
Forecast skill validation: PINN vs persistence vs climatology.

Evaluates 1-, 3-, and 7-day ahead temperature MAE/RMSE on the time hold-out
test split (dataset/split_test.csv).
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _scaler_transform(scaler, X):
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn):
        return scaler.transform(pd.DataFrame(np.asarray(X), columns=fn))
    return scaler.transform(np.asarray(X))


def _scaler_inverse(scaler, X):
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn):
        return scaler.inverse_transform(pd.DataFrame(np.asarray(X), columns=fn))
    return scaler.inverse_transform(np.asarray(X))


def load_artifacts():
    model = tf.keras.models.load_model(
        os.path.join(_ROOT, "pinn_model_best.h5"), compile=False
    )
    with open(os.path.join(_ROOT, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(_ROOT, "sensor_info.pkl"), "rb") as f:
        sensor_info = pickle.load(f)
    return model, scalers, sensor_info


def monthly_climatology(train_df: pd.DataFrame) -> Dict[tuple, float]:
    """(location, month) → mean SST."""
    tmp = train_df.copy()
    tmp["month"] = tmp["time"].dt.month
    g = tmp.groupby(["location", "month"])["analysed_sst"].mean()
    return {(loc, m): float(v) for (loc, m), v in g.items()}


def pinn_predict(model, scalers, lats, lons, days) -> np.ndarray:
    lat_n = _scaler_transform(scalers["scaler_lat"], np.asarray(lats).reshape(-1, 1))
    lon_n = _scaler_transform(scalers["scaler_lon"], np.asarray(lons).reshape(-1, 1))
    # Extrapolate time linearly (same as forecaster)
    t_min = float(np.asarray(scalers["scaler_time"].data_min_).ravel()[0])
    t_max = float(np.asarray(scalers["scaler_time"].data_max_).ravel()[0])
    t_scale = max(t_max - t_min, 1e-8)
    time_n = ((np.asarray(days, dtype=float) - t_min) / t_scale).reshape(-1, 1)
    x = np.hstack([lat_n, lon_n, time_n]).astype(np.float32)
    pred_n = model.predict(x, verbose=0)
    return _scaler_inverse(scalers["scaler_temp"], pred_n).flatten()


def evaluate_horizons(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    scalers,
    horizons_days: Optional[List[int]] = None,
) -> pd.DataFrame:
    horizons_days = horizons_days or [1, 3, 7]
    clim = monthly_climatology(train_df)

    # Index history by location for persistence lookups
    hist = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )

    rows = []
    for loc, g in test_df.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        loc_hist = hist[hist["location"] == loc].sort_values("time").reset_index(drop=True)
        for h in horizons_days:
            # Targets: rows where we can look back h days in the same location
            targets = g.iloc[h:].copy() if h < len(g) else g.iloc[0:0].copy()
            if targets.empty:
                # Try calendar shift matching
                targets = g.copy()

            y_true = []
            y_pinn = []
            y_persist = []
            y_clim = []

            for _, row in g.iterrows():
                t = row["time"]
                # Persistence: last available SST at or before t - h days
                t_ref = t - pd.Timedelta(days=h)
                past = loc_hist[loc_hist["time"] <= t_ref]
                if past.empty:
                    continue
                persist = float(past.iloc[-1]["analysed_sst"])
                month = int(t.month)
                clim_val = clim.get((loc, month), persist)

                pinn_val = float(
                    pinn_predict(
                        model,
                        scalers,
                        [row["latitude"]],
                        [row["longitude"]],
                        [row["days"]],
                    )[0]
                )

                y_true.append(float(row["analysed_sst"]))
                y_pinn.append(pinn_val)
                y_persist.append(persist)
                y_clim.append(clim_val)

            if not y_true:
                continue

            yt = np.asarray(y_true)
            metrics = {}
            for name, yp in (
                ("pinn", np.asarray(y_pinn)),
                ("persistence", np.asarray(y_persist)),
                ("climatology", np.asarray(y_clim)),
            ):
                err = yp - yt
                metrics[f"{name}_mae"] = float(np.mean(np.abs(err)))
                metrics[f"{name}_rmse"] = float(np.sqrt(np.mean(err**2)))

            rows.append(
                {
                    "location": loc,
                    "horizon_days": h,
                    "n": len(yt),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def skill_summary(detail: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across locations; positive skill_vs_persistence = PINN better."""
    if detail.empty:
        return detail
    agg = (
        detail.groupby("horizon_days")
        .agg(
            pinn_mae=("pinn_mae", "mean"),
            persistence_mae=("persistence_mae", "mean"),
            climatology_mae=("climatology_mae", "mean"),
            pinn_rmse=("pinn_rmse", "mean"),
            persistence_rmse=("persistence_rmse", "mean"),
            climatology_rmse=("climatology_rmse", "mean"),
            n=("n", "sum"),
        )
        .reset_index()
    )
    agg["skill_vs_persistence"] = 1.0 - agg["pinn_mae"] / agg["persistence_mae"].clip(
        lower=1e-6
    )
    agg["skill_vs_climatology"] = 1.0 - agg["pinn_mae"] / agg["climatology_mae"].clip(
        lower=1e-6
    )
    return agg


def run(output_csv: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    train_path = os.path.join(_ROOT, "dataset", "split_train.csv")
    test_path = os.path.join(_ROOT, "dataset", "split_test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Missing dataset/split_*.csv — run prepare_data.py first"
        )

    train_df = pd.read_csv(train_path, parse_dates=["time"])
    test_df = pd.read_csv(test_path, parse_dates=["time"])
    model, scalers, _ = load_artifacts()

    detail = evaluate_horizons(train_df, test_df, model, scalers)
    summary = skill_summary(detail)

    out = output_csv or os.path.join(_ROOT, "forecast_validation.csv")
    detail.to_csv(out, index=False)
    summary_path = os.path.join(_ROOT, "forecast_validation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=" * 60)
    print("FORECAST VALIDATION (MAE °C, lower is better)")
    print("=" * 60)
    if summary.empty:
        print("No rows evaluated — check splits / model.")
    else:
        print(summary.to_string(index=False))
    print(f"\nDetail -> {out}")
    print(f"Summary -> {summary_path}")
    return {"detail": detail, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN forecast skill validation")
    parser.add_argument("--output", default=None, help="Detail CSV path")
    args = parser.parse_args()
    run(output_csv=args.output)
