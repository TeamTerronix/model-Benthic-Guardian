"""
Forecast skill validation: raw PINN vs PINN+bias vs persistence vs climatology.

Bias correction mirrors forecaster.py:
  bias = mean(actual - pinn_pred) over a recent context window at issue time,
  then applied as a constant offset to the horizon prediction.
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Match forecaster context (hours → days for daily SST)
DEFAULT_BIAS_WINDOW_DAYS = 7
DEFAULT_HORIZONS = [1, 3, 7]


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


def load_artifacts(model_path: Optional[str] = None):
    path = model_path or os.path.join(_ROOT, "pinn_model_best.h5")
    forecast_path = os.path.join(_ROOT, "pinn_forecast_best.h5")
    is_forecast = False
    if model_path is None and os.path.exists(forecast_path):
        path = forecast_path
        is_forecast = True
    model = tf.keras.models.load_model(path, compile=False)
    # Detect forecast head by input width
    try:
        in_dim = int(model.input_shape[-1])
        is_forecast = in_dim >= 6
    except Exception:
        pass
    with open(os.path.join(_ROOT, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(_ROOT, "sensor_info.pkl"), "rb") as f:
        sensor_info = pickle.load(f)
    return model, scalers, sensor_info, path, is_forecast


def pinn_predict_batch(model, scalers, lats, lons, days) -> np.ndarray:
    """Vectorised interpolating PINN SST prediction (3-D input)."""
    lat_n = _scaler_transform(scalers["scaler_lat"], np.asarray(lats).reshape(-1, 1))
    lon_n = _scaler_transform(scalers["scaler_lon"], np.asarray(lons).reshape(-1, 1))
    t_min = float(np.asarray(scalers["scaler_time"].data_min_).ravel()[0])
    t_max = float(np.asarray(scalers["scaler_time"].data_max_).ravel()[0])
    t_scale = max(t_max - t_min, 1e-8)
    time_n = ((np.asarray(days, dtype=float) - t_min) / t_scale).reshape(-1, 1)
    x = np.hstack([lat_n, lon_n, time_n]).astype(np.float32)
    pred_n = model.predict(x, verbose=0)
    if pred_n.ndim == 2 and pred_n.shape[1] > 1:
        pred_n = pred_n[:, 0:1]
    return _scaler_inverse(scalers["scaler_temp"], pred_n).flatten()


def forecast_predict_one(
    model,
    scalers,
    lat: float,
    lon: float,
    days_target: float,
    horizon_days: int,
    sst_issue: float,
    dhw_issue: float,
    max_horizon: int = 7,
) -> Tuple[float, float]:
    """
    6-D forecast head → (sst_pred, dhw_pred) in physical units.
    """
    lat_n = float(_scaler_transform(scalers["scaler_lat"], [[lat]])[0, 0])
    lon_n = float(_scaler_transform(scalers["scaler_lon"], [[lon]])[0, 0])
    t_min = float(np.asarray(scalers["scaler_time"].data_min_).ravel()[0])
    t_max = float(np.asarray(scalers["scaler_time"].data_max_).ravel()[0])
    t_scale = max(t_max - t_min, 1e-8)
    time_n = (float(days_target) - t_min) / t_scale
    h_n = float(horizon_days) / float(max_horizon)
    temp_n = float(_scaler_transform(scalers["scaler_temp"], [[sst_issue]])[0, 0])
    dhw_n = float(_scaler_transform(scalers["scaler_dhw"], [[dhw_issue]])[0, 0])
    x = np.array([[lat_n, lon_n, time_n, h_n, temp_n, dhw_n]], dtype=np.float32)
    pred_n = np.asarray(model.predict(x, verbose=0))
    if pred_n.ndim == 1:
        pred_n = pred_n.reshape(1, -1)
    sst = float(_scaler_inverse(scalers["scaler_temp"], pred_n[:, 0:1]).ravel()[0])
    if pred_n.shape[1] > 1:
        dhw = float(_scaler_inverse(scalers["scaler_dhw"], pred_n[:, 1:2]).ravel()[0])
    else:
        dhw = 0.0
    return sst, dhw


def monthly_climatology(train_df: pd.DataFrame) -> Dict[tuple, float]:
    tmp = train_df.copy()
    tmp["month"] = pd.to_datetime(tmp["time"]).dt.month
    g = tmp.groupby(["location", "month"])["analysed_sst"].mean()
    return {(loc, m): float(v) for (loc, m), v in g.items()}


def compute_bias_at_issue(
    model,
    scalers,
    loc_hist: pd.DataFrame,
    issue_time: pd.Timestamp,
    lat: float,
    lon: float,
    window_days: int = DEFAULT_BIAS_WINDOW_DAYS,
) -> float:
    """
    mean(actual - pinn) over observations in (issue_time - window, issue_time].
    """
    t0 = issue_time - pd.Timedelta(days=window_days)
    window = loc_hist[(loc_hist["time"] > t0) & (loc_hist["time"] <= issue_time)]
    if window.empty:
        return 0.0
    preds = pinn_predict_batch(
        model,
        scalers,
        window["latitude"].values,
        window["longitude"].values,
        window["days"].values,
    )
    return float(np.mean(window["analysed_sst"].values - preds))


def load_recalibration(path: Optional[str] = None) -> Dict[str, float]:
    """Optional global / per-location additive offsets from recalibrate.py."""
    path = path or os.path.join(_ROOT, "recalibration.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("offsets", data) if isinstance(data, dict) else {}


def evaluate_horizons(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    scalers,
    horizons_days: Optional[List[int]] = None,
    bias_window_days: int = DEFAULT_BIAS_WINDOW_DAYS,
    recalibration: Optional[Dict[str, float]] = None,
    is_forecast: bool = False,
    interp_model=None,
) -> pd.DataFrame:
    """
    True forecast protocol:
      issue at t-h, predict target at t
      - persistence: SST at issue
      - PINN raw / forecast head
      - PINN+bias: raw + bias from window ending at issue (interp model)
      - PINN+bias+recal: above + rolling recalibration offset
    """
    horizons_days = horizons_days or DEFAULT_HORIZONS
    clim = monthly_climatology(train_df)
    recalibration = recalibration or {}
    bias_model = interp_model if interp_model is not None else model

    hist = (
        pd.concat([train_df, test_df], ignore_index=True)
        .sort_values(["location", "time"])
        .reset_index(drop=True)
    )
    hist["time"] = pd.to_datetime(hist["time"], utc=True)

    rows = []
    for loc, g in test_df.groupby("location"):
        g = g.sort_values("time").reset_index(drop=True)
        g["time"] = pd.to_datetime(g["time"], utc=True)
        loc_hist = hist[hist["location"] == loc].sort_values("time")
        recal_offset = float(recalibration.get(loc, recalibration.get("global", 0.0)))

        for h in horizons_days:
            y_true, y_raw, y_bias, y_recal, y_persist, y_clim = (
                [],
                [],
                [],
                [],
                [],
                [],
            )

            for _, row in g.iterrows():
                t = row["time"]
                issue = t - pd.Timedelta(days=h)
                past = loc_hist[loc_hist["time"] <= issue]
                if past.empty:
                    continue

                issue_row = past.iloc[-1]
                persist = float(issue_row["analysed_sst"])
                dhw_issue = float(issue_row.get("degree_heating_week", 0.0))
                month = int(t.month)
                clim_val = clim.get((loc, month), persist)

                if is_forecast:
                    pinn_raw, _dhw = forecast_predict_one(
                        model,
                        scalers,
                        float(row["latitude"]),
                        float(row["longitude"]),
                        float(row["days"]),
                        int(h),
                        persist,
                        dhw_issue,
                    )
                    # Forecast head already conditions on issue SST; residual bias≈0
                    bias = 0.0
                else:
                    pinn_raw = float(
                        pinn_predict_batch(
                            model,
                            scalers,
                            [row["latitude"]],
                            [row["longitude"]],
                            [row["days"]],
                        )[0]
                    )
                    bias = compute_bias_at_issue(
                        bias_model,
                        scalers,
                        loc_hist,
                        issue,
                        float(row["latitude"]),
                        float(row["longitude"]),
                        window_days=bias_window_days,
                    )

                pinn_biased = pinn_raw + bias
                pinn_recal = pinn_biased + recal_offset

                y_true.append(float(row["analysed_sst"]))
                y_raw.append(pinn_raw)
                y_bias.append(pinn_biased)
                y_recal.append(pinn_recal)
                y_persist.append(persist)
                y_clim.append(clim_val)

            if not y_true:
                continue

            yt = np.asarray(y_true)
            metrics = {}
            for name, yp in (
                ("pinn_raw", np.asarray(y_raw)),
                ("pinn_bias", np.asarray(y_bias)),
                ("pinn_recal", np.asarray(y_recal)),
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
    """
    Aggregate metrics. Primary model column: pinn_bias (production-like).
    Success focus: beat persistence at 3–7d; beat climatology always.
    """
    if detail.empty:
        return detail

    agg_cols = {
        "pinn_raw_mae": "mean",
        "pinn_bias_mae": "mean",
        "pinn_recal_mae": "mean",
        "persistence_mae": "mean",
        "climatology_mae": "mean",
        "pinn_raw_rmse": "mean",
        "pinn_bias_rmse": "mean",
        "pinn_recal_rmse": "mean",
        "persistence_rmse": "mean",
        "climatology_rmse": "mean",
        "n": "sum",
    }
    # Only aggregate columns that exist
    agg_cols = {k: v for k, v in agg_cols.items() if k in detail.columns}
    agg = detail.groupby("horizon_days").agg(agg_cols).reset_index()

    def _skill(model_col, base_col):
        return 1.0 - agg[model_col] / agg[base_col].clip(lower=1e-6)

    if "pinn_bias_mae" in agg.columns:
        agg["skill_bias_vs_persistence"] = _skill("pinn_bias_mae", "persistence_mae")
        agg["skill_bias_vs_climatology"] = _skill("pinn_bias_mae", "climatology_mae")
    if "pinn_raw_mae" in agg.columns:
        agg["skill_raw_vs_persistence"] = _skill("pinn_raw_mae", "persistence_mae")
    if "pinn_recal_mae" in agg.columns:
        agg["skill_recal_vs_persistence"] = _skill("pinn_recal_mae", "persistence_mae")

    # Pass/fail flags for the healthy success bar
    if "skill_bias_vs_persistence" in agg.columns:
        agg["pass_beat_persistence"] = agg.apply(
            lambda r: bool(r["skill_bias_vs_persistence"] > 0)
            if r["horizon_days"] >= 3
            else None,  # 1-day: persistence is the bar we don't require beating
            axis=1,
        )
    if "skill_bias_vs_climatology" in agg.columns:
        agg["pass_beat_climatology"] = agg["skill_bias_vs_climatology"] > 0

    return agg


def print_success_criteria(summary: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA (honest bars)")
    print("=" * 60)
    print("  1-day : do NOT need to beat persistence (ocean is sticky)")
    print("  3-7d  : aim skill_bias_vs_persistence > 0")
    print("  all   : aim skill_bias_vs_climatology > 0")
    if summary.empty:
        return
    for _, row in summary.iterrows():
        h = int(row["horizon_days"])
        bits = [f"{h}-day:"]
        if "pinn_bias_mae" in row:
            bits.append(f"PINN+bias MAE={row['pinn_bias_mae']:.3f}")
        if "persistence_mae" in row:
            bits.append(f"persist={row['persistence_mae']:.3f}")
        if "skill_bias_vs_persistence" in row and pd.notna(
            row["skill_bias_vs_persistence"]
        ):
            bits.append(f"skill_vs_persist={row['skill_bias_vs_persistence']:+.3f}")
        if "pass_beat_persistence" in row and pd.notna(row["pass_beat_persistence"]):
            bits.append("PASS" if row["pass_beat_persistence"] else "FAIL")
        elif h == 1:
            bits.append("(1d persistence bar optional)")
        print("  " + " | ".join(bits))


def run(output_csv: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    train_path = os.path.join(_ROOT, "dataset", "split_train.csv")
    test_path = os.path.join(_ROOT, "dataset", "split_test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            "Missing dataset/split_*.csv — run prepare_data.py first"
        )

    train_df = pd.read_csv(train_path, parse_dates=["time"])
    test_df = pd.read_csv(test_path, parse_dates=["time"])
    model, scalers, _, model_path, is_forecast = load_artifacts()
    recal = load_recalibration()

    # For bias on forecast runs, also load interpolating backbone if available
    interp_model = None
    interp_path = os.path.join(_ROOT, "pinn_model_best.h5")
    if is_forecast and os.path.exists(interp_path):
        interp_model = tf.keras.models.load_model(interp_path, compile=False)

    detail = evaluate_horizons(
        train_df,
        test_df,
        model,
        scalers,
        recalibration=recal,
        is_forecast=is_forecast,
        interp_model=interp_model,
    )
    summary = skill_summary(detail)

    out = output_csv or os.path.join(_ROOT, "forecast_validation.csv")
    detail.to_csv(out, index=False)
    summary_path = os.path.join(_ROOT, "forecast_validation_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("=" * 60)
    print("FORECAST VALIDATION (MAE °C, lower is better)")
    print(f"Model: {model_path}  (forecast_head={is_forecast})")
    print(f"Recalibration offsets: {recal or '(none)'}")
    print("=" * 60)
    if summary.empty:
        print("No rows evaluated — check splits / model.")
    else:
        cols = [
            c
            for c in [
                "horizon_days",
                "pinn_raw_mae",
                "pinn_bias_mae",
                "pinn_recal_mae",
                "persistence_mae",
                "climatology_mae",
                "skill_bias_vs_persistence",
                "skill_bias_vs_climatology",
            ]
            if c in summary.columns
        ]
        print(summary[cols].to_string(index=False))
    print_success_criteria(summary)
    print(f"\nDetail -> {out}")
    print(f"Summary -> {summary_path}")
    return {"detail": detail, "summary": summary}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN forecast skill validation")
    parser.add_argument("--output", default=None, help="Detail CSV path")
    args = parser.parse_args()
    run(output_csv=args.output)
