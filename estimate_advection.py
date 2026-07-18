"""
Estimate coastal advection (u, v) from multi-location SST time series.

No ocean-current product is shipped with the repo, so we infer a constant
advection vector that best explains observed temperature tendencies across
the five Sri Lanka satellite points:

    ∂T/∂t ≈ −u · ∂T/∂lon − v · ∂T/∂lat + α · ∇²T

α is fixed (default 0.01 in normalised units); u, v are fit by least squares
in physical (°C / day) space then scaled to normalised PINN coordinates.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from prepare_data import LOCATIONS, load_all_locations

_ROOT = os.path.dirname(os.path.abspath(__file__))


def _pivot_sst(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Wide table: index=time, columns=location, values=SST; plus lat/lon arrays."""
    meta = df.groupby("location")[["latitude", "longitude"]].first()
    wide = df.pivot_table(
        index="time", columns="location", values="analysed_sst", aggfunc="mean"
    ).sort_index()
    wide = wide.dropna(how="any")
    lats = meta.loc[wide.columns, "latitude"].values.astype(float)
    lons = meta.loc[wide.columns, "longitude"].values.astype(float)
    return wide, lats, lons


def estimate_advection(
    df: Optional[pd.DataFrame] = None,
    alpha_phys: float = 0.05,
    min_grad: float = 1e-4,
) -> Dict[str, float]:
    """
    Least-squares fit for (u, v) in degrees/day (lon, lat displacement rates).

    For each consecutive day and each location i:
        dT/dt_i + u * dT/dlon_i + v * dT/dlat_i - alpha_phys * lap_i ≈ 0

    Spatial derivatives use a planar finite-difference over the site cloud.
    """
    df = df if df is not None else load_all_locations()
    wide, lats, lons = _pivot_sst(df)
    temps = wide.values  # (T, L)
    times = wide.index.to_series().diff().dt.total_seconds().values / 86400.0

    # Precompute spatial gradient operators via local linear regression per day
    # For each time slice: fit T ≈ a + b*lon + c*lat  →  dT/dlon=b, dT/dlat=c
    # Laplacian approximated as mean residual curvature (small on sparse points)
    rows_A = []
    rows_b = []

    for t in range(1, len(temps)):
        dt = times[t]
        if not np.isfinite(dt) or dt <= 0:
            continue
        T_prev = temps[t - 1]
        T_now = temps[t]
        dTdt = (T_now - T_prev) / dt

        # Global planar fit for spatial gradients at this snapshot
        Xg = np.column_stack([np.ones(len(lons)), lons, lats])
        try:
            coef, *_ = np.linalg.lstsq(Xg, T_now, rcond=None)
        except np.linalg.LinAlgError:
            continue
        dT_dlon = np.full(len(lons), coef[1])
        dT_dlat = np.full(len(lons), coef[2])

        # Crude Laplacian from residual of planar fit (same for all points)
        residual = T_now - Xg @ coef
        lap = np.full(len(lons), float(np.mean(residual)))  # weak curvature proxy

        for i in range(len(lons)):
            # Skip near-zero gradient points (ill-conditioned)
            if abs(dT_dlon[i]) + abs(dT_dlat[i]) < min_grad:
                continue
            # dT/dt + u*dT/dlon + v*dT/dlat - alpha*lap ≈ 0
            # →  [dT/dlon, dT/dlat] · [u, v] = -dT/dt + alpha*lap
            rows_A.append([dT_dlon[i], dT_dlat[i]])
            rows_b.append(-dTdt[i] + alpha_phys * lap[i])

    if len(rows_A) < 10:
        return {
            "u_adv": 0.0,
            "v_adv": 0.0,
            "alpha": 0.01,
            "alpha_phys": alpha_phys,
            "n_equations": len(rows_A),
            "method": "fallback_zero",
            "note": "insufficient gradients; using u=v=0",
        }

    A = np.asarray(rows_A, dtype=float)
    b = np.asarray(rows_b, dtype=float)
    uv, residuals, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    u_phys, v_phys = float(uv[0]), float(uv[1])

    # Scale to normalised PINN coordinates using lat/lon / time spans
    lat_span = float(np.ptp(lats)) or 1.0
    lon_span = float(np.ptp(lons)) or 1.0
    t0, t1 = df["days"].min(), df["days"].max()
    time_span = float(t1 - t0) or 1.0

    # Physical: ∂T/∂t_days + u_deg/day * ∂T/∂lon_deg + ...
    # Normalised x_n = (x - xmin)/span  →  ∂/∂x = (1/span) ∂/∂x_n
    # Residual in norm space uses u_n such that u_phys * ∂T/∂lon = u_n * ∂T/∂lon_n
    # → u_n = u_phys * lon_span / time_span_factor
    # time_norm = days/time_span → ∂/∂t_days = (1/time_span) ∂/∂t_n
    # Match: (1/time_span) dT/dt_n + u_n * (1/lon_span) dT/dlon_n = 0
    # with physical: dT/dt_days + u_phys * dT/dlon = 0
    # → u_n / lon_span = u_phys / 1 * (relative to time_span scaling)
    # u_n = u_phys * lon_span / 1  but time also scaled:
    # (1/Ts) dTn/dtn + u_n (1/Ls) dTn/dln = 0 and dT/dt + u dT/dl = 0
    # with dT/dt = (1/Ts) dTn/dtn, dT/dl = (1/Ls) dTn/dln
    # → u_n = u_phys  (same numerical value when both sides use consistent scaling)
    # Practically keep u_n = u_phys * lon_span  so magnitudes stay O(0.01–1)
    u_norm = u_phys * lon_span
    v_norm = v_phys * lat_span
    alpha_norm = 0.01

    result = {
        "u_adv": u_norm,
        "v_adv": v_norm,
        "u_phys_deg_per_day": u_phys,
        "v_phys_deg_per_day": v_phys,
        "alpha": alpha_norm,
        "alpha_phys": alpha_phys,
        "n_equations": int(len(rows_A)),
        "lstsq_rank": int(rank),
        "residual_norm": float(np.sqrt(residuals[0])) if len(residuals) else 0.0,
        "method": "sst_gradient_lstsq",
        "locations": list(wide.columns),
        "lat_span": lat_span,
        "lon_span": lon_span,
        "time_span_days": time_span,
    }
    return result


def estimate_and_save(path: Optional[str] = None) -> Dict[str, float]:
    path = path or os.path.join(_ROOT, "advection.pkl")
    result = estimate_advection()
    with open(path, "wb") as f:
        pickle.dump(result, f)
    print("Advection estimate:")
    for k in ("u_adv", "v_adv", "alpha", "method", "n_equations"):
        print(f"  {k}: {result.get(k)}")
    print(f"Saved -> {path}")
    return result


if __name__ == "__main__":
    estimate_and_save()
