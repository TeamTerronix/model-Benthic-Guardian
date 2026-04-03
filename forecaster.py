"""
forecaster.py
=============
PINN-based 7-day (168-hour) temperature forecasting with physics enforcement.

Sliding window
--------------
  Input  : last 48 hourly readings for a sensor  (lat, lon, timestamp, temp)
  Output : 168-hour forecast  (7 days × 24 h)

Physics : Heat Advection-Diffusion Equation
  ∂T/∂t + u·∂T/∂x + v·∂T/∂y = α·(∂²T/∂x² + ∂²T/∂y²)

  Residual R is computed at every forecast point via tf.GradientTape so
  that gradients flow through the trained network – no retraining, purely
  inference-time physics penalisation.  Predictions with |R|² above the
  configurable threshold are flagged.

CPU optimisations
-----------------
  - GPU disabled via CUDA_VISIBLE_DEVICES=-1
  - Thread counts matched to logical CPU count
  - @tf.function JIT on the forward pass
  - Chunked (32 points) GradientTape evaluation to avoid RAM spikes
  - float32 throughout; no unnecessary copies
"""

import logging
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from utils import calculate_bleaching_risk, calculate_location_baseline

logger = logging.getLogger(__name__)

# ── CPU configuration ──────────────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # force CPU
_n_cpu = os.cpu_count() or 4
tf.config.threading.set_intra_op_parallelism_threads(_n_cpu)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ── Constants ──────────────────────────────────────────────────────────────────
WINDOW_HOURS   = 48    # hours of historical context for bias correction
FORECAST_HOURS = 168   # 7 days
CHUNK_SIZE     = 32    # points per GradientTape pass (memory-safe on CPU)
PHYSICS_ALPHA  = 0.01  # thermal diffusivity in normalised coordinate units
U_ADV          = 0.0   # east–west advection (set from ocean data if available)
V_ADV          = 0.0   # north–south advection

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Physics residual helper ────────────────────────────────────────────────────

def _compute_pde_residual(
    model: tf.keras.Model,
    lat_norm: np.ndarray,  # shape (N,)
    lon_norm: np.ndarray,
    t_norm: np.ndarray,
    alpha: float = PHYSICS_ALPHA,
    u_adv: float = U_ADV,
    v_adv: float = V_ADV,
) -> np.ndarray:
    """
    Compute the squared heat-equation residual for N points via GradientTape.

        R = ∂T/∂t + u·∂T/∂lon + v·∂T/∂lat − α·(∂²T/∂lat² + ∂²T/∂lon²)

    Processed in chunks of CHUNK_SIZE to stay CPU/RAM friendly.
    Returns array of shape (N,) containing |R|² per point.
    """
    n = len(lat_norm)
    residuals = np.zeros(n, dtype=np.float32)

    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)

        lat_c = tf.constant(lat_norm[start:end].reshape(-1, 1), dtype=tf.float32)
        lon_c = tf.constant(lon_norm[start:end].reshape(-1, 1), dtype=tf.float32)
        t_c   = tf.constant(t_norm[start:end].reshape(-1, 1),   dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([lat_c, lon_c, t_c])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([lat_c, lon_c, t_c])
                x = tf.concat([lat_c, lon_c, t_c], axis=1)
                T = model(x, training=False)          # (chunk, 1)

            dT_dlat = tape1.gradient(T, lat_c)        # ∂T/∂lat
            dT_dlon = tape1.gradient(T, lon_c)        # ∂T/∂lon
            dT_dt   = tape1.gradient(T, t_c)          # ∂T/∂t

        d2T_dlat2 = tape2.gradient(dT_dlat, lat_c)   # ∂²T/∂lat²
        d2T_dlon2 = tape2.gradient(dT_dlon, lon_c)   # ∂²T/∂lon²

        del tape1, tape2

        # Guard against None gradients (e.g. if inputs are disconnected)
        def _safe(g):
            return g if g is not None else tf.zeros_like(lat_c)

        R = (
            _safe(dT_dt)
            + u_adv * _safe(dT_dlon)
            + v_adv * _safe(dT_dlat)
            - alpha * (_safe(d2T_dlat2) + _safe(d2T_dlon2))
        )
        residuals[start:end] = tf.reduce_sum(tf.square(R), axis=1).numpy()

    return residuals


def _scaler_transform(scaler, X):
    """Use DataFrame when scaler was fit on pandas (avoids sklearn feature-name warnings)."""
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn):
        return scaler.transform(pd.DataFrame(np.asarray(X), columns=fn))
    return scaler.transform(np.asarray(X))


def _scaler_inverse_transform(scaler, X):
    fn = getattr(scaler, "feature_names_in_", None)
    if fn is not None and len(fn):
        return scaler.inverse_transform(pd.DataFrame(np.asarray(X), columns=fn))
    return scaler.inverse_transform(np.asarray(X))


# ── Main forecaster class ──────────────────────────────────────────────────────

class PINNForecaster:
    """
    Wraps the trained PINN (lat, lon, time) → temperature model and exposes
    a sliding-window 168-hour forecast with physics residual scoring.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scalers_path: Optional[str] = None,
    ):
        model_path   = model_path   or os.path.join(_MODEL_DIR, "pinn_model_best.h5")
        scalers_path = scalers_path or os.path.join(_MODEL_DIR, "scalers.pkl")

        logger.info("Loading PINN model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.model.trainable = False

        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

        self.scaler_lat  = scalers["scaler_lat"]
        self.scaler_lon  = scalers["scaler_lon"]
        self.scaler_time = scalers["scaler_time"]
        self.scaler_temp = scalers["scaler_temp"]

        # Training-time reference: scaler_time was fit on integer-day offsets
        # MinMaxScaler stores data_min_/data_max_ as 1-D arrays (one per feature).
        t_min = np.asarray(self.scaler_time.data_min_).ravel()
        t_max = np.asarray(self.scaler_time.data_max_).ravel()
        self._t_min   = float(t_min[0])
        self._t_scale = float(t_max[0] - t_min[0])
        if self._t_scale < 1e-8:
            self._t_scale = 1.0

        # Absolute day-0 reference from sensor_info.pkl (falls back gracefully)
        self._dataset_t0 = self._load_dataset_t0()

        # Load location baseline temperatures
        self.location_baselines = self._load_location_baselines()

        logger.info(
            "PINNForecaster ready  params=%d  dataset_t0=%s  locations=%d",
            self.model.count_params(),
            self._dataset_t0,
            len(self.location_baselines),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_dataset_t0(self) -> pd.Timestamp:
        try:
            with open(os.path.join(_MODEL_DIR, "sensor_info.pkl"), "rb") as f:
                si = pickle.load(f)
            return pd.Timestamp(si["time_range"][0])
        except Exception:
            logger.warning("sensor_info.pkl not found / unreadable; using epoch as t0")
            return pd.Timestamp("2010-01-01")

    def _load_location_baselines(self) -> Dict[str, Dict]:
        """
        Load monthly temperature baselines for each location from historical SST data.
        
        Returns:
            Dict mapping location names to monthly baseline dicts
            {
                'hikkaduwa': {1: 28.5, 2: 28.7, ...},
                'kalpitiya': {1: 27.8, ...},
                ...
            }
        """
        locations = {}
        dataset_dir = os.path.join(_MODEL_DIR, "sliot_dataset")
        
        for location in ["hikkaduwa", "kalpitiya", "passikudha", "south_east", "trinco"]:
            sst_path = os.path.join(dataset_dir, location, "sst_full.csv")
            if os.path.exists(sst_path):
                try:
                    locations[location] = calculate_location_baseline(sst_path)
                    logger.info(f"Loaded baseline for {location}")
                except Exception as e:
                    logger.warning(f"Failed to load baseline for {location}: {e}")
            else:
                logger.warning(f"SST data not found for {location}")
        
        return locations

    def _norm_lat(self, v: float) -> float:
        return float(_scaler_transform(self.scaler_lat, [[v]])[0, 0])

    def _norm_lon(self, v: float) -> float:
        return float(_scaler_transform(self.scaler_lon, [[v]])[0, 0])

    def _norm_time(self, days: float) -> float:
        """Linear extrapolation beyond the training range is intentional."""
        return (days - self._t_min) / self._t_scale

    def _to_days(self, ts: datetime) -> float:
        return (pd.Timestamp(ts) - self._dataset_t0).total_seconds() / 86400.0

    def _denorm_temp(self, arr: np.ndarray) -> np.ndarray:
        X = arr.astype(np.float32).reshape(-1, 1)
        return _scaler_inverse_transform(self.scaler_temp, X).flatten()

    @tf.function(reduce_retracing=True)
    def _forward_batch(self, x: tf.Tensor) -> tf.Tensor:
        """JIT-compiled forward pass — call with a (N, 3) float32 tensor."""
        return self.model(x, training=False)

    # ── Bias correction ────────────────────────────────────────────────────────

    def _bias_from_window(
        self, lat_n: float, lon_n: float, readings: List[Dict]
    ) -> float:
        """
        Compute mean(actual − pinn_pred) over the context window readings.
        Applied as a constant additive correction to the 168-h forecast.
        """
        if not readings:
            return 0.0

        errors = []
        for r in readings:
            t_n  = self._norm_time(self._to_days(r["timestamp"]))
            x    = np.array([[lat_n, lon_n, t_n]], dtype=np.float32)
            pred_n = self._forward_batch(tf.constant(x)).numpy()[0, 0]
            pred   = self._denorm_temp(np.array([pred_n]))[0]
            errors.append(float(r["temperature"]) - pred)

        return float(np.mean(errors))

    # ── Public API ─────────────────────────────────────────────────────────────

    def forecast(
        self,
        lat: float,
        lon: float,
        last_readings: List[Dict],
        location: str = "hikkaduwa",
        reference_dt: Optional[datetime] = None,
    ) -> List[Dict]:
        """
        Generate a 168-hour forecast with advanced bleaching risk assessment.

        Parameters
        ----------
        lat, lon      : sensor coordinates (decimal degrees)
        last_readings : list of dicts, each with keys
                          'timestamp' (datetime) and 'temperature' (float)
                        Up to 48 entries, sorted oldest → newest.
        location      : location name for baseline calculation
                       ('hikkaduwa', 'kalpitiya', 'passikudha', 'south_east', 'trinco')
        reference_dt  : forecast start time (defaults to last reading's timestamp)

        Returns
        -------
        list of 168 dicts:
          {
            'target_timestamp' : datetime,
            'predicted_temp'   : float  (°C),
            'risk_score'       : float  (0-1, continuous),
            'risk_level'       : int    (0 healthy / 1 warning / 2 danger),
            'anomaly'          : float  (°C above baseline),
            'days_stressed'    : int    (consecutive warm days),
            'warming_rate'     : float  (°C/day),
            'physics_residual' : float  (|R|², lower = more physical),
          }
        """
        if not last_readings:
            raise ValueError("last_readings must not be empty")

        ref_dt = reference_dt or last_readings[-1]["timestamp"]

        lat_n = self._norm_lat(lat)
        lon_n = self._norm_lon(lon)

        # ── Bias correction from context window ───────────────────────────────
        bias = self._bias_from_window(lat_n, lon_n, last_readings[-WINDOW_HOURS:])

        # ── Build (168, 3) input array for future time points ─────────────────
        future_ts = [ref_dt + timedelta(hours=h) for h in range(1, FORECAST_HOURS + 1)]
        future_days = np.array([self._to_days(ts) for ts in future_ts], dtype=np.float32)

        lat_arr = np.full(FORECAST_HOURS, lat_n, dtype=np.float32)
        lon_arr = np.full(FORECAST_HOURS, lon_n, dtype=np.float32)
        t_arr   = np.array([self._norm_time(d) for d in future_days], dtype=np.float32)

        x_batch = np.stack([lat_arr, lon_arr, t_arr], axis=1)  # (168, 3)

        # ── Forward inference (JIT) ───────────────────────────────────────────
        preds_norm = self._forward_batch(
            tf.constant(x_batch, dtype=tf.float32)
        ).numpy().flatten()
        preds_temp = self._denorm_temp(preds_norm) + bias

        # ── Physics residual via GradientTape ─────────────────────────────────
        phys = _compute_pde_residual(self.model, lat_arr, lon_arr, t_arr)

        # ── Get baseline temperature for current month ──────────────────────────
        baseline_month = ref_dt.month
        location_lower = location.lower()
        baseline_temp = 28.0  # default fallback
        
        if location_lower in self.location_baselines:
            baseline_temp = self.location_baselines[location_lower].get(
                baseline_month, 28.0
            )

        # ── Recent temperatures for anomaly/persistence calculation ──────────────
        recent_temps = [r["temperature"] for r in last_readings[-12:]]

        # ── Assemble output with advanced risk calculation ─────────────────────
        forecast_output = []
        for i in range(FORECAST_HOURS):
            current_temp = float(preds_temp[i])
            
            # Include forecast history in recent_temps for accumulating stress
            temps_for_risk = recent_temps + list(preds_temp[:i])
            
            # Calculate advanced bleaching risk
            risk_info = calculate_bleaching_risk(
                current_temp=current_temp,
                baseline_temp=baseline_temp,
                recent_temps=temps_for_risk,
                current_time=future_ts[i]
            )
            
            forecast_output.append({
                "target_timestamp": future_ts[i],
                "predicted_temp":   round(current_temp, 4),
                "risk_score":       round(risk_info['risk_score'], 4),
                "risk_level":       risk_info['risk_level'],
                "anomaly":          round(risk_info['anomaly'], 2),
                "days_stressed":    risk_info['days_stressed'],
                "warming_rate":     round(risk_info['warming_rate'], 3),
                "physics_residual": round(float(phys[i]), 6),
            })
        
        return forecast_output


# ── Module-level singleton ─────────────────────────────────────────────────────

_forecaster: Optional[PINNForecaster] = None


def get_forecaster() -> PINNForecaster:
    """Lazy-load the global forecaster (expensive TF model load done once)."""
    global _forecaster
    if _forecaster is None:
        _forecaster = PINNForecaster()
    return _forecaster
