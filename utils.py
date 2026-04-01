"""
Utility functions for Coral Reef Digital Twin PINN
Author: Digital Twin System
Date: 2026
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull


def point_in_triangle(p, triangle):
    """
    Check if point p is inside the triangle using barycentric coordinates.
    Kept for backward compatibility with 3-node setups.
    
    Args:
        p: Point coordinates [lat, lon]
        triangle: Triangle vertices [[lat1, lon1], [lat2, lon2], [lat3, lon3]]
    
    Returns:
        bool: True if point is inside triangle
    """
    v0 = triangle[2] - triangle[0]
    v1 = triangle[1] - triangle[0]
    v2 = p - triangle[0]
    
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    
    inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def point_in_convex_hull(point, delaunay_tri):
    """
    Check if a point is inside the convex hull using Delaunay triangulation.
    
    Args:
        point: Point coordinates [lat, lon]
        delaunay_tri: scipy.spatial.Delaunay object built from sensor coordinates
    
    Returns:
        bool: True if point is inside the convex hull
    """
    return delaunay_tri.find_simplex(point) >= 0


def get_convex_hull_boundary(sensor_coords):
    """
    Compute the convex hull boundary of N sensor coordinates.
    
    Args:
        sensor_coords: List of tuples [(lat1, lon1), (lat2, lon2), ..., (latN, lonN)]
    
    Returns:
        hull_coords: List of [lat, lon] forming the convex hull boundary (closed polygon)
        hull: scipy.spatial.ConvexHull object
    """
    points = np.array(sensor_coords)
    hull = ConvexHull(points)
    hull_vertices = hull.vertices
    hull_coords = [points[v].tolist() for v in hull_vertices]
    # Close the polygon
    hull_coords.append(hull_coords[0])
    return hull_coords, hull


def generate_grid_points(sensor_coords, resolution=50):
    """
    Generate a dense grid of points inside the convex hull defined by N sensor coordinates.
    Works with any number of sensors >= 3.
    
    Args:
        sensor_coords: List of tuples [(lat1, lon1), ..., (latN, lonN)]  (N >= 3)
        resolution: Number of points per side of the bounding grid
    
    Returns:
        lats_inside: List of latitudes inside the convex hull
        lons_inside: List of longitudes inside the convex hull
    """
    points = np.array(sensor_coords)
    
    # Build Delaunay triangulation for fast point-in-hull testing
    delaunay_tri = Delaunay(points)
    
    lats = points[:, 0]
    lons = points[:, 1]
    
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    
    # Add small padding to ensure edge sensors are included
    lat_pad = (lat_max - lat_min) * 0.02
    lon_pad = (lon_max - lon_min) * 0.02
    
    lat_grid = np.linspace(lat_min - lat_pad, lat_max + lat_pad, resolution)
    lon_grid = np.linspace(lon_min - lon_pad, lon_max + lon_pad, resolution)
    lat_mesh, lon_mesh = np.meshgrid(lat_grid, lon_grid)
    
    # Vectorised point-in-hull check
    grid_points = np.column_stack([lat_mesh.ravel(), lon_mesh.ravel()])
    inside_mask = delaunay_tri.find_simplex(grid_points) >= 0
    
    points_inside = grid_points[inside_mask]
    lats_inside = points_inside[:, 0].tolist()
    lons_inside = points_inside[:, 1].tolist()
    
    return lats_inside, lons_inside


def calculate_location_baseline(sst_csv_path):
    """
    Calculate monthly baseline temperatures from historical SST data.
    
    Args:
        sst_csv_path: Path to location's sst_full.csv file
    
    Returns:
        dict: Monthly baseline temperatures {1: 28.5, 2: 28.7, ...}
    """
    df = pd.read_csv(sst_csv_path)
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    
    baselines = {}
    for month in range(1, 13):
        month_data = df[df['month'] == month]['analysed_sst']
        if len(month_data) > 0:
            baselines[month] = float(month_data.mean())
    
    return baselines


def calculate_bleaching_risk(
    current_temp,
    baseline_temp,
    recent_temps,
    current_time=None
):
    """
    Multi-factor temperature-based bleaching risk assessment.
    
    Combines:
    1. Temperature anomaly from baseline (50% weight)
    2. Duration of warm stress (30% weight) 
    3. Rate of warming (20% weight)
    
    Args:
        current_temp: Current predicted temperature (°C)
        baseline_temp: Baseline temperature for current month (°C)
        recent_temps: List of recent temperatures [oldest...newest] (last 12+ days)
        current_time: Current datetime (optional, for detailed logging)
    
    Returns:
        dict with:
            'risk_score': 0-1 continuous score
            'risk_level': 0 (healthy), 1 (warning), 2 (danger)
            'anomaly': °C above baseline
            'days_stressed': consecutive warm days
            'warming_rate': °C/day
    """
    # 1. Temperature anomaly (most important signal)
    anomaly = current_temp - baseline_temp
    anomaly_score = np.clip(max(0, anomaly / 2.0), 0, 1.0)
    
    # 2. Duration of stress (consecutive days above baseline + 1°C)
    if len(recent_temps) > 0:
        stress_threshold = baseline_temp + 1.0
        warm_days = np.sum(np.array(recent_temps[-12:]) > stress_threshold)
        consecutive_warm = 0
        for i in range(len(recent_temps) - 1, -1, -1):
            if recent_temps[i] > stress_threshold:
                consecutive_warm += 1
            else:
                break
        days_stressed = consecutive_warm
        duration_score = np.clip(days_stressed / 7.0, 0, 1.0)
    else:
        days_stressed = 0
        duration_score = 0.0
    
    # 3. Rate of warming (rapid warming is stressful)
    if len(recent_temps) >= 4:
        temp_rate = (recent_temps[-1] - recent_temps[-4]) / 3.0  # °C per day
        rate_score = np.clip(max(0, temp_rate) / 0.5, 0, 1.0)
    else:
        temp_rate = 0.0
        rate_score = 0.0
    
    # Weighted combination
    risk_score = (
        0.5 * anomaly_score +
        0.3 * duration_score +
        0.2 * rate_score
    )
    
    # Convert to risk level (0-2)
    if risk_score < 0.33:
        risk_level = 0  # Healthy
    elif risk_score < 0.66:
        risk_level = 1  # Warning
    else:
        risk_level = 2  # Danger
    
    return {
        'risk_score': float(risk_score),
        'risk_level': risk_level,
        'anomaly': float(anomaly),
        'days_stressed': days_stressed,
        'warming_rate': float(temp_rate)
    }


def temperature_to_risk(temp, thresholds={'healthy': 28.0, 'warning': 30.0}):
    """
    DEPRECATED: Use calculate_bleaching_risk instead for better accuracy.
    Simple threshold-based risk (kept for backward compatibility).
    
    Args:
        temp: Temperature in Celsius
        thresholds: Dict with 'healthy' and 'warning' temperature thresholds
    
    Returns:
        risk_level: 0 (healthy), 1 (warning), 2 (danger)
        color: Color name for visualization
        risk_name: Human-readable risk name
    """
    if temp < thresholds['healthy']:
        return 0, 'green', 'Healthy'
    elif temp < thresholds['warning']:
        return 1, 'yellow', 'Warning'
    else:
        return 2, 'red', 'Danger'


def get_risk_score(temp, thresholds={'healthy': 28.0, 'warning': 30.0}):
    """
    DEPRECATED: Use calculate_bleaching_risk instead for better accuracy.
    Simple threshold-based continuous risk score (kept for backward compatibility).
    
    Args:
        temp: Temperature in Celsius
        thresholds: Dict with 'healthy' and 'warning' temperature thresholds
    
    Returns:
        risk_score: Float between 0 (no risk) and 1 (high risk)
    """
    temp_healthy = thresholds['healthy']
    temp_warning = thresholds['warning']
    
    if temp < temp_healthy:
        return max(0, (temp - 26.0) / (temp_healthy - 26.0) * 0.33)
    elif temp < temp_warning:
        return 0.33 + (temp - temp_healthy) / (temp_warning - temp_healthy) * 0.33
    else:
        return min(1.0, 0.66 + (temp - temp_warning) / 2.0 * 0.34)


def simulate_sensor_variations(df, sensor_coords, temp_column='analysed_sst', 
                               dhw_column='degree_heating_week', 
                               temp_variation=0.2, dhw_variation=0.05):
    """
    Create simulated data for multiple sensors from a single dataset
    
    Args:
        df: Original DataFrame with one location
        sensor_coords: List of sensor coordinates
        temp_column: Name of temperature column
        dhw_column: Name of DHW column
        temp_variation: Max random temperature offset (°C)
        dhw_variation: Max random DHW offset
    
    Returns:
        DataFrame with data for all sensors
    """
    all_sensors_data = []
    
    for sensor_id, (lat, lon) in enumerate(sensor_coords, 1):
        sensor_df = df.copy()
        sensor_df['latitude'] = lat
        sensor_df['longitude'] = lon
        sensor_df['sensor_id'] = sensor_id
        
        # Add random variations
        np.random.seed(42 + sensor_id)
        
        if temp_column in sensor_df.columns:
            temp_offset = np.random.uniform(-temp_variation, temp_variation, len(sensor_df))
            sensor_df[temp_column] = sensor_df[temp_column] + temp_offset
        
        if dhw_column in sensor_df.columns:
            dhw_offset = np.random.uniform(-dhw_variation, dhw_variation, len(sensor_df))
            sensor_df[dhw_column] = sensor_df[dhw_column] + dhw_offset
            sensor_df[dhw_column] = sensor_df[dhw_column].clip(lower=0)
        
        all_sensors_data.append(sensor_df)
    
    return pd.concat(all_sensors_data, ignore_index=True)


def normalize_features(df, feature_columns):
    """
    Normalize features to 0-1 range and return scalers
    
    Args:
        df: DataFrame
        feature_columns: List of column names to normalize
    
    Returns:
        df: DataFrame with normalized columns (original columns + '_norm' suffix)
        scalers: Dict of fitted scalers
    """
    from sklearn.preprocessing import MinMaxScaler
    
    scalers = {}
    
    for col in feature_columns:
        scaler = MinMaxScaler()
        df[f'{col}_norm'] = scaler.fit_transform(df[[col]])
        scalers[f'scaler_{col}'] = scaler
    
    return df, scalers


def calculate_statistics(temperatures, predictions, thresholds={'healthy': 28.0, 'warning': 30.0}):
    """
    Calculate comprehensive statistics for model evaluation
    
    Args:
        temperatures: Array of actual temperatures
        predictions: Array of predicted temperatures
        thresholds: Temperature thresholds for risk levels
    
    Returns:
        Dict with various statistics
    """
    errors = predictions - temperatures
    
    stats = {
        'mae': np.mean(np.abs(errors)),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mse': np.mean(errors**2),
        'mean_error': np.mean(errors),
        'std_error': np.std(errors),
        'max_error': np.max(np.abs(errors)),
        'r2_score': 1 - np.sum(errors**2) / np.sum((temperatures - np.mean(temperatures))**2),
        'temp_range': (np.min(temperatures), np.max(temperatures)),
        'pred_range': (np.min(predictions), np.max(predictions)),
        'healthy_count': np.sum(predictions < thresholds['healthy']),
        'warning_count': np.sum((predictions >= thresholds['healthy']) & 
                               (predictions < thresholds['warning'])),
        'danger_count': np.sum(predictions >= thresholds['warning']),
        'total_count': len(predictions)
    }
    
    return stats


def export_to_geojson(lats, lons, temperatures, risk_scores, filename='predictions.geojson'):
    """
    Export predictions to GeoJSON format for use in GIS applications
    
    Args:
        lats: Array of latitudes
        lons: Array of longitudes
        temperatures: Array of temperatures
        risk_scores: Array of risk scores
        filename: Output filename
    """
    import json
    
    features = []
    for lat, lon, temp, risk in zip(lats, lons, temperatures, risk_scores):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "temperature": float(temp),
                "risk_score": float(risk),
                "risk_level": int(temperature_to_risk(temp)[0])
            }
        }
        features.append(feature)
    
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    
    with open(filename, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Exported {len(features)} points to {filename}")


def load_model_and_scalers(model_path='pinn_model_best.h5', scalers_path='scalers.pkl'):
    """
    Load trained model and scalers
    
    Args:
        model_path: Path to saved model
        scalers_path: Path to saved scalers
    
    Returns:
        model: Loaded Keras model
        scalers: Dict of scalers
    """
    import pickle
    from tensorflow import keras
    
    model = keras.models.load_model(model_path, compile=False)
    
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    return model, scalers


def predict_temperature(model, scalers, lat, lon, time_days):
    """
    Predict temperature for a single point
    
    Args:
        model: Trained PINN model
        scalers: Dict of scalers
        lat: Latitude
        lon: Longitude
        time_days: Time in days from start
    
    Returns:
        Predicted temperature in Celsius
    """
    # Normalize inputs
    lat_norm = scalers['scaler_lat'].transform([[lat]])[0, 0]
    lon_norm = scalers['scaler_lon'].transform([[lon]])[0, 0]
    time_norm = scalers['scaler_time'].transform([[time_days]])[0, 0]
    
    # Prepare input
    input_data = np.array([[lat_norm, lon_norm, time_norm]])
    
    # Predict
    prediction_norm = model.predict(input_data, verbose=0)
    prediction = scalers['scaler_temp'].inverse_transform(prediction_norm)[0, 0]
    
    return prediction


if __name__ == "__main__":
    print("Coral Reef Digital Twin - Utility Functions")
    print("=" * 60)
    print("Available functions:")
    print("  - point_in_triangle: Check if a point is inside a triangle (legacy)")
    print("  - point_in_convex_hull: Check if a point is inside N-node convex hull")
    print("  - get_convex_hull_boundary: Get polygon boundary for N sensors")
    print("  - generate_grid_points: Generate grid inside convex hull (N sensors)")
    print("  - temperature_to_risk: Convert temperature to risk level")
    print("  - get_risk_score: Calculate continuous risk score")
    print("  - simulate_sensor_variations: Simulate multi-sensor data")
    print("  - normalize_features: Normalize DataFrame features")
    print("  - calculate_statistics: Calculate model statistics")
    print("  - export_to_geojson: Export predictions to GeoJSON")
    print("  - load_model_and_scalers: Load trained model")
    print("  - predict_temperature: Make single-point predictions")
    print("=" * 60)
