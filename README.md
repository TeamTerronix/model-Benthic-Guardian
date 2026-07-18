# Coral Reef Digital Twin - Physics-Informed Neural Network (PINN)

## 🌊 Project Overview

This project implements a **Digital Twin** for coral reef monitoring using a **Physics-Informed Neural Network (PINN)**. The system predicts ocean temperatures and coral bleaching risk in real-time by interpolating data from multiple underwater sensors.

### Key Features
- ✅ **3-Sensor Triangle Configuration** (scalable to more sensors)
- ✅ **Physics-Informed Neural Network** for temperature prediction
- ✅ **Real-time Bleaching Risk Assessment**
- ✅ **Interactive Heat Maps** with Folium
- ✅ **Time-series Visualization**
- ✅ **Easy-to-use Jupyter Notebooks**

---

## 📁 Project Structure

```
model/
├── dataset/
│   ├── dhw/                    # Degree Heating Week data (4 CSV files)
│   ├── sst/                    # Sea Surface Temperature data (4 CSV files)
│   ├── dhw.csv                 # Merged DHW data
│   ├── sst.csv                 # Merged SST data
│   └── triangle_data.csv       # Processed 3-sensor data
│
├── 01_data_preparation.ipynb   # Step 1: Load and prepare data
├── 02_pinn_model.ipynb         # Step 2: Build and train PINN
├── 03_visualization.ipynb      # Step 3: Create risk maps
├── test.ipynb                  # Your existing test notebook
│
├── utils.py                    # Utility functions
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── scalers.pkl                 # Saved data scalers
├── sensor_info.pkl             # Sensor configuration
├── X_train.npy                 # Training inputs
├── y_train.npy                 # Training outputs
│
├── pinn_model_best.h5          # Best trained model
├── pinn_model_final.h5         # Final trained model
├── pinn_model_saved/           # TensorFlow SavedModel format
│
└── coral_reef_risk_map.html    # Interactive map output
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Rebuild data + physics (notebooks — recommended)

Run in order:

1. **`04_real_spatial_data.ipynb`** — real multi-site SST/DHW + hold-outs  
2. **`05_estimate_advection.ipynb`** — estimate coastal `u, v`  
3. **`08_tune_physics.ipynb`** — sweep PDE weight λ → `physics_best.pkl`  
4. **`02_pinn_model.ipynb`** — train interpolating PINN with tuned physics  
5. **`07_improve_forecasts.ipynb`** — bias eval, recalibration, forecast head (SST+DHW)  
6. **`06_evaluation.ipynb`** — hold-outs + skill vs persistence/climatology  

**Honest bars:** beat persistence at **3–7 days**; beat climatology always; 1-day persistence winning is OK.

### 3. Legacy notebook pipeline

#### **Step 1: Data Preparation** 📊
Open and run: `01_data_preparation.ipynb`

> Prefer `04_real_spatial_data.ipynb` above. The notebook’s triangle + random-noise path is kept for demos only.

This notebook will:
- Load merged SST and DHW CSV files
- Create simulated data for 3 sensor points in a triangle
- Apply random variations to mimic real sensor differences
- Normalize all features (0-1 range)
- Save processed data for training

**Output Files:**
- `dataset/triangle_data.csv`
- `X_train.npy`, `y_train.npy`
- `scalers.pkl`
- `sensor_info.pkl`

---

#### **Step 2: PINN Model Training** 🧠
Open and run: `02_pinn_model.ipynb`

This notebook will:
- Build a Physics-Informed Neural Network
  - Input: Latitude, Longitude, Time (3 features)
  - Hidden Layers: [128, 128, 64, 64, 32]
  - Output: Temperature (1 value)
- Train with custom loss function (Data Loss + Physics Loss)
- Use early stopping and learning rate scheduling
- Evaluate model performance
- Save trained model

**Output Files:**
- `pinn_model_best.h5`
- `pinn_model_final.h5`
- `pinn_model_saved/` (TensorFlow format)
- `training_history.csv`
- `training_history.png`
- `predictions_vs_actual.png`
- `error_analysis.png`

**Expected Performance:**
- MAE: < 0.1°C (normalized)
- RMSE: < 0.15°C (normalized)
- R² Score: > 0.90

---

#### **Step 3: Visualization** 🗺️
Open and run: `03_visualization.ipynb`

This notebook will:
- Load the trained PINN model
- Generate a dense grid of points inside the sensor triangle
- Predict temperature for all grid points
- Calculate bleaching risk scores:
  - 🟢 **Healthy**: < 28°C
  - 🟡 **Warning**: 28-30°C
  - 🔴 **Danger**: > 30°C
- Create interactive Folium map
- Generate static visualizations

**Output Files:**
- `coral_reef_risk_map.html` (Interactive map - open in browser!)
- `heatmap_visualization.png`
- `temperature_evolution.png`
- `prediction_results.csv`

---

## 🎯 How It Works

### The Triangle Configuration

The system uses 3 underwater temperature sensors arranged in a triangle:

```
        Sensor 1 (8.880, 79.525)
              /\
             /  \
            /    \
           /      \
          /________\
Sensor 3           Sensor 2
(8.870, 79.520)   (8.870, 79.530)
```

The PINN interpolates temperatures for **all points inside this triangle**, creating a continuous heat map.

### Physics-Informed Neural Network (PINN)

Unlike standard neural networks, PINNs incorporate physical laws:

1. **Data Loss**: Minimize difference between predictions and sensor readings
2. **Physics Loss**: Enforce smoothness (simplified heat equation)

**Total Loss = Data Loss + λ × Physics Loss**

This ensures predictions are both accurate and physically realistic.

---

## 📊 Coral Bleaching Risk

Coral bleaching occurs when water temperatures exceed normal levels:

| Temperature | Risk Level | Description |
|------------|------------|-------------|
| < 28°C     | 🟢 Healthy | Normal conditions |
| 28-30°C    | 🟡 Warning | Stress begins, monitor closely |
| > 30°C     | 🔴 Danger  | High bleaching risk |

The system calculates a continuous **Risk Score (0-1)** for fine-grained monitoring.

---

## 🔧 Scaling to More Sensors

The current prototype uses **3 sensors**, but the system is designed to scale:

### For 4+ Sensors:

1. **Update sensor coordinates** in `01_data_preparation.ipynb`:
   ```python
   sensor_coords = [
       (8.880, 79.525),  # Sensor 1
       (8.870, 79.530),  # Sensor 2
       (8.870, 79.520),  # Sensor 3
       (8.875, 79.525),  # Sensor 4 (new)
       # Add more as needed...
   ]
   ```

2. **Modify grid generation** in `03_visualization.ipynb`:
   - For non-triangular shapes, use `scipy.spatial.ConvexHull`
   - Update the `point_in_triangle()` function to `point_in_polygon()`

3. **Retrain the model** - it will automatically handle any number of sensors!

---

## 📈 Model Performance

After running the notebooks, you should see:

### Training Metrics:
- Training Loss: ~0.001-0.005
- Validation Loss: ~0.001-0.005
- Training MAE: ~0.02-0.05
- Validation MAE: ~0.02-0.05

### Physical Temperature Scale:
- MAE: < 0.5°C
- RMSE: < 0.7°C
- Max Error: < 2°C

---

## 🌐 Viewing the Interactive Map

1. Run all cells in `03_visualization.ipynb`
2. Open the generated file: `coral_reef_risk_map.html`
3. View in any web browser (Chrome, Firefox, Edge, etc.)

The map shows:
- 📍 Sensor locations (blue markers)
- 🔺 Triangle boundary (blue dashed line)
- 🎨 Temperature heat map (color gradient)
- 📊 Risk legend (top right)

You can zoom, pan, and click on sensors for details!

---

## 🐍 Using Utility Functions

The `utils.py` file contains reusable functions:

```python
from utils import predict_temperature, load_model_and_scalers

# Load model
model, scalers = load_model_and_scalers()

# Predict temperature at a specific location and time
temp = predict_temperature(
    model, scalers,
    lat=8.875,
    lon=79.525,
    time_days=500
)

print(f"Predicted temperature: {temp:.2f}°C")
```

---

## 📦 Dependencies

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib**: Static visualizations
- **Scikit-learn**: Data normalization
- **TensorFlow**: Deep learning (PINN)
- **Folium**: Interactive maps
- **SciPy**: Scientific computing

All dependencies are listed in `requirements.txt`.

---

## 🔬 Technical Details

### Neural Network Architecture:
```
Input Layer (3):         [latitude, longitude, time]
Hidden Layer 1 (128):    tanh activation + BatchNorm
Hidden Layer 2 (128):    tanh activation + BatchNorm
Hidden Layer 3 (64):     tanh activation + BatchNorm
Hidden Layer 4 (64):     tanh activation + BatchNorm
Hidden Layer 5 (32):     tanh activation + BatchNorm
Output Layer (1):        sigmoid activation → temperature
```

### Training Configuration:
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Custom PINN Loss (MSE + Physics)
- **Batch Size**: 128
- **Epochs**: 200 (with early stopping)
- **Validation Split**: 20%

---

## 🎓 Understanding the Code

Each notebook is broken into **many small cells** for easy testing:

### 01_data_preparation.ipynb (26 cells)
- Import libraries
- Load CSV files
- Merge datasets
- Create triangle sensors
- Add variations
- Normalize data
- Visualize sensors
- Save outputs

### 02_pinn_model.ipynb (25 cells)
- Load prepared data
- Build PINN architecture
- Define custom loss
- Train model
- Plot training curves
- Evaluate performance
- Test predictions
- Save model

### 03_visualization.ipynb (27 cells)
- Load trained model
- Define risk thresholds
- Generate grid points
- Predict temperatures
- Calculate risk scores
- Create matplotlib plots
- Build Folium map
- Export results

**Total: 78 cells** - Run them one by one to see each step!

---

## 🚨 Troubleshooting

### Issue: TensorFlow not installing
**Solution**: Use conda:
```bash
conda install tensorflow
```

### Issue: Folium map not showing
**Solution**: 
- Save the map with `m.save('map.html')`
- Open `map.html` in a web browser

### Issue: Model overfitting
**Solution**:
- Increase physics loss weight in `02_pinn_model.ipynb`
- Add more data augmentation
- Reduce model complexity

### Issue: Poor predictions
**Solution**:
- Check data normalization
- Increase training epochs
- Verify sensor coordinates are correct

---

## 📚 Next Steps

1. **Real Sensor Integration**: Replace simulated data with actual sensor readings
2. **Real-time Updates**: Set up automatic retraining with new data
3. **Alert System**: Add email/SMS alerts for high-risk conditions
4. **Historical Analysis**: Analyze trends over years
5. **Multiple Reefs**: Scale to monitor multiple locations
6. **Mobile App**: Create a dashboard for field scientists

---

## 📄 License

This project is for educational and research purposes.

---

## 👥 Contact

For questions or improvements, please refer to the project documentation.

---

## ✅ Checklist

- [x] Merge CSV files
- [x] Create triangle sensor configuration
- [x] Build PINN model
- [x] Train and evaluate
- [x] Generate visualizations
- [ ] Deploy to production
- [ ] Integrate real sensors
- [ ] Add real-time monitoring

---

**Happy Coral Monitoring! 🐠🪸🌊**
