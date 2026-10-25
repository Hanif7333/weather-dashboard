import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Excel file path
file_name = 'weather_station_logs_2026-10-25.xlsx'

print("Loading dataset...")
df = pd.read_excel(file_name, sheet_name='Logs')
df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
df = df.sort_values('TIMESTAMP').reset_index(drop=True)

feature_cols = [
    'Wind Speed (km/h)', 
    'Ambient Temperature (°C)', 
    'Humidity (%)', 
    'Pressure (hPa)'
]
target_col = 'Ambient Temperature (°C)'

# Clean data (remove NaNs and startup zeros)
clean_df = df[['TIMESTAMP'] + feature_cols].dropna()
clean_df = clean_df[clean_df[target_col] > 0.0].reset_index(drop=True)

# Model & Scaler setup
scaler = StandardScaler()
model = SGDRegressor(max_iter=1000, tol=1e-3)

initial_window_size = 120
train_data = clean_df.iloc[:initial_window_size]
X_train = train_data[feature_cols].values
y_train = train_data[target_col].shift(-1).dropna().values
X_train = X_train[:-1]

X_train_scaled = scaler.fit_transform(X_train)
model.partial_fit(X_train_scaled, y_train)

# Index tracker for live streaming simulation via web
current_index = initial_window_size
error_history = []  # To track recent absolute errors for MAE calculation

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})

@app.get("/get_next_prediction")
async def get_next_prediction():
    global current_index, error_history
    
    if current_index >= len(clean_df) - 1:
        current_index = initial_window_size  # Reset if loop ends
        error_history = []
        
    row = clean_df.iloc[current_index]
    next_row = clean_df.iloc[current_index + 1]
    
    # Predict next minute temperature
    X_new = row[feature_cols].values.reshape(1, -1)
    X_new_scaled = scaler.transform(X_new)
    prediction = float(model.predict(X_new_scaled)[0])
    
    actual_val = float(next_row[target_col])
    
    # Calculate absolute error for this step and store in history (keep last 20)
    step_error = abs(prediction - actual_val)
    error_history.append(step_error)
    if len(error_history) > 20:
        error_history.pop(0)
    
    current_mae = round(float(np.mean(error_history)), 3)
    
    # Calculate Accuracy % based on MAE and actual temperature value (Bounded between 0% and 100%)
    current_temp = float(row[target_col])
    if current_temp > 0:
        # Formula: 100 - (MAE / Current Temperature * 100)
        raw_accuracy = 100.0 - (current_mae / abs(current_temp) * 100.0)
        current_accuracy = round(max(0.0, min(100.0, raw_accuracy)), 2)
    else:
        current_accuracy = 0.0
    
    # Continuous Online Training (partial_fit with new incoming row)
    X_feed = next_row[feature_cols].values.reshape(1, -1)
    y_feed = np.array([actual_val])
    X_feed_scaled = scaler.partial_fit(X_feed).transform(X_feed)
    model.partial_fit(X_feed_scaled, y_feed)
    
    result = {
        "time": row['TIMESTAMP'].strftime('%Y-%m-%d %H:%M:%S'),
        "input_temp": round(float(row[target_col]), 2),
        "predicted_temp": round(prediction, 2),
        "actual_temp": actual_val,
        "wind_speed": round(float(row['Wind Speed (km/h)']), 2),
        "humidity": round(float(row['Humidity (%)']), 2),
        "pressure": round(float(row['Pressure (hPa)']), 2),
        "mae": current_mae,
        "accuracy": current_accuracy
    }
    
    current_index += 1
    return result