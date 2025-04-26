import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout, Concatenate, Attention
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import joblib
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
import uvicorn
import os
import json
from datetime import datetime, timedelta

# Initialize FastAPI application
app = FastAPI(
    title="Sales and Inventory Forecasting API",
    description="API for forecasting sales and inventory using a bidirectional LSTM with attention",
    version="1.0.0"
)

# Define input data models for API
class TrainingData(BaseModel):
    time_series: List[Dict[str, Union[str, float]]]
    feature_columns: List[str]
    target_column: str
    date_column: str
    sequence_length: int = Field(default=12, description="Number of time steps to use for prediction")
    forecast_horizon: int = Field(default=7, description="Number of steps to forecast into the future")
    
class ForecastRequest(BaseModel):
    model_id: str
    input_data: List[Dict[str, Union[str, float]]]
    horizon: Optional[int] = Field(default=7, description="Number of steps to forecast")

class ModelInfo(BaseModel):
    model_id: str
    feature_columns: List[str]
    target_column: str
    date_column: str
    sequence_length: int
    forecast_horizon: int
    created_at: str

# Dictionary to store model metadata
models_registry = {}

# Directories for model storage
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Helper functions for data preprocessing
def preprocess_data(df, feature_columns, target_column, sequence_length):
    """Preprocess data for Bi-LSTM model"""
    # Scale features
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    # Fit and transform features
    features_scaled = feature_scaler.fit_transform(df[feature_columns])
    
    # Fit and transform target
    target_scaled = target_scaler.fit_transform(df[[target_column]])
    
    # Create sequences
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(features_scaled[i:i+sequence_length])
        y.append(target_scaled[i+sequence_length])
    
    return np.array(X), np.array(y), feature_scaler, target_scaler

def create_sequences_from_new_data(data, feature_columns, feature_scaler, sequence_length):
    """Create sequences from new data for prediction"""
    features = feature_scaler.transform(data[feature_columns])
    
    # Use the last sequence_length data points for prediction
    if len(features) >= sequence_length:
        return np.array([features[-sequence_length:]])
    else:
        raise ValueError(f"Not enough data points. Need at least {sequence_length} data points.")

# Function to build the Bi-LSTM model with attention
def build_bilstm_attention_model(sequence_length, n_features):
    """Build a bidirectional LSTM model with attention layer"""
    # Input layer
    inputs = Input(shape=(sequence_length, n_features))
    
    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm2 = Bidirectional(LSTM(32, return_sequences=True))(lstm1)
    
    # Self-attention layer
    attention_layer = Attention()([lstm2, lstm2])
    
    # Concatenate attention output with lstm2
    concat = Concatenate()([lstm2, attention_layer])
    
    # Output layers
    dropout = Dropout(0.2)(concat)
    lstm_out = LSTM(16)(dropout)
    outputs = Dense(1)(lstm_out)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    
    return model

@app.post("/train", response_model=ModelInfo)
async def train_model(data: TrainingData):
    """Train a new forecasting model"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data.time_series)
        
        # Validate input data
        required_columns = data.feature_columns + [data.target_column, data.date_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        # Convert date column to datetime
        df[data.date_column] = pd.to_datetime(df[data.date_column])
        
        # Sort by date
        df = df.sort_values(by=data.date_column)
        
        # Preprocess data
        X, y, feature_scaler, target_scaler = preprocess_data(
            df, data.feature_columns, data.target_column, data.sequence_length
        )
        
        # Build and train model
        model = build_bilstm_attention_model(data.sequence_length, len(data.feature_columns))
        
        # Split data into train/validation
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Train model with early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Generate model ID
        model_id = f"forecast_model_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Save model and scalers
        model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")
        feature_scaler_path = os.path.join(MODEL_DIR, f"{model_id}_feature_scaler.pkl")
        target_scaler_path = os.path.join(MODEL_DIR, f"{model_id}_target_scaler.pkl")
        
        model.save(model_path)
        joblib.dump(feature_scaler, feature_scaler_path)
        joblib.dump(target_scaler, target_scaler_path)
        
        # Store model metadata
        model_info = ModelInfo(
            model_id=model_id,
            feature_columns=data.feature_columns,
            target_column=data.target_column,
            date_column=data.date_column,
            sequence_length=data.sequence_length,
            forecast_horizon=data.forecast_horizon,
            created_at=datetime.now().isoformat()
        )
        
        models_registry[model_id] = model_info.dict()
        
        # Save registry to file
        with open(os.path.join(MODEL_DIR, "models_registry.json"), "w") as f:
            json.dump(models_registry, f)
        
        return model_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast")
async def forecast(request: ForecastRequest):
    """Generate forecasts using a trained model"""
    try:
        # Check if model exists
        if request.model_id not in models_registry:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Load model metadata
        model_info = models_registry[request.model_id]
        
        # Load model and scalers
        model_path = os.path.join(MODEL_DIR, f"{request.model_id}.h5")
        feature_scaler_path = os.path.join(MODEL_DIR, f"{request.model_id}_feature_scaler.pkl")
        target_scaler_path = os.path.join(MODEL_DIR, f"{request.model_id}_target_scaler.pkl")
        
        model = tf.keras.models.load_model(model_path)
        feature_scaler = joblib.load(feature_scaler_path)
        target_scaler = joblib.load(target_scaler_path)
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame(request.input_data)
        
        # Check required columns
        required_columns = model_info["feature_columns"] + [model_info["date_column"]]
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing_columns}")
        
        # Convert date column to datetime
        input_df[model_info["date_column"]] = pd.to_datetime(input_df[model_info["date_column"]])
        
        # Sort by date
        input_df = input_df.sort_values(by=model_info["date_column"])
        
        # Create sequences for prediction
        X_pred = create_sequences_from_new_data(
            input_df, 
            model_info["feature_columns"], 
            feature_scaler, 
            model_info["sequence_length"]
        )
        
        # Determine forecast horizon
        horizon = request.horizon if request.horizon else model_info["forecast_horizon"]
        
        # Generate forecasts
        forecasts = []
        last_sequence = X_pred[0].copy()
        last_date = input_df[model_info["date_column"]].iloc[-1]
        
        for i in range(horizon):
            # Make prediction
            pred = model.predict(np.array([last_sequence]), verbose=0)[0][0]
            
            # Inverse transform the prediction
            pred_value = target_scaler.inverse_transform([[pred]])[0][0]
            
            # Next date
            next_date = last_date + timedelta(days=1)
            
            # Store forecast
            forecasts.append({
                "date": next_date.strftime("%Y-%m-%d"),
                model_info["target_column"]: float(pred_value)
            })
            
            # Update last sequence by removing the first time step and adding the new prediction
            # This assumes the target is also a feature, otherwise needs adjustment
            if model_info["target_column"] in model_info["feature_columns"]:
                target_idx = model_info["feature_columns"].index(model_info["target_column"])
                # Shift sequence and add new prediction
                last_sequence = np.vstack([last_sequence[1:], np.zeros((1, len(model_info["feature_columns"])))])
                last_sequence[-1, target_idx] = pred
            
            # Update last date
            last_date = next_date
            
        return {"forecasts": forecasts}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all trained models"""
    return {"models": list(models_registry.values())}

@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model"""
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return models_registry[model_id]

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    if model_id not in models_registry:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Remove model files
    model_path = os.path.join(MODEL_DIR, f"{model_id}.h5")
    feature_scaler_path = os.path.join(MODEL_DIR, f"{model_id}_feature_scaler.pkl")
    target_scaler_path = os.path.join(MODEL_DIR, f"{model_id}_target_scaler.pkl")
    
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(feature_scaler_path):
        os.remove(feature_scaler_path)
    if os.path.exists(target_scaler_path):
        os.remove(target_scaler_path)
    
    # Remove from registry
    del models_registry[model_id]
    
    # Update registry file
    with open(os.path.join(MODEL_DIR, "models_registry.json"), "w") as f:
        json.dump(models_registry, f)
    
    return {"message": f"Model {model_id} deleted successfully"}

# Run the application
if __name__ == "__main__":
    # Load existing models registry if it exists
    registry_path = os.path.join(MODEL_DIR, "models_registry.json")
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            models_registry = json.load(f)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
