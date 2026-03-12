import os
import io
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib

app = FastAPI(
    title="Flight Price Prediction API",
    description="Predicts Indian domestic flight prices using XGBoost + LightGBM Ensemble",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models ──
xgb_model = None
lgb_model = None
feature_cols = None

@app.on_event("startup")
async def load_models():
    global xgb_model, lgb_model, feature_cols
    xgb_model    = joblib.load("flight_price_xgb_final.pkl")
    lgb_model    = joblib.load("flight_price_lgb_final.pkl")
    feature_cols = joblib.load("flight_feature_cols.pkl")
    print("✅ Models loaded!")

# ── Mappings ──
AIRLINE_ENC = {
    "IndiGo":0,"Air India":1,"Jet Airways":2,"SpiceJet":3,
    "Multiple carriers":4,"GoAir":5,"Vistara":6,"Air Asia":7,
    "Jet Airways Business":8,"Multiple carriers Premium economy":9,
    "Vistara Premium economy":10,"TruJet":11
}
SOURCE_ENC = {"Banglore":0,"Chennai":1,"Delhi":2,"Kolkata":3,"Mumbai":4}
DEST_ENC   = {"Banglore":0,"Cochin":1,"Delhi":2,"Hyderabad":3,"Kolkata":4,"New Delhi":5}
ADDINFO_ENC = {"No info":0,"In-flight meal not included":1,"No check-in baggage included":2,
               "1 Short layover":3,"1 Long layover":4,"Change airports":5,
               "Business class":6,"Red-eye flight":7,"2 Long layover":8}
STOPS_MAP = {"non-stop":0,"1 stop":1,"2 stops":2,"3 stops":3,"4 stops":4}
TOD_ENC   = {"morning":0,"afternoon":1,"evening":2,"night":3}
BUDGET    = ["IndiGo","SpiceJet","GoAir","Air Asia","TruJet"]
BUSINESS  = ["Jet Airways Business","Multiple carriers Premium economy","Vistara Premium economy"]

class FlightInput(BaseModel):
    airline: str
    source: str
    destination: str
    dep_hour: int
    dep_minute: int
    duration_mins: int
    stops: str
    journey_day: int
    journey_month: int
    journey_weekday: int
    additional_info: str = "No info"
    arr_hour: int = 12
    arr_minute: int = 0

def time_of_day(hour):
    if 5 <= hour < 12: return "morning"
    elif 12 <= hour < 17: return "afternoon"
    elif 17 <= hour < 21: return "evening"
    else: return "night"

def build_features(inp: FlightInput) -> pd.DataFrame:
    tod = time_of_day(inp.dep_hour)
    stops_num = STOPS_MAP.get(inp.stops, 1)
    airline_enc = AIRLINE_ENC.get(inp.airline, 0)
    source_enc  = SOURCE_ENC.get(inp.source, 0)
    dest_enc    = DEST_ENC.get(inp.destination, 0)
    tod_enc     = TOD_ENC.get(tod, 0)
    addinfo_enc = ADDINFO_ENC.get(inp.additional_info, 0)
    is_weekend  = 1 if inp.journey_weekday in [5,6] else 0
    airline_tier = 0 if inp.airline in BUDGET else (2 if inp.airline in BUSINESS else 1)
    route_length = stops_num + 1

    d = {
        "journey_day": inp.journey_day,
        "journey_month": inp.journey_month,
        "journey_weekday": inp.journey_weekday,
        "is_weekend": is_weekend,
        "Dep_Hour": inp.dep_hour,
        "Dep_Minute": inp.dep_minute,
        "Arr_Hour": inp.arr_hour,
        "Arr_Minute": inp.arr_minute,
        "Duration_mins": inp.duration_mins,
        "Duration_hours": inp.duration_mins / 60,
        "Stops_num": stops_num,
        "Route_length": route_length,
        "Airline_tier": airline_tier,
        "Airline_enc": airline_enc,
        "Source_enc": source_enc,
        "Destination_enc": dest_enc,
        "Dep_TimeOfDay_enc": tod_enc,
        "Additional_Info_enc": addinfo_enc,
        "Duration_sq": inp.duration_mins ** 2,
        "Stops_x_Duration": stops_num * inp.duration_mins,
        "Dep_x_Stops": inp.dep_hour * stops_num,
        "Month_x_Airline": inp.journey_month * airline_enc,
        "Is_morning": 1 if 5 <= inp.dep_hour < 12 else 0,
        "Is_night": 1 if inp.dep_hour >= 21 else 0,
        "Stops_x_Airline": stops_num * airline_enc,
        "Duration_x_Stops": inp.duration_mins * stops_num,
        "Airline_x_Month": airline_enc * inp.journey_month,
        "Src_x_Dst": source_enc * dest_enc,
        "Dep_x_Month": inp.dep_hour * inp.journey_month,
        "Duration_log": np.log1p(inp.duration_mins),
        "Stops_sq": stops_num ** 2,
        "Hour_sq": inp.dep_hour ** 2,
        "Airline_tier_x_Stops": airline_tier * stops_num,
        "Route_x_Stops": route_length * stops_num,
    }
    df = pd.DataFrame([d])
    # Align with training feature order
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]

@app.get("/")
def root():
    return {"status": "online", "model": "XGBoost + LightGBM Ensemble", "r2": 0.9405, "mae": 554}

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": xgb_model is not None}

@app.get("/options")
def options():
    return {
        "airlines": list(AIRLINE_ENC.keys()),
        "sources": list(SOURCE_ENC.keys()),
        "destinations": list(DEST_ENC.keys()),
        "stops": list(STOPS_MAP.keys()),
        "additional_info": list(ADDINFO_ENC.keys())
    }

@app.post("/predict")
async def predict(inp: FlightInput):
    if xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    try:
        features = build_features(inp)
        xgb_log = xgb_model.predict(features)[0]
        lgb_log = lgb_model.predict(features)[0]
        avg_log = 0.5 * xgb_log + 0.5 * lgb_log
        price = float(np.expm1(avg_log))
        xgb_price = float(np.expm1(xgb_log))
        lgb_price = float(np.expm1(lgb_log))

        # Price range (±10%)
        price_low  = round(price * 0.90)
        price_high = round(price * 1.10)

        return JSONResponse({
            "predicted_price": round(price),
            "price_range": {"low": price_low, "high": price_high},
            "model_predictions": {
                "xgboost": round(xgb_price),
                "lightgbm": round(lgb_price),
                "ensemble": round(price)
            },
            "model_stats": {"r2": 0.9405, "mae": 554, "model": "XGBoost + LightGBM Ensemble"}
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))