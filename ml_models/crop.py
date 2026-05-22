import joblib
import numpy as np
from pathlib import Path

_PKL = Path(__file__).parent.parent / "Page Two" / "models" / "crop_model.pkl"

_model = None
_features_order = None
_reverse_crop = None


def _load():
    global _model, _features_order, _reverse_crop  # prevent create new local variable 
    try:
        data = joblib.load(_PKL)
        _model = data["model"]
        _features_order = data["features"]
        crop_map = data["crop_map"]
        _reverse_crop = {v: k for k, v in crop_map.items()}
        print("Crop model loaded")
        print(f"   Features: {_features_order}")
    except Exception as e:
        print(f"Crop model not found: {e} — running in demo mode")


_load()

SOIL_MAP = {"Clay": 0, "Loamy": 1, "Calcareous": 2, "Sandy": 3}
REGION_MAP = {"Delta": 0, "Upper_Egypt": 1, "North_Coast": 2, "Sinai": 3, "NewLands": 4}
SEASON_MAP = {"Winter": 0, "Summer": 1, "Nile": 2}

CLIMATE_DEFAULTS = {
    ("Clay",       "Delta",       "Winter"): {"N": 90,  "P": 40,  "K": 225, "ph": 7.2, "om": 2.5, "temp": 14, "humid": 70, "rain": 22},
    ("Loamy",      "Delta",       "Winter"): {"N": 75,  "P": 32,  "K": 190, "ph": 6.8, "om": 2.1, "temp": 14, "humid": 70, "rain": 22},
    ("Calcareous", "Delta",       "Winter"): {"N": 55,  "P": 25,  "K": 160, "ph": 8.0, "om": 1.4, "temp": 14, "humid": 65, "rain": 20},
    ("Sandy",      "Delta",       "Winter"): {"N": 35,  "P": 19,  "K": 110, "ph": 6.5, "om": 0.7, "temp": 14, "humid": 65, "rain": 20},
    ("Clay",       "Delta",       "Summer"): {"N": 90,  "P": 40,  "K": 225, "ph": 7.2, "om": 2.5, "temp": 32, "humid": 70, "rain": 4},
    ("Loamy",      "Delta",       "Summer"): {"N": 75,  "P": 32,  "K": 190, "ph": 6.8, "om": 2.1, "temp": 32, "humid": 70, "rain": 4},
    ("Sandy",      "Delta",       "Summer"): {"N": 35,  "P": 19,  "K": 110, "ph": 6.5, "om": 0.7, "temp": 34, "humid": 62, "rain": 2},
    ("Calcareous", "Delta",       "Summer"): {"N": 55,  "P": 25,  "K": 160, "ph": 8.0, "om": 1.4, "temp": 33, "humid": 63, "rain": 3},
    ("Clay",       "Delta",       "Nile"):   {"N": 90,  "P": 40,  "K": 225, "ph": 7.2, "om": 2.5, "temp": 26, "humid": 68, "rain": 3},
    ("Loamy",      "Delta",       "Nile"):   {"N": 75,  "P": 32,  "K": 190, "ph": 6.8, "om": 2.1, "temp": 26, "humid": 67, "rain": 3},
    ("Sandy",      "Delta",       "Nile"):   {"N": 35,  "P": 19,  "K": 110, "ph": 6.5, "om": 0.7, "temp": 26, "humid": 65, "rain": 2},
    ("Clay",       "Upper_Egypt", "Winter"): {"N": 88,  "P": 38,  "K": 220, "ph": 7.3, "om": 2.3, "temp": 17, "humid": 45, "rain": 5},
    ("Loamy",      "Upper_Egypt", "Winter"): {"N": 72,  "P": 30,  "K": 185, "ph": 6.9, "om": 2.0, "temp": 17, "humid": 45, "rain": 5},
    ("Sandy",      "Upper_Egypt", "Winter"): {"N": 33,  "P": 18,  "K": 108, "ph": 6.4, "om": 0.6, "temp": 17, "humid": 43, "rain": 4},
    ("Clay",       "Upper_Egypt", "Summer"): {"N": 88,  "P": 38,  "K": 220, "ph": 7.3, "om": 2.3, "temp": 38, "humid": 35, "rain": 1},
    ("Loamy",      "Upper_Egypt", "Summer"): {"N": 72,  "P": 30,  "K": 185, "ph": 6.9, "om": 2.0, "temp": 38, "humid": 35, "rain": 1},
    ("Sandy",      "Upper_Egypt", "Summer"): {"N": 33,  "P": 18,  "K": 108, "ph": 6.4, "om": 0.6, "temp": 40, "humid": 30, "rain": 0},
    ("Clay",       "Upper_Egypt", "Nile"):   {"N": 88,  "P": 38,  "K": 220, "ph": 7.3, "om": 2.3, "temp": 29, "humid": 43, "rain": 1},
    ("Loamy",      "Upper_Egypt", "Nile"):   {"N": 72,  "P": 30,  "K": 185, "ph": 6.9, "om": 2.0, "temp": 29, "humid": 43, "rain": 1},
    ("Clay",       "North_Coast", "Winter"): {"N": 85,  "P": 36,  "K": 210, "ph": 7.0, "om": 2.2, "temp": 13, "humid": 75, "rain": 60},
    ("Loamy",      "North_Coast", "Winter"): {"N": 70,  "P": 28,  "K": 180, "ph": 6.7, "om": 1.9, "temp": 13, "humid": 75, "rain": 60},
    ("Clay",       "North_Coast", "Summer"): {"N": 85,  "P": 36,  "K": 210, "ph": 7.0, "om": 2.2, "temp": 30, "humid": 66, "rain": 2},
    ("Loamy",      "North_Coast", "Summer"): {"N": 70,  "P": 28,  "K": 180, "ph": 6.7, "om": 1.9, "temp": 30, "humid": 65, "rain": 2},
    ("Clay",       "North_Coast", "Nile"):   {"N": 85,  "P": 36,  "K": 210, "ph": 7.0, "om": 2.2, "temp": 24, "humid": 64, "rain": 4},
    ("Clay",       "Sinai",       "Winter"): {"N": 75,  "P": 28,  "K": 180, "ph": 7.4, "om": 1.8, "temp": 12, "humid": 55, "rain": 18},
    ("Sandy",      "Sinai",       "Winter"): {"N": 30,  "P": 16,  "K": 100, "ph": 6.3, "om": 0.5, "temp": 12, "humid": 53, "rain": 16},
    ("Clay",       "Sinai",       "Nile"):   {"N": 75,  "P": 28,  "K": 180, "ph": 7.4, "om": 1.8, "temp": 26, "humid": 50, "rain": 2},
    ("Sandy",      "Sinai",       "Nile"):   {"N": 30,  "P": 16,  "K": 100, "ph": 6.3, "om": 0.5, "temp": 26, "humid": 48, "rain": 2},
    ("Clay",       "NewLands",    "Winter"): {"N": 70,  "P": 25,  "K": 175, "ph": 7.5, "om": 1.5, "temp": 15, "humid": 50, "rain": 2},
    ("Clay",       "NewLands",    "Summer"): {"N": 70,  "P": 25,  "K": 175, "ph": 7.5, "om": 1.5, "temp": 34, "humid": 35, "rain": 1},
    ("Sandy",      "NewLands",    "Summer"): {"N": 30,  "P": 15,  "K": 100, "ph": 6.5, "om": 0.5, "temp": 35, "humid": 33, "rain": 0},
}

DEMO_PREDICTIONS = {
    ("Clay",       "Delta",       "Winter"): "Wheat",
    ("Loamy",      "Delta",       "Winter"): "Wheat",
    ("Calcareous", "Delta",       "Winter"): "Wheat",
    ("Sandy",      "Delta",       "Winter"): "Fava Beans",
    ("Clay",       "Delta",       "Summer"): "Rice",
    ("Loamy",      "Delta",       "Summer"): "Maize",
    ("Sandy",      "Delta",       "Summer"): "Maize",
    ("Calcareous", "Delta",       "Summer"): "Maize",
    ("Clay",       "Delta",       "Nile"):   "Tomato",
    ("Loamy",      "Delta",       "Nile"):   "Tomato",
    ("Sandy",      "Delta",       "Nile"):   "Tomato",
    ("Clay",       "Upper_Egypt", "Winter"): "Wheat",
    ("Loamy",      "Upper_Egypt", "Winter"): "Wheat",
    ("Sandy",      "Upper_Egypt", "Winter"): "Fava Beans",
    ("Clay",       "Upper_Egypt", "Summer"): "Sugarcane",
    ("Loamy",      "Upper_Egypt", "Summer"): "Sugarcane",
    ("Sandy",      "Upper_Egypt", "Summer"): "Cotton",
    ("Clay",       "Upper_Egypt", "Nile"):   "Tomato",
    ("Loamy",      "Upper_Egypt", "Nile"):   "Tomato",
    ("Clay",       "North_Coast", "Winter"): "Wheat",
    ("Loamy",      "North_Coast", "Winter"): "Wheat",
    ("Clay",       "North_Coast", "Summer"): "Maize",
    ("Loamy",      "North_Coast", "Summer"): "Tomato",
    ("Clay",       "North_Coast", "Nile"):   "Tomato",
    ("Clay",       "Sinai",       "Winter"): "Wheat",
    ("Sandy",      "Sinai",       "Winter"): "Fava Beans",
    ("Clay",       "Sinai",       "Nile"):   "Tomato",
    ("Sandy",      "Sinai",       "Nile"):   "Tomato",
    ("Clay",       "NewLands",    "Winter"): "Wheat",
    ("Clay",       "NewLands",    "Summer"): "Maize",
    ("Sandy",      "NewLands",    "Summer"): "Maize",
}


def predict(soil: str, region: str, season: str) -> dict:
    if _model is not None:
        key = (soil, region, season)
        # search using the key in the climate defaults
        defaults = CLIMATE_DEFAULTS.get(key, {
            "N": 70, "P": 30, "K": 180, "ph": 7.0, "om": 1.5,
            "temp": 25, "humid": 60, "rain": 10,
        })
        # put all inputs and features in the dict
        input_dict = {
            "nitrogen_mg_kg":     defaults["N"],
            "phosphorus_mg_kg":   defaults["P"],
            "potassium_mg_kg":    defaults["K"],
            "soil_ph":            defaults["ph"],
            "organic_matter_pct": defaults["om"],
            "temperature_avg_c":  defaults["temp"],
            "humidity_pct":       defaults["humid"],
            "rainfall_mm":        defaults["rain"],
            "soil_type":          SOIL_MAP[soil],
            "region":             REGION_MAP[region],
            "season":             SEASON_MAP[season],
        }
        X = np.array([[input_dict[f] for f in _features_order]]) # features order in the same oredr of training 
        pred = _model.predict(X)[0]
        proba = _model.predict_proba(X)[0]               # probability of all classes
        confidence = round(float(proba.max()) * 100, 1)  # top probability 
        crop_name = _reverse_crop[int(pred)] 
        return {
            "crop": crop_name,
            "confidence": confidence,
            "mode": "model",
            "inputs": {"soil": soil, "region": region, "season": season},
        }
    else:
        key = (soil, region, season)
        crop_name = DEMO_PREDICTIONS.get(key, "Wheat")
        return {
            "crop": crop_name,
            "confidence": 88,
            "mode": "demo",
            "inputs": {"soil": soil, "region": region, "season": season},
        }
