import joblib
import numpy as np
from pathlib import Path

_PKL = Path(__file__).parent.parent / "Page Three" / "Model" / "yield_model.pkl"

_model = None
_features_order = None
_crop_map = None
_region_map = None
_season_map = None
_soil_map = None
_avg_yield = None


def _load():
    global _model, _features_order, _crop_map, _region_map, _season_map, _soil_map, _avg_yield
    try:
        data = joblib.load(_PKL)
        _model = data["model"]
        _features_order = data["features"]
        _crop_map = data["crop_map"]
        _region_map = data["region_map"]
        _season_map = data["season_map"]
        _soil_map = data["soil_map"]
        _avg_yield = data["avg_yield"]
        print("Yield model loaded")
        print(f" Features: {_features_order}")
    except Exception as e:
        print(f" Yield model not found: {e} — running in demo mode")


_load()
# Real numbers for model comparison
YIELD_PER_FEDDAN = {
    "Wheat":      {"Clay": 2.85, "Loamy": 2.75, "Calcareous": 2.20, "Sandy": 1.80},
    "Rice":       {"Clay": 3.90, "Loamy": 3.65, "Calcareous": 2.80, "Sandy": 2.40},
    "Maize":      {"Clay": 3.40, "Loamy": 3.50, "Calcareous": 2.70, "Sandy": 2.60},
    "Cotton":     {"Clay": 0.78, "Loamy": 0.75, "Calcareous": 0.65, "Sandy": 0.68},
    "Sugarcane":  {"Clay": 48.0, "Loamy": 45.0, "Calcareous": 38.0, "Sandy": 35.0},
    "Tomato":     {"Clay": 12.0, "Loamy": 12.5, "Calcareous": 9.50, "Sandy": 10.5},
    "Potato":     {"Clay": 8.50, "Loamy": 9.00, "Calcareous": 7.20, "Sandy": 8.80},
    "Fava Beans": {"Clay": 0.90, "Loamy": 0.85, "Calcareous": 0.92, "Sandy": 0.78},
}

REGION_FACTOR = {
    "Delta": 1.00, "Upper_Egypt": 0.92,
    "North_Coast": 0.85, "Sinai": 0.75, "NewLands": 0.80,
}

SOIL_EFFICIENCY = {"Clay": 100, "Loamy": 96, "Calcareous": 80, "Sandy": 72}

NAT_AVG = {
    "Wheat": 2.85, "Rice": 3.80, "Maize": 3.40, "Cotton": 0.78,
    "Sugarcane": 47.0, "Tomato": 11.5, "Potato": 8.50, "Fava Beans": 0.88,
}


def predict(crop: str, season: str, region: str, soil: str, area_feddans: float) -> dict:
    if _model is not None:
        input_dict = {
            "crop":         _crop_map[crop],
            "region":       _region_map[region],
            "season":       _season_map[season],
            "soil_type":    _soil_map[soil],
            "area_feddans": area_feddans,
        }
        X = np.array([[input_dict[f] for f in _features_order]])
        yield_per_feddan = round(float(_model.predict(X)[0]), 2)
        total_yield = round(yield_per_feddan * area_feddans, 2)
        soil_eff = SOIL_EFFICIENCY.get(soil, 85)
        nat_avg = _avg_yield.get(crop, NAT_AVG.get(crop, 3))
        bar_pct = min(round((yield_per_feddan / nat_avg) * 100), 100)
        return {
            "yield_per_feddan": yield_per_feddan,
            "total_yield":      total_yield,
            "soil_efficiency":  soil_eff,
            "national_avg":     nat_avg,
            "bar_pct":          bar_pct,
            "mode":             "model",
            "inputs": {"crop": crop, "season": season, "region": region, "soil": soil, "area": area_feddans},
        }
    else:
        base_yield = YIELD_PER_FEDDAN.get(crop, {}).get(soil, 2.5)
        region_factor = REGION_FACTOR.get(region, 1.0)
        yield_per_feddan = round(base_yield * region_factor, 2)
        total_yield = round(yield_per_feddan * area_feddans, 2)
        soil_eff = SOIL_EFFICIENCY.get(soil, 85)
        nat_avg = NAT_AVG.get(crop, 3)
        bar_pct = min(round((yield_per_feddan / nat_avg) * 100), 100)
        return {
            "yield_per_feddan": yield_per_feddan,
            "total_yield":      total_yield,
            "soil_efficiency":  soil_eff,
            "national_avg":     nat_avg,
            "bar_pct":          bar_pct,
            "mode":             "demo",
            "inputs": {"crop": crop, "season": season, "region": region, "soil": soil, "area": area_feddans},
        }
