import joblib
import numpy as np
from pathlib import Path

_PKL = Path(__file__).parent.parent / "Page Four" / "Model" / "profit_model.pkl"

_model = None
_features_order = None
_crop_map = None
_season_map = None
_soil_map = None
_region_map = None
_avg_price = None


def _load():
    global _model, _features_order, _crop_map, _season_map, _soil_map, _region_map, _avg_price
    try:
        data = joblib.load(_PKL)
        _model = data["model"]
        _features_order = data["features"]
        _crop_map = data["crop_map"]
        _season_map = data["season_map"]
        _soil_map = data["soil_map"]
        _region_map = data["region_map"]
        _avg_price = data["avg_price"]
        print("Profit model loaded")
        print(f"   Features: {_features_order}")
    except Exception as e:
        print(f" Profit model not found: {e} — running in demo mode")


_load()

REAL_PRICES = {
    "Wheat": 18000, "Maize": 13024, "Rice": 23001, "Cotton": 29845,
    "Tomato": 4418, "Potato": 5946, "Sugarcane": 2500, "Fava Beans": 44502,
}

AVG_YIELD = {
    "Wheat": 2.2, "Maize": 2.7, "Rice": 3.9, "Cotton": 1.1,
    "Sugarcane": 48.0, "Tomato": 13.7, "Potato": 11.2, "Fava Beans": 4.6,
}

AVG_COSTS = {
    "Wheat":      {"seed": 900,  "fertilizer": 700,  "irrigation": 500,  "labor": 600,  "pesticide": 350},
    "Rice":       {"seed": 1000, "fertilizer": 800,  "irrigation": 900,  "labor": 700,  "pesticide": 450},
    "Maize":      {"seed": 800,  "fertilizer": 650,  "irrigation": 500,  "labor": 550,  "pesticide": 350},
    "Cotton":     {"seed": 1300, "fertilizer": 900,  "irrigation": 700,  "labor": 900,  "pesticide": 600},
    "Sugarcane":  {"seed": 2200, "fertilizer": 2000, "irrigation": 1800, "labor": 2200, "pesticide": 600},
    "Tomato":     {"seed": 1600, "fertilizer": 1300, "irrigation": 1400, "labor": 1600, "pesticide": 900},
    "Potato":     {"seed": 2200, "fertilizer": 1600, "irrigation": 1400, "labor": 1600, "pesticide": 700},
    "Fava Beans": {"seed": 650,  "fertilizer": 350,  "irrigation": 350,  "labor": 450,  "pesticide": 220},
}

DEFAULT_REGION = {
    "Wheat": "Delta", "Rice": "Delta", "Maize": "Delta", "Cotton": "Delta",
    "Tomato": "Sinai", "Potato": "North_Coast", "Sugarcane": "Upper_Egypt", "Fava Beans": "Delta",
}


def predict(crop: str, season: str, soil: str, area_feddans: float, region: str = None) -> dict:
    if region is None:
        region = DEFAULT_REGION.get(crop, "Delta")

    costs = AVG_COSTS.get(crop, {})
    scale = area_feddans ** 0.92
    seed_cost  = round(costs.get("seed",       800) * scale)
    fert_cost  = round(costs.get("fertilizer", 700) * scale)
    irr_cost   = round(costs.get("irrigation", 500) * scale)
    labor_cost = round(costs.get("labor",      600) * scale)
    pest_cost  = round(costs.get("pesticide",  350) * scale)

    yield_per_feddan = AVG_YIELD.get(crop, 3.0)
    total_yield = round(yield_per_feddan * area_feddans, 2)
    price = REAL_PRICES.get(crop, 10000)
    gross_revenue = round(total_yield * price)

    if _model is not None:
        input_dict = {
            "crop":                _crop_map[crop],
            "area_feddans":        area_feddans,
            "season":              _season_map[season],
            "soil_type":           _soil_map[soil],
            "region":              _region_map.get(region, 0),
            "seed_cost_egp":       seed_cost,
            "fertilizer_cost_egp": fert_cost,
            "irrigation_cost_egp": irr_cost,
            "labor_cost_egp":      labor_cost,
            "pesticide_cost_egp":  pest_cost,
        }
        X = np.array([[input_dict[f] for f in _features_order]])
        profit_per_feddan = round(float(_model.predict(X)[0]))
        net_profit = round(profit_per_feddan * area_feddans)
        total_cost = gross_revenue - net_profit
        mode = "model"
    else:
        net_profit = gross_revenue - total_cost
        profit_per_feddan = round(net_profit / area_feddans)
        mode = "demo"

    return {
        "profit_per_feddan": profit_per_feddan,
        "net_profit":        net_profit,
        "gross_revenue":     gross_revenue,
        "seed_cost":         seed_cost,
        "fertilizer_cost":   fert_cost,
        "irrigation_cost":   irr_cost,
        "labor_cost":        labor_cost,
        "pesticide_cost":    pest_cost,
        "total_cost":        total_cost,
        "total_yield":       total_yield,
        "price":             price,
        "mode":              mode,
        "inputs": {"crop": crop, "season": season, "soil": soil, "area": area_feddans},
    }
