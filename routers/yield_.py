from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import ml_models.yield_ as yield_model

router = APIRouter()        # create a separate router for each feature in the project 

_VALID_SEASONS = {
    "Wheat":      ["Winter"],
    "Fava Beans": ["Winter"],
    "Maize":      ["Summer", "Nile"],
    "Rice":       ["Summer"],
    "Cotton":     ["Summer"],
    "Sugarcane":  ["Summer", "Nile"],
    "Tomato":     ["Nile", "Winter", "Summer"],
    "Potato":     ["Winter", "Nile"],
}

_VALID_SOILS = {
    "Rice":       ["Clay", "Loamy"],
    "Sugarcane":  ["Clay", "Loamy", "Sandy"],
    "Cotton":     ["Sandy", "Loamy", "Calcareous"],
    "Wheat":      ["Clay", "Loamy", "Sandy", "Calcareous"],
    "Maize":      ["Clay", "Loamy", "Sandy", "Calcareous"],
    "Tomato":     ["Sandy", "Loamy", "Clay", "Calcareous"],
    "Potato":     ["Sandy", "Loamy"],
    "Fava Beans": ["Clay", "Sandy", "Loamy", "Calcareous"],
}


class YieldInput(BaseModel):         #automatically validates incoming request data before passing it to the prediction function.
    crop: str
    season: str
    region: str
    soil: str
    area_feddans: float


@router.post("/predict")
def predict(data: YieldInput):   # Receive validated input data from the frontend
    if data.area_feddans <= 0:
        return JSONResponse(status_code=400, content={"error": "Area must be greater than 0"})
    if data.season not in _VALID_SEASONS.get(data.crop, []):
        return JSONResponse(status_code=400, content={
            "error":      "invalid_combination",
            "message":    f"{data.crop} cannot be grown in {data.season} season",
            "suggestion": f"Valid seasons for {data.crop}: {_VALID_SEASONS.get(data.crop, [])}",
        })
    if data.soil not in _VALID_SOILS.get(data.crop, []):
        return JSONResponse(status_code=400, content={
            "error":      "invalid_combination",
            "message":    f"{data.crop} is not suitable for {data.soil} soil",
            "suggestion": f"Valid soils for {data.crop}: {_VALID_SOILS.get(data.crop, [])}",
        })
    return yield_model.predict(data.crop, data.season, data.region, data.soil, data.area_feddans)  # Pass inputs to the model and return prediction result to the frontend
