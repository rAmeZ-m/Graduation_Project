from fastapi import APIRouter
from fastapi.responses import JSONResponse    # responsible for returning the response
from pydantic import BaseModel                # validate from data shape
from typing import Optional                   #import to determine the type of data.
import ml_models.profit as profit_model

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


class ProfitInput(BaseModel):           #automatically validates incoming request data before passing it to the prediction function.
    crop: str
    season: str
    soil: str
    area_feddans: float
    region: Optional[str] = None


@router.post("/predict")
def predict(data: ProfitInput):           # Receive validated input data from the frontend
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
    return profit_model.predict(data.crop, data.season, data.soil, data.area_feddans, data.region)   # Pass inputs to the model and return prediction result to the frontend
