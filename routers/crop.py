from fastapi import APIRouter
from fastapi.responses import JSONResponse   # responsible for returning the response
from pydantic import BaseModel               # validate from data shape
import ml_models.crop as crop_model

router = APIRouter()     # create a separate router for each feature in the project 

_VALID_SOILS   = {"Clay", "Loamy", "Calcareous", "Sandy"}
_VALID_REGIONS = {"Delta", "Upper_Egypt", "North_Coast", "Sinai", "NewLands"}
_VALID_SEASONS = {"Winter", "Summer", "Nile"}

# any request must be this 
class CropInput(BaseModel):  #automatically validates incoming request data before passing it to the prediction function.
    soil: str
    region: str
    season: str


@router.post("/predict")  # recive inputs from the web
def predict(data: CropInput):
    if data.soil not in _VALID_SOILS:
        return JSONResponse(status_code=400, content={"error": f"Invalid soil: {data.soil}"})
    if data.region not in _VALID_REGIONS:
        return JSONResponse(status_code=400, content={"error": f"Invalid region: {data.region}"})
    if data.season not in _VALID_SEASONS:
        return JSONResponse(status_code=400, content={"error": f"Invalid season: {data.season}"})
    return crop_model.predict(data.soil, data.region, data.season) #calls the prediction function from the crop model and return prediction result to the frontend
