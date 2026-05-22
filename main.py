from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # enables cross-origin communication between the frontend and backend
from routers import crop, yield_, profit  # to bring each router for all models 

app = FastAPI(title="AgriAdvisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # Accept from any location or address
    allow_methods=["*"],     # Accept GET, POST, and all types of requests.
    allow_headers=["*"],     # Accept any headers within the order
)

app.include_router(crop.router,    prefix="/crop",   tags=["Crop Predictor"])   # Register crop router with '/crop' base path
app.include_router(yield_.router,  prefix="/yield",  tags=["Yield Estimator"])   # Register yield router with '/yield' base path
app.include_router(profit.router,  prefix="/profit", tags=["Profit Calculator"])  # Register profit router with '/profit' base path
