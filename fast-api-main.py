
## import libraries
from utils import process_new
from fastapi import FastAPI, Form
import joblib  
import numpy as np
import utils
from datetime import datetime
import pandas as pd
import os

## Load the model
MODEL_PATH = os.path.join(os.getcwd(), 'grid_search_model.pkl')
model = joblib.load(MODEL_PATH)


## Initialize an app
app = FastAPI()

@app.post('/predict_house_price')
async def model_regression(Bedrooms: int = Form(...),
                     Bathrooms: int = Form(...),
                     PropertySize: float = Form(...),
                     NumberOfFloors: int = Form(...),
                     CompletionYear: int = Form(...),
                     TotalUnits: int = Form(...),
                     ParkingLot: int = Form(...),
                     TenureType: str = Form(..., description="Tenure Type", enum=list(utils.X_train['tenure_type'].value_counts().index.sort_values())),
                     PropertyType: str = Form(..., description="Property Type", enum=list(utils.X_train['property_type'].value_counts().index.sort_values())),
                     FloorRange: str = Form(..., description="Floor Range", enum=list(utils.X_train['floor_range'].value_counts().index.sort_values())),
                     LandTitle: str = Form(..., description="Land Title", enum=list(utils.X_train['land_title'].value_counts().index.sort_values())),
                     FirmType: str = Form(..., description="Firm Type", enum=list(utils.X_train['firm_type'].value_counts().index.sort_values())),
                     FirmNumber: str = Form(..., description="Firm Number", enum=list(utils.X_train['firm_number'].value_counts().index.sort_values())),
                     RenType: str = Form(..., description="Ren Type", enum=list(utils.X_train['ren_type'].value_counts().index.sort_values())),
                     RenNumber: str = Form(..., description="Ren Number", enum=list(utils.X_train['ren_number'].value_counts().index.sort_values().astype(int))),
                     NearbyMall: bool = Form(...),
                     NearbySchool: bool = Form(...),
                     NearbyRailwayStation: bool = Form(...),
                     NearbyHighway: bool = Form(...),
                     NearbyHospital: bool = Form(...),
                     NearbyBusStop: bool = Form(...),
                     NearbyPark: bool = Form(...),
                     FacilityParking: bool = Form(...),
                     FacilitySecurity: bool = Form(...),
                     FacilitySwimmingPool: bool = Form(...),
                     FacilityPlayground: bool = Form(...),
                     FacilityBarbequeArea: bool = Form(...),
                     FacilityJoggingTrack: bool = Form(...),
                     FacilityMinimart: bool = Form(...),
                     FacilityLift: bool = Form(...), 
                     FacilityGymnasium: bool = Form(...),
                     FacilityMultipurposeHall: bool = Form(...),
                     FacilitySauna: bool = Form(...),
                     FacilityTennisCourt: bool = Form(...),
                     FacilityClubHouse: bool = Form(...),
                     FacilitySquashCourt: bool = Form(...),
                     State: str = Form(..., description="Tenure Type", enum=list(utils.X_train['state'].value_counts().index.sort_values())),
                     City: str = Form(..., description="Tenure Type", enum=list(utils.X_train['city'].value_counts().index.sort_values())),
                     BuildingName: str = Form(..., description="Tenure Type", enum=list(utils.X_train['building_name'].value_counts().index.sort_values())),
                     Developer: str = Form(..., description="Tenure Type", enum=list(utils.X_train['developer'].value_counts().index.sort_values()))):

    ## Concatenate the users data
    X_new = np.array([Bedrooms, Bathrooms, PropertySize, NearbySchool, NearbyMall, BuildingName, Developer, TenureType, CompletionYear, 
                        NumberOfFloors, TotalUnits, PropertyType, ParkingLot, FloorRange, LandTitle, FirmType, FirmNumber, RenNumber,
                        NearbyRailwayStation , NearbyHighway, NearbyHospital, NearbyBusStop, NearbyPark, FacilityParking,
                        FacilitySecurity  , FacilitySwimmingPool, FacilityPlayground, FacilityBarbequeArea, FacilityJoggingTrack,
                        FacilityMinimart , FacilityLift, FacilityGymnasium, FacilityMultipurposeHall, FacilitySauna, FacilityTennisCourt,
                        FacilityClubHouse ,FacilitySquashCourt , RenType, State, City])
    

    ## Preprocessing
    X_processed = utils.process_new(X_new)

    ## Prediction
    y_pred = model.predict(X_processed)
 

    return f'Predicted Price = {np.exp(y_pred[0]):,.2f}'