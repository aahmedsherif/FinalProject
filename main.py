
## import libraries
from utils import process_new
import streamlit as st
import joblib  
import numpy as np
import utils
from datetime import datetime
import pandas as pd
import os


## Load the model
MODEL_PATH = os.path.join(os.getcwd, 'grid_search_model.pkl')
model = joblib.load(MODEL_PATH)

def model_regression():

    ## Title
    st.title('🏙️ Malaysian Condominium Prices Prediction')
    st.markdown('<hr>', unsafe_allow_html=True)


    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            bedroom = st.number_input('Bedrooms', value=0, step=1)
        with col2:
            bathroom = st.number_input('Bathrooms', value=0, step=1)
        with col3:
            property_size = st.number_input('Property Size', value=0.0, step=0.01)
        with col4:
            number_of_floors = st.number_input('Number of Floors', value=0, step=1)
             

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            completion_year = st.number_input('Completion Year', value=datetime.now().year, step=1)
        with col2:
            total_units = st.number_input('Total Units', value=0, step=1)
        with col3:
            parking_lot = st.number_input('Parking Lot', value=0, step=1)
        #with col4:
            
    st.markdown('<hr>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            tenure_type = st.selectbox('Tenure Type', options=utils.X_train['tenure_type'].value_counts().index.sort_values())
        with col2: 
            property_type = st.selectbox('Property Type', options=utils.X_train['property_type'].value_counts().index.sort_values())
        with col3: 
            floor_range = st.selectbox('Floor Range', options=utils.X_train['floor_range'].value_counts().index.sort_values())
        with col4:
            land_title = st.selectbox('Land Title', options=utils.X_train['land_title'].value_counts().index.sort_values())
    
    

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            firm_type = st.selectbox('Firm Type', options=utils.X_train['firm_type'].value_counts().index.sort_values())
        with col2: 
            firm_number = st.selectbox('Firm Number', options=utils.X_train['firm_number'].value_counts().index.sort_values())
        with col3: 
            ren_type = st.selectbox('Ren Type', options=utils.X_train['ren_type'].value_counts().index.sort_values())
        with col4:
            ren_number = st.selectbox('Ren Number', options=utils.X_train['ren_number'].value_counts().index.sort_values())
    
    st.markdown('<hr>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nearby_mall = st.checkbox('Nearby Mall', value=False)
        with col2: 
            nearby_school = st.checkbox('Nearby School', value=False)
        with col3: 
            nearby_railway_station = st.checkbox('Nearby Railway Station', value=False)
        with col4:
            nearby_highway = st.checkbox('Nearby Highway', value=False)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            nearby_hospital = st.checkbox('Nearby Hospital', value=False) 
        with col2: 
            nearby_bus_stop = st.checkbox('Nearby Bus Stop', value=False)
        with col3: 
            nearby_park = st.checkbox('Nearby Park', value=False)
        #with col4: 
    
 
    st.markdown('<hr>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            facility_Parking = st.checkbox('Facility Parking', value=False)
        with col2: 
            facility_Security = st.checkbox('Facility Security', value=False)
        with col3: 
            facility_Swimming_Pool = st.checkbox('Facility Swimming Pool', value=False)
        with col4:
            facility_Playground = st.checkbox('Facility Playground', value=False)
    
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            facility_Barbeque_area = st.checkbox('Facility Barbeque area', value=False)
        with col2: 
            facility_Jogging_Track = st.checkbox('Facility Jogging Track', value=False)
        with col3: 
            facility_Minimart = st.checkbox('Facility Minimart', value=False)
        with col4:
            facility_Lift = st.checkbox('Facility Lift', value=False)
     
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            facility_Gymnasium = st.checkbox('Facility Gymnasium', value=False)
        with col2: 
            facility_Multipurpose_hall = st.checkbox('Facility Multipurpose Hall', value=False)
        with col3: 
            facility_Sauna = st.checkbox('Facility Sauna', value=False)
        with col4:
            facility_Tennis_Court = st.checkbox('Facility Tennis Court', value=False)
     
     
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            facility_Club_house = st.checkbox('Facility Club House', value=False)
        with col2: 
            facility_Squash_Court = st.checkbox('Facility Squash Court', value=False)
        #with col3: 
             
        #with col4:
    
    st.markdown('<hr>', unsafe_allow_html=True)
             
     

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            state = st.selectbox('State', options=np.append([np.nan], utils.X_train['state'].value_counts().index.sort_values()))
        with col2: 
            city = st.selectbox('City', options=np.append([np.nan], utils.X_train['city'].value_counts().index.sort_values()))
        #with col3: 
             
        #with col4:
     
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            building_name = st.selectbox('Building Name', options=np.append([np.nan], utils.X_train['building_name'].value_counts().index.sort_values())) 
        with col2:
            developer = st.selectbox('Developer', options=np.append([np.nan], utils.X_train['developer'].value_counts().index.sort_values()))

    if st.button('Predict Price'):

        ## Concatenate the users data
        X_new = np.array([bedroom, bathroom, property_size, nearby_school, nearby_mall, building_name, developer, tenure_type, completion_year, 
                          number_of_floors, total_units, property_type, parking_lot, floor_range, land_title, firm_type, firm_number, ren_number,
                          nearby_railway_station , nearby_highway, nearby_hospital, nearby_bus_stop, nearby_park, facility_Parking,
                          facility_Security  , facility_Swimming_Pool, facility_Playground, facility_Barbeque_area, facility_Jogging_Track,
                           facility_Minimart , facility_Lift, facility_Gymnasium, facility_Multipurpose_hall, facility_Sauna, facility_Tennis_Court,
                            facility_Club_house ,facility_Squash_Court , ren_type, state, city])
        

        ## Preprocessing
        X_processed = utils.process_new(X_new)

        ## Prediction
        y_pred = model.predict(X_processed)

        ## Display results
        st.success(f'Predicted Price = {np.exp(y_pred[0]):,.2f}')
        #st.success(f'Predicted Price = {type(developer)}')

    return None

if __name__ == '__main__':
    model_regression()