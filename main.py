
## import libraries
from utils import process_new
import streamlit as st
import joblib  
import numpy as np
import utils

## Load the model
model = joblib.load('grid_search_model.pkl')

def model_regression():

    ## Title
    st.title('🏙️ Malaysian Condominium Prices Prediction')
    st.markdown('<hr>', unsafe_allow_html=True)

    ## Input Fields
    score = st.number_input('Score', value=0, step=1)
    sel = st.selectbox('Geography', options=['France', 'Germany', 'Egypt'])
    sel_num = st.selectbox('Geography', options=[1, 2])
    tenure = st.slider('Tenure', min_value=1, max_value=10, step=1)
    st.markdown('<hr>', unsafe_allow_html=True)

    if st.button('Predict Price'):
        ## Concatenate the users data
        X_new = np.array([score, sel, sel_num, tenure])

        ## Preprocessing
        X_processed = utils.process_new(X_new)

        ## Prediction
        y_pred = model.predict(X_processed)

        ## Display results
        st.success(f'Predicted Price = {np.exp(y_pred[0])}')

    return None

if __name__ == '__main__':
    model_regression()