import streamlit as st 
import numpy as np
import pandas as pd

model = pd.read_pickle('model.pkl')
scaler = pd.read_pickle('scaler.pkl')

def predict_price(bedrooms, bathrooms, sqft_living, sqft_lot):
    input = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot]])
    input_scale = scaler.transform(input)
    
    prediction = model.predict(input_scale)
    return prediction[0]



st.title("House Price Prediction")

bedrooms = st.number_input("Bedrooms",min_value=0.0,step=0.5)
bathrooms = st.number_input("Bathrooms",min_value=0.0, step=0.5)
sqft_living = st.number_input("Square Footage (Living)",min_value=0.0, step=0.5)
sqft_lot = st.number_input("Square Footage (Lot)",min_value=0.0, step=0.5)

if st.button("Predict Price"):
    if bedrooms >= 0 and bathrooms >= 0 and sqft_living >= 0 and sqft_lot >=0:
        predicted_price = predict_price(bedrooms, bathrooms, sqft_living, sqft_lot)
        st.success(f'Predicted Price:- ${predicted_price:,.2f}')

    else:
        st.error("Please enter valid input.")
    


