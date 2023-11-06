""""
THIS APP DOESN'T WORK WELL, THERE'S A SCALING ISSUE IN THE MACHINE LEARNING MODEL

"""



# FastAPI module
from fastapi import FastAPI
import uvicorn

# data handling
import numpy as np
import pandas as pd

# data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# model
import pickle 
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file) # the ML Model

# scaler
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file) # the Scaler

# data type specs
from baseModel import Iris_data_specs



# app objects
app = FastAPI()

# on load
@app.get('/')
def ok():
    return {'status code':'200'}


# model API
@app.post('/predict')
def make_predictions(data:Iris_data_specs):
    data = data.dict() # convert data to a dictionary
    sepal_length_cm = data['sepal_length_cm']
    sepal_width_cm = data['sepal_width_cm']
    petal_length_cm = data['petal_length_cm'] 
    petal_width_cm = data['petal_width_cm']

    # transform data
    entry = scaler.transform([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])

    prediction = model.predict(entry)

    if prediction[0] == 0:
        prediction = 'Setosa'
    elif prediction[0] == 1:
        prediction = 'Versicolor'
    elif prediction[0] == 2:
        prediction = 'Virginica'

    return {'prediction' : prediction}



# run app
if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000
