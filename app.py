""""
ML POWERED WEB APP THAT PREDICTS THE SPECIES AN IRIS PLANT BELONGS TO BASED ON CERTAIN PARAMETERS
"""

# FastAPI module
from fastapi import FastAPI
import uvicorn

# input validation
from pydantic import ValidationError

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
from fastapi import Depends
from baseModel import Iris_data_specs

# HTML
from fastapi import Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

#CSS
from fastapi.staticfiles import StaticFiles

# app objects
app = FastAPI()
templates = Jinja2Templates(directory="templates")
# mount css to api
app.mount("/static", StaticFiles(directory="static"), name="static")


# on load
@app.get('/')
async def ok():
    return {'status code':'200'}


# model API
# get the html
@app.get('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request):
    return templates.TemplateResponse("index.html", {'request':request})

# without pydantic

@app.post('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request, sepal_length=Form(None), sepal_width=Form(None), petal_length=Form(None), petal_width=Form(None)):

    try:
        entries = [sepal_length, sepal_width, petal_length, petal_width]
        for length in entries:
            if length is None:
                error_message = "You didn't enter enything"
                return templates.TemplateResponse("index.html", {'request':request, 'error_message':error_message})
        
        entry = [float(sepal_length), float(sepal_width), float(petal_length), float(petal_width)]
       
            
    except (ValueError, ValidationError):
        error_message = "Wrong Input, Try Again"
        return templates.TemplateResponse("index.html", {'request':request, 'error_message':error_message})

    # transform data
    entry = scaler.transform([entry])

    prediction = model.predict(entry)

    if prediction[0] == 0:
        prediction = 'Setosa'
    elif prediction[0] == 1:
        prediction = 'Versicolor'
    elif prediction[0] == 2:
        prediction = 'Virginica'

    prediction = f"The iris belongs to the {prediction} species"
    return templates.TemplateResponse("index.html", {'request':request, 'prediction':prediction})




# WITH BASEMODEL
# @app.post('/predict', response_class=HTMLResponse)
# async def make_predictions(request: Request, form_data: Iris_data_specs = Depends(Iris_data_specs.as_form)):
#     try:
#         # data = float(int(data.dict())) # convert data to a dictionary
#         data = data.dict() # convert data to a dictionary
#         sepal_length_cm = data['sepal_length_cm']
#         sepal_width_cm = data['sepal_width_cm']
#         petal_length_cm = data['petal_length_cm'] 
#         petal_width_cm = data['petal_width_cm']

#         # transform data
#         entry = scaler.transform([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])

#         prediction = model.predict(entry)

#         if prediction[0] == 0:
#             prediction = 'Setosa'
#         elif prediction[0] == 1:
#             prediction = 'Versicolor'
#         elif prediction[0] == 2:
#             prediction = 'Virginica'
#     except (ValueError, ValidationError):
#         error_message = "Wrong Input, Try Again"
#         return templates.TemplateResponse("index.html", {'request':request, 'error_message':error_message})


#     return templates.TemplateResponse("index.html", {'request':request, 'prediction':prediction})



# run app
if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000
