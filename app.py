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


@app.post('/predict', response_class=HTMLResponse)
async def make_predictions(request: Request, sepal_length:float=Form(...), sepal_width:float=Form(...), petal_length:float=Form(...), petal_width:float=Form(...)):

    entry = [sepal_length, sepal_width, petal_length, petal_width]

    # transform data
    entry = scaler.transform([entry])

    prediction = model.predict(entry)

    if prediction[0] == 0:
        prediction = 'Setosa'
    elif prediction[0] == 1:
        prediction = 'Versicolor'
    elif prediction[0] == 2:
        prediction = 'Virginica'

    return templates.TemplateResponse("index.html", {'request':request, 'prediction':prediction})




# WITH BASEMODEL
# @app.post('/predict', response_class=HTMLResponse)
# async def make_predictions(request: Request, data:Iris_data_specs):
#     data = data.dict() # convert data to a dictionary
#     sepal_length_cm = data['sepal_length_cm']
#     sepal_width_cm = data['sepal_width_cm']
#     petal_length_cm = data['petal_length_cm'] 
#     petal_width_cm = data['petal_width_cm']

#     # transform data
#     entry = scaler.transform([[sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm]])

#     prediction = model.predict(entry)

#     if prediction[0] == 0:
#         prediction = 'Setosa'
#     elif prediction[0] == 1:
#         prediction = 'Versicolor'
#     elif prediction[0] == 2:
#         prediction = 'Virginica'

#     return templates.TemplateResponse("index.html", {'request':request, 'prediction':prediction})



# run app
if __name__ == "__main__":
    uvicorn.run(app)

# http://127.0.0.1:8000
