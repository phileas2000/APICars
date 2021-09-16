
from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from model import *
import json

app = Flask(__name__)
#api = Api(app)
#from fastapi  import FastAPI

#app = FastAPI()

modele =pickle.load(open("LinearRegressionModel.sav", 'rb'))

@app.route("/")
def main():
    parser = reqparse.RequestParser()  # initialize
    parser.add_argument('car_ID')  # add arguments
    parser.add_argument('price')
    parser.add_argument('CarName')
    parser.add_argument('horsepower')
    parser.add_argument('curbweight')
    parser.add_argument('symboling')
    data = str(parser.parse_args())
    data = chargement(data)
    data = encodage(data)
    #data = prediction(data)
    return str(data)
    #return str(pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}))
    
    

