
from flask import Flask
from flask_restful import Resource, Api, reqparse
import pandas as pd
from model import *

app = Flask(__name__)
#api = Api(app)
#from fastapi  import FastAPI

#app = FastAPI()

modele =pickle.load(open("LinearRegressionModel.sav", 'rb'))

@app.route("/")
def main():
    parser = reqparse.RequestParser()  # initialize
    #parser.add_argument('car_ID')  # add arguments
    parser.add_argument('price')
    #parser.add_argument('horsepower')
    parser.add_argument('curbweight')
    #parser.add_argument('symboling')
    data = str(parser.parse_args())
    data = chargement(data)
    #data = encodage(data)
    #data = prediction(data)
    #return "Prix prédit: "+ str(data["predict"]) +" Prix réel: " + str(data["price"])
    return str(data)
    
    

