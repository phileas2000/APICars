import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle
import json

from random import sample

modele =pickle.load(open("LinearRegressionModel.sav", 'rb'))
encoder =pickle.load(open("OrdinalEncoder.sav", 'rb'))

def chargement(data):
    data = data.replace("'","\"")
    data = json.loads(data)
    data = pd.DataFrame(data.items())
    data = data.T
    data.columns = data.iloc[0]
    data = data.drop(0)
    data = data.reset_index(drop=True)
    return data

def nettoyage(data):
    data = data.drop_duplicates()
    return data

def encodage(data):
    from sklearn.preprocessing import OrdinalEncoder
    data["CarName"] = encoder.transform(data["CarName"].values.reshape(-1,1))
    return data
    

def prediction(data):
    x = data.drop(columns=['price']).values.reshape(-1,1)
    y = data['price']
    data['predict'] = modele.predict(x)
    return data['predict']
    
def score(data,modele):
    return 'R2:'+ str(metrics.r2_score(data['price'], data['predict']))


  