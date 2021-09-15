from fastapi import FastAPI
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

lin_reg = pickle.load(open('linear_reg_model.sav', 'rb'))

app = FastAPI()

@app.get('/predict/')
def predict_price(curbweight: int, enginesize: int, ):
    X = pd.DataFrame({'curbweight': [curbweight], 'enginesize': [enginesize]})
    y_pred = lin_reg.predict(X)
    return {'price': y_pred[0][0]}