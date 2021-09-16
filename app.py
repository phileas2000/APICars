from fastapi import FastAPI, Request
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

from starlette.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory = "templates")

lin_reg = pickle.load(open('linear_reg_model.sav', 'rb'))

app = FastAPI()

@app.get("/")
async def read_index():
    return FileResponse('templates/index.html')

@app.get('/predict/', response_class=HTMLResponse)
def predict_price(request: Request, curbweight: int, enginesize: int, ):
    X = pd.DataFrame({'curbweight': [curbweight], 'enginesize': [enginesize]})
    y_pred = lin_reg.predict(X)[0][0]
    return templates.TemplateResponse("index.html", {'request': request, 'price': y_pred})