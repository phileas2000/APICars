import pandas as pd  

from sklearn import metrics
import pickle
import json



modele =pickle.load(open("RegressorSimple.sav", 'rb'))
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
    return data
    
def score(data):
    return 'R2:'+ str(metrics.r2_score(data['price'], data['predict']))

