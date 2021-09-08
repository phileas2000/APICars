import pandas as pd
import utils as utl

def cleaning_global() -> pd.DataFrame:
    df = pd.read_csv("../data/RAW/cars.csv")
    utl.check_unique(df['car_ID'])

    df_positive = df[['wheelbase', 'carlength', 'carwidth', 'curbweight', 
                    'enginesize', 'boreratio', 'stroke', 'horsepower', 'peakrpm', 
                    'citympg', 'highwaympg', 'price']]
    df_positive.apply(utl.check_positive)
    utl.check_duplicates_without_id(df, 'car_ID')
    df.CarName = df.CarName.str.replace('alfa-romero', 'alfa-romeo')
    df.CarName = df.CarName.str.replace('maxda', 'mazda')
    df.CarName = df.CarName.str.replace('Nissan', 'nissan')
    df.CarName = df.CarName.str.replace('porcshce', 'porsche')
    df.CarName = df.CarName.str.replace('toyouta', 'toyota')
    df.CarName = df.CarName.str.replace('vokswagen', 'volkswagen')
    df.CarName = df.CarName.str.replace('vw', 'volkswagen')
    return df