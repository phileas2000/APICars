import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as st

import utils as utl

def lin_reg_curbweight(df: pd.DataFrame):
    X = df.curbweight.values.reshape(-1, 1)
    y = df.price.values.reshape(-1, 1)
    lin_reg = lin_reg_simple(X, y)
    a = lin_reg.coef_[0][0]
    b = lin_reg.intercept_[0]
    print('coeff :', a)
    print('intercept', b)
    print('r² score :', lin_reg.score(X, y))
    y_pred = lin_reg.predict(X)    
    plt.plot(X, a * X + b, color = 'red', alpha = 0.35)
    sns.scatterplot(X.reshape(1, -1)[0], y.reshape(1, -1)[0])
    plt.figure()
    residuals = y - y_pred
    sns.distplot(residuals)
    print('kurtosis : ', st.kurtosis(residuals)[0])
    print('skewness : ', st.skew(residuals)[0])
    return

def lin_reg_simple(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg

def lin_reg_multiple_price(df: pd.DataFrame):
    X = df[['curbweight', 'enginesize']]
    y = df.price.values.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print('coeff :', lin_reg.coef_)
    print('intercept', lin_reg.intercept_)
    print('r² score :', lin_reg.score(X, y))
    return

def encoding_cat_df(df: pd.DataFrame) -> pd.DataFrame:
    df['brand'] = df.CarName.str.split().str[0]
    cat_df = df.apply(utl.encoding_cat)
    cat_df['doornumber'] = df.doornumber.map({'two': 2, 'four': 4})
    cat_df['cylindernumber'] = df.cylindernumber.map({'two': 2, 'three': 3, 'four': 4, 
                                                    'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})
    return cat_df

