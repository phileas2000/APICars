import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import scipy.stats as st

import src.utils as utl

def lin_reg_curbweight(df: pd.DataFrame):
    X = df.curbweight.values.reshape(-1, 1)
    y = df.price.values.reshape(-1, 1)
    lin_reg = utl.lin_reg_simple(X, y)
    a = lin_reg.coef_[0][0]
    b = lin_reg.intercept_[0]
    print('coeff :', a)
    print('intercept', b)
    print('r² score :', lin_reg.score(X, y))
    y_pred = lin_reg.predict(X)
    sns.regplot(X, y, line_kws={"color": "red", 'alpha': 0.35})
    plt.figure()
    residuals = y - y_pred
    sns.distplot(residuals)
    print('kurtosis : ', st.kurtosis(residuals)[0])
    print('skewness : ', st.skew(residuals)[0])
    return

def lin_reg_multiple_price(df: pd.DataFrame):
    X = df[['curbweight', 'enginesize']]
    y = df.price.values.reshape(-1, 1)
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    print('coeff :', lin_reg.coef_)
    print('intercept', lin_reg.intercept_)
    print('r² score :', lin_reg.score(X, y))
    return

def encoding_bin(df: pd.DataFrame):
    df['doornumber'] = df.doornumber.map({'two': 2, 'four': 4})
    df['cylindernumber'] = df.cylindernumber.map({'two': 2, 'three': 3, 'four': 4, 
                                                    'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})
    return pd.get_dummies(df)

def encoding_cat_df(df: pd.DataFrame) -> pd.DataFrame:
    df['doornumber'] = df.doornumber.map({'two': 2, 'four': 4})
    df['cylindernumber'] = df.cylindernumber.map({'two': 2, 'three': 3, 'four': 4, 
                                                    'five': 5, 'six': 6, 'eight': 8, 'twelve': 12})
    cat_df = df.select_dtypes(include = ['object', 'category'])
    cat_keys = []
    for col in cat_df.columns.to_list():
        cat_df[col], keys = utl.encoding_cat(cat_df[col])
        cat_keys.append(keys)
    cat_keys = dict(zip(cat_df.columns, cat_keys))
    df.update(cat_df)
    return df, cat_keys

def lin_reg_cross_simple(df: pd.DataFrame, random_state: int):
    train, test = train_test_split(df, test_size = 0.3, random_state = random_state)
    y_train = train.price
    y_test = test.price
    X_train = train.drop('price', axis = 1)
    X_test = test.drop('price', axis = 1)
    lin_reg = LinearRegression()
    X = X_train.curbweight.values.reshape(-1, 1)
    y = y_train.values.reshape(-1, 1)
    lin_reg.fit(X, y)
    print('train r² : ', lin_reg.score(X, y))
    y_pred = lin_reg.predict(X_test.curbweight.values.reshape(-1, 1))
    print('test r² : ', metrics.r2_score(y_test, y_pred))

def lin_reg_cross_multiple(df: pd.DataFrame, random_state: int):
    train, test = train_test_split(df, test_size = 0.3, random_state = random_state)
    y_train = train.price
    y_test = test.price
    X_train = train.drop('price', axis = 1)
    X_test = test.drop('price', axis = 1)
    lin_reg = LinearRegression()
    X = X_train[['curbweight', 'enginesize']]
    y = y_train.values.reshape(-1, 1)
    lin_reg.fit(X, y)
    train_r2 = lin_reg.score(X, y)
    print('train r² : ', train_r2)
    y_pred = lin_reg.predict(X_test[['curbweight', 'enginesize']])
    test_r2 = metrics.r2_score(y_test, y_pred)
    print('test r² : ', test_r2)
    return lin_reg