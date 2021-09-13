import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def check_unique(series: pd.Series):
    if series.nunique() != series.count():
        print(f"{series.name} is not unique")
    return

def check_positive(series: pd.Series):
    if series.min() < 0:
        print(f"{series.name} has negative values")
    return

def check_duplicates_without_id(df: pd.DataFrame, index: str):
    duplicates = df[df[df.columns.drop(index)].duplicated()]
    if len(duplicates) > 0:
        print(duplicates)
    return

def encoding_cat(series: pd.Series) -> pd.Series:
    list_str = series.astype('category').cat.categories.to_list()
    list_num = list(np.sort(series.astype('category').cat.codes.unique()))
    return series.astype('category').cat.codes, dict(zip(list_str, list_num))

def lin_reg_simple(X, y):
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    return lin_reg