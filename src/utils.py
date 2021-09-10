import pandas as pd

def check_unique(series: pd.Series):
    if series.nunique() != series.count():
        print(f"{series.name} is not unique")
    return

def check_positive(series: pd.Series):
    if series.min() < 0:
        print(f"{series.name} has negative values")
    return

def check_duplicates_without_id(df: pd.DataFrame, index: str):
    duplicates = df[df[df.columns.drop('car_ID')].duplicated()]
    if len(duplicates) > 0:
        print(duplicates)
    return

def encoding_cat(series: pd.Series) -> pd.Series:
    return series.astype('category').cat.codes