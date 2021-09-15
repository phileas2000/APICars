import pandas as pd
import pickle

import src.cleaning_utils as cln
import src.modeling_utils as mdl
import src.utils as utl

df_og = cln.cleaning_global()
df = df_og.copy()

vehicle_price = df.groupby('CarName').price.mean()
df['price_order'] = pd.qcut(df.price, 3, labels=['cheap', 'medium', 'high'])

reg_df = df[['price', 'curbweight', 'enginesize', 'price_order']]

df_categ, cat_keys = mdl.encoding_cat_df(df)

X = df[['curbweight', 'enginesize']]
y = df.price.values.reshape(-1, 1)
lin_reg = utl.lin_reg_simple(X, y)

pickle.dump(lin_reg, open("linear_reg_model.sav", 'wb'))