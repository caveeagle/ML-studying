"""
XGBoost model
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.pipeline import Pipeline

from xgboost import XGBRegressor

import time
#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

###  Split dataset  ###

X = df.drop("price", axis=1)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#############################################

### Frequency encoding ###

cat_cols = X_train.select_dtypes(include='object').columns

freq_maps = {}

for col in cat_cols:
    freq = X_train[col].value_counts(normalize=True)   # frequencies on train data
    freq_maps[col] = freq
    X_train[col + '_freq'] = X_train[col].map(freq)

for col in cat_cols:
    X_test[col + '_freq'] = X_test[col].map(freq_maps[col]).fillna(0)  # unknown categ -> 0

X_train = X_train.drop(columns=cat_cols)
X_test = X_test.drop(columns=cat_cols)

#############################################

model = XGBRegressor(
    n_estimators=1200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8
)

model.fit(X_train, y_train)

#############################################

preds = model.predict(X_test)
r2 = r2_score(y_test, preds)

print("Test R2:", r2)

#############################################

print('\nTask completed!')

#############################################
