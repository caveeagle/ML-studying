"""
XGBoost model
"""
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
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
#############################################

### Class Frequency encoding ###

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_maps_ = {}
        self.cat_cols_ = None
        
    def fit(self, X, y=None):
        self.cat_cols_ = X.select_dtypes(include='object').columns
        for col in self.cat_cols_:
            self.freq_maps_[col] = X[col].value_counts(normalize=True)
        return self
    
    def transform(self, X):
        X_new = X.copy()
        for col in self.cat_cols_:
            X_new[col + '_freq'] = X_new[col].map(self.freq_maps_[col]).fillna(0)
        X_new = X_new.drop(columns=self.cat_cols_)
        return X_new

#############################################
#############################################

pipe = Pipeline([
    ('freq_enc', FrequencyEncoder()),
    ('model', XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8
    ))
])

pipe.fit(X_train, y_train)

#############################################

preds = pipe.predict(X_test)
r2 = r2_score(y_test, preds)

print("Test R2:", r2)

#############################################

print('\nTask completed!')

#############################################
