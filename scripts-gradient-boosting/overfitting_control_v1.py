"""
Gradient Boosting Model
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold


import time
#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

###  Split dataset  ###

X = df.drop("price", axis=1)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=None)

#############################################
start_time = time.perf_counter()
################################

model = GradientBoostingRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    random_state=None
)

kf = KFold(n_splits=5, shuffle=True, random_state=202)

scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print("R2 on each fold:", scores)
print("Mean R2:", np.mean(scores))
print("Std for R2:", np.std(scores))
