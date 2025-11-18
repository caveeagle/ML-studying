"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle

import time
#############################################

filename = '../data/fitted_dataset.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

###  PARAMETERS  ###

TEST_SIZE = 0.2

RANDOM_SEED = 2020

TREES_NUMBER = 200

MAX_DEPTH = None

#############################################

###  Split dataset  ###

# Target variable
y = df.iloc[:, 0]  # Price (in Euro)

# Features
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=TEST_SIZE,
                                                    random_state=RANDOM_SEED)

#############################################
start_time = time.perf_counter()
################################

###  Model training   ###

model = RandomForestRegressor(n_estimators=TREES_NUMBER,
                              max_depth=MAX_DEPTH,
                              min_samples_leaf=MIN_LEAFS_NUM
                              random_state=RANDOM_SEED)

model.fit(X_train, y_train)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

###  Prediction and evaluation  ###

y_pred = model.predict(X_test)

# Root Mean Squared Error (RMSE)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE, in Euros

print(f'RMSE = {rmse:.0f}')

# Mean Absolute Error (MAE)

mae = mean_absolute_error(y_test, y_pred)

print(f'MAE = {mae:.0f}')

# R2 (coefficient of determination)

r2 = r2_score(y_test, y_pred)

print(f'R2 = {r2:.3f}')

#############################################

SAVE_MODEL = 0

models_file_name = '../models/random_forest_model.pkl'

if(SAVE_MODEL):
    with open(models_file_name, 'wb') as f:
        pickle.dump(model, f)

#############################################

print('\nTask completed!')

#############################################
