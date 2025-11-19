"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import pickle
import logging
import time
#############################################

filename = '../data/fitted_dataset.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

###  PARAMETERS  ###

TEST_SIZE = 0.15

TREES_NUMBER = 300

MODEL_CYCLES_NUM = 10

#############################################


rmse_list = []
mae_list = []
r2_list = []

for n in range(MODEL_CYCLES_NUM):

    #############################################
    #############################################
    #############################################
    
    y = df.iloc[:, 0]  # Price (in Euro)
    X = df.iloc[:, 1:]
    
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=TEST_SIZE,
                                                        random_state=None)
    
    #############################################
    start_time = time.perf_counter()
    ################################
    
    model = RandomForestRegressor(n_estimators=TREES_NUMBER,
                                  max_depth=None,
                                  min_samples_leaf=2,
                                  random_state=None)
    
    model.fit(X_train, y_train)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("\nCycle:",n)
    #print(f"Timer: {elapsed_time:.0f} sec\n")
    
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE, in Euros
    
    rmse_list.append(rmse)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    mae_list.append(mae)
    
    r2 = r2_score(y_test, y_pred)
    
    r2_list.append(r2)
    
    ### END OF CYCLE ###

#############################################
#############################################
#############################################
    
rmse_mean = np.mean(rmse_list)
mae_mean = np.mean(mae_list)
r2_mean = np.mean(r2_list)

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler('log_03.txt')
logger.addHandler(file_handler)

#############################################

logger.info(f"For the best model (TREES_NUMBER={TREES_NUMBER})\n")


logger.info(f"RMSE: {rmse_mean:.3f}")
logger.info(f"MAE: {mae_mean:.3f}")
logger.info(f"R2: {r2_mean:.3f}")

#############################################

print('\nTask completed!')

#############################################
