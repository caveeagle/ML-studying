"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

import pickle
import logging

import time
#############################################

filename = '../data/fitted_dataset_v4.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler('log_02.txt')
logger.addHandler(file_handler)

#############################################
#############################################

TEST_SIZE = 0.2

###  PARAMETERS SETS ###

#TREES_NUMBER_SET = [100,200,300,400,500]
TREES_NUMBER_SET = [200,500,800]


#MAX_DEPTH_SET = [None,10,15,20,25]
MAX_DEPTH_SET = [20]

MIN_LEAFS_NUM = [1, 2]
#MIN_LEAFS_NUM = [2]

#############################################
#############################################
#############################################
logger.info('Begin to work...\n')

###  Split dataset  ###

# Target variable
y = df.iloc[:, 0]  # Price (in Euro)

# Features
X = df.iloc[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=TEST_SIZE,
                                                    random_state=None)

#############################################
start_time = time.perf_counter()
################################

###  Model training   ###

param_grid = {
    'n_estimators': TREES_NUMBER_SET,
    'max_depth': MAX_DEPTH_SET,
    'min_samples_leaf': MIN_LEAFS_NUM
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=None),
    param_grid,
    cv=5,            # 5-fold cross-validation
    scoring='r2',  
    n_jobs=3
)

grid.fit(X_train, y_train)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
logger.info(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

logger.info(grid.best_params_)
logger.info(f"Best R2: {grid.best_score_}")

#############################################
#############################################
#############################################

logger.info('\nTask completed!')

#############################################
