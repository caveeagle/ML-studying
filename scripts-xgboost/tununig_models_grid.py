"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
from xgboost import callback

import logging
import time
#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler('log_grid.txt')
logger.addHandler(file_handler)

#############################################
#############################################

###  PARAMETERS SETS ###

#TREES_NUMBER_SET = [100,300,500,800]
#TREES_NUMBER_SET = [500]
#TREES_NUMBER_SET = [450,500,550,600,700,800]

TREES_NUMBER_SET = [800]



#MAX_DEPTH_SET = [3,4,5,6]
MAX_DEPTH_SET = [6]

#LEARNING_RATE_SET = [0.05, 0.1]
LEARNING_RATE_SET = [0.05]

#SUBSAMPLE_SET = [0.7, 0.8, 0.9, 1.0]
SUBSAMPLE_SET = [0.8]

#############################################
#############################################
#############################################
logger.info('Begin to work...')
logger.info('GRID PARAMS:')

grid_str = f'TREES_NUMBER_SET = {TREES_NUMBER_SET}\n'
grid_str += f'MAX_DEPTH_SET = {MAX_DEPTH_SET}\n'

logger.info(grid_str)

###  Split dataset  ###

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

#############################################
start_time = time.perf_counter()
################################

###  Model training   ###

param_grid = {
    'n_estimators': TREES_NUMBER_SET,
    'max_depth': MAX_DEPTH_SET,
    'learning_rate': LEARNING_RATE_SET,
    'subsample': SUBSAMPLE_SET
}

grid = GridSearchCV(
    XGBRegressor(),
    param_grid,
    cv=5,    # N-fold cross-validation
    scoring='r2',
    verbose=1, # ! Changed!
    return_train_score=True, 
    n_jobs=3 # ! Changed!
)

grid.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
logger.info(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

logger.info(grid.best_params_)
logger.info(f"Best R2: {grid.best_score_}")


if 'mean_train_score' in grid.cv_results_:
    
    results = grid.cv_results_
    
    for mean_train, mean_test, params in zip(results['mean_train_score'], results['mean_test_score'], results['params']):
        logger.info(f"Train R2: {mean_train:.4f}, Test R2: {mean_test:.4f}, Params: {params}")

#############################################
#############################################
#############################################

logger.info('\nTask completed!\n\n')

if(0):
    send_telegramm_message("Job completed")

#############################################
