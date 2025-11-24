"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from service_functions import send_telegramm_message

import pickle
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
file_handler = logging.FileHandler('log_deep.txt')
logger.addHandler(file_handler)

#############################################
#############################################

###  PARAMETERS SETS ###

TREES_NUMBER_SET = [800,1200,1600,2000,3000]
#TREES_NUMBER_SET = [300]


#MAX_DEPTH_SET = [3,5,7,9]
MAX_DEPTH_SET = [5]

LEARNING_RATE_SET = [0.05]
#LEARNING_RATE_SET = [0.03, 0.04, 0.05, 0.06, 0.07]

SUBSAMPLE_SET = [0.8]
#SUBSAMPLE_SET = [0.5, 0.7, 0.8, 0.9, 1.0]

#############################################
#############################################
#############################################
logger.info('Begin to work...\n')

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

###  Model training   ###

param_grid = {
    'n_estimators': TREES_NUMBER_SET,
    'max_depth': MAX_DEPTH_SET,
    'learning_rate': LEARNING_RATE_SET,
    'subsample': SUBSAMPLE_SET    
}

grid = GridSearchCV(
    GradientBoostingRegressor(random_state=None),
    param_grid,
    cv=5,            # 5-fold cross-validation
    scoring='r2',
    verbose=2,  
    n_jobs=-1
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

if(1):
    send_telegramm_message("Job completed")

#############################################
