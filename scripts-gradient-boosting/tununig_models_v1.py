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
file_handler = logging.FileHandler('log_1.txt')
logger.addHandler(file_handler)

#############################################
#############################################

###  PARAMETERS SETS ###

TREES_NUMBER_SET = [800,1200]
#TREES_NUMBER_SET = [300]


MAX_DEPTH_SET = [5,7]
#MAX_DEPTH_SET = [5]

LEARNING_RATE_SET = [0.05]

SUBSAMPLE_SET = [0.8]

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
    n_jobs=4
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

send_telegramm_message("Job completed")

#############################################
