"""
Cat boosting model
"""
import pandas as pd
import numpy as np

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from service_functions import send_telegramm_message

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
file_handler = logging.FileHandler('log_cat.txt')
logger.addHandler(file_handler)

#############################################
#############################################

X = df.drop("price", axis=1)

y = df["price"]

cat_features = ["rooms", "bathrooms", "toilets", "facades_number", "number_floors", "has_terrace", "has_garden", "has_garage", "elevator", "is_furnished", "leased", "running_water", "primary_energy_consumption", "postal_code", "has_swimming_pool", "property_type_house", "property_type_other", "locality_antwerp", "locality_braine-l-alleud", "locality_brussels", "locality_gent", "locality_laken", "locality_liege", "locality_lier", "locality_mons", "locality_mouscron", "locality_namur", "locality_nivelles", "locality_oostende", "locality_other", "locality_pont-a-celles", "locality_roeselare", "locality_seraing", "locality_tournai", "locality_tubize", "locality_turnhout", "locality_wavre", "property_subtype_duplex", "property_subtype_other", "property_subtype_residence", "property_subtype_studio", "property_subtype_villa", "has_equipped_kitchen_Not equipped", "has_equipped_kitchen_Partially equipped", "has_equipped_kitchen_Super equipped"]

#############################################
#############################################
#############################################
logger.info('Begin to work...\n')

###  Split dataset  ###

float_cat_cols = [col for col in cat_features if X[col].dtype == 'float64']

for col in float_cat_cols:
    X[col] = X[col].fillna(-1).astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

#############################################
start_time = time.perf_counter()
################################

if(0):
    param_dist = {
        'learning_rate': [0.01, 0.03, 0.05, 0.08],
        'depth': [4, 5, 6, 7],
        'l2_leaf_reg': [3, 5, 7, 9],
        'bagging_temperature': [0.0, 0.5, 1.0, 1.5],
        'iterations': [500, 1000, 1500]
    }

if(1):
    param_dist = {
        'learning_rate': [0.05],
        'depth': [6],
        'l2_leaf_reg': [3],
        'bagging_temperature': [1.5],
        'iterations': [500, 1000, 2000, 3000]
    }


# Base model
model = CatBoostRegressor(
    loss_function='RMSE',
    random_seed=None,
    verbose=0,
    early_stopping_rounds=50
)

# RandomizedSearchCV
rs = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,  
    scoring='neg_mean_squared_error',
    cv=3,
    random_state=None,
    n_jobs=1
)

# FITTING 
rs.fit(
    X_train, y_train,
    eval_set=(X_valid, y_valid),
    cat_features=cat_features
)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
logger.info(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

logger.info(f"Best params of model:{rs.best_params_}")

best_model = rs.best_estimator_
y_pred = best_model.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
r2 = r2_score(y_valid, y_pred)
logger.info(f"Validation RMSE:{rmse}")
logger.info(f"Validation R2:{r2}")

logger.info(f"Job completed")

send_telegramm_message("Job completed")

#############################################
