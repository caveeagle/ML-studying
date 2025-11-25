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

#############################################

filename = '../data/preproc_dataset_v4_amine.csv'
#filename = '../data/preproc_dataset_xgboost.csv'


df = pd.read_csv(filename, delimiter=',')

#############################################
#############################################
#############################################

ITER_NUM = 10

R2_LIST = []

for i in range(ITER_NUM):
    
    print('Iteration:',i)
    
    ###  Split dataset  ###
    
    X = df.drop("price", axis=1)
    
    y = df["price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    
    #############################################
    
    model = XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8
    )
    
    model.fit(X_train, y_train)
    
    #############################################
    
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    
    print(f"\nr2: {r2:.02f}")
    
    R2_LIST.append(r2)
    
    ### End of cycle ###
    
    
#############################################
#############################################

average_r2 = pd.Series(R2_LIST).mean()

print(f"\nMean R2: {average_r2:.02f}")

print('\nTask completed!')

#############################################
