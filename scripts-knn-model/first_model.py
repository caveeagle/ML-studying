"""
Random Forest model
"""
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import time
#############################################

filename = '../data/preproc_dataset_knn.csv'

df = pd.read_csv(filename, delimiter=',')
#############################################

X = df.drop("price", axis=1)

y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=None)

#############################################

###  PARAMETERS  ###

NEIGHBORS_NUMBER = 4

#############################################
start_time = time.perf_counter()
################################

knn = KNeighborsRegressor(n_neighbors=NEIGHBORS_NUMBER)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

r2 = r2_score(y_test, y_pred)

print("R2:", r2)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

print('\nTask completed!')

#############################################
