"""
Shallow neural network model - FNN (feedforward neural networks)
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras

import time
#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

numeric_cols = ['price', 'area', 'build_year', 'garden_surface', 'terrace_surface', 'land_surface', 'primary_energy_consumption', 'living_room_surface', 'cadastral_income', 'postal_code']


preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols)
    ],
    remainder='passthrough'   
)

#############################################

print('\nTask completed!')

#############################################
