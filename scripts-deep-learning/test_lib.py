"""
Shallow neural network model
FNN (feedforward neural networks)
"""
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras

print('\nTask completed!')

#############################################
