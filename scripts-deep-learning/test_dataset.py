"""
Shallow neural network model
FNN (feedforward neural networks)
"""
import pandas as pd

#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

print(df.shape)
print(df.dtypes)

'''


price                                      float64

area                                       float64
build_year                                 float64
primary_energy_consumption                 float64

postal_code                                  int64
cadastral_income                           float64




'''

print('\nTask completed!')

#############################################
