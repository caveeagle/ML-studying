import pandas as pd
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

################################################

filename = '../data/original_dataset_v4.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

df.drop(columns=['url', 'locality', 'low_energy','kitchen_surface','diningrooms','maintenance_cost',
                'availability','g_score','p_score','terrain_width_roadside',
                'frontage_width','parking_places_indoor','co2',
                'flooding_area_type','rain_water_tank'],
        inplace=True)

df.drop(columns=['property_id'], inplace=True)  # Delete index

##################################################################

categorical_cols = [col for col in df.columns if df[col].nunique() < 100]

numerical_cols = [col for col in df.columns if col not in categorical_cols]

for col in categorical_cols:
    df[col] = df[col].fillna('MISSING')  
    df[col] = df[col].astype(str)        
    df[col] = LabelEncoder().fit_transform(df[col])

for col in numerical_cols:
    nan_ratio = df[col].isna().mean()
    if nan_ratio > 0.05:
        df[col + '_missing'] = df[col].isna().astype(int)  # 1 = NaN, 0 = not NaN
    
    df[col] = df[col].fillna(df[col].median())

##################################################################

corr_matrix = df.corr()

price_corr = corr_matrix['price'].sort_values(ascending=False)

if(0):
    pd.set_option('display.max_rows', None)
    print(price_corr)

##################################################################

if(0):
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

##################################################################

if (1):
    
    print('Numerical cols:')
    print(numerical_cols)
    print('\n\n')
    
    print(df.dtypes)

##################################################################

if (0):

    filename = '../data/preproc_dataset_v4_cave.csv'

    df.to_csv(filename, sep=',', index=False)

##################################################################

print('The job have done')
