import pandas as pd

from sklearn.preprocessing import LabelEncoder

################################################

filename = '../data/original_dataset_v4_caveeagle.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

df.drop(columns=['property_id'], inplace=True)  # Delete index

df[df.select_dtypes('float64').columns] = df.select_dtypes('float64').fillna(-1)

df[df.select_dtypes('object').columns] = df.select_dtypes('object').fillna('None')

##################################################################

none_categorical_attributes = [
    "price",
    "area",
    "build_year",
    "kitchen_surface",
    "garden_surface",
    "terrace_surface",
    "land_surface",
    "primary_energy_consumption",
    "co2",
    "living_room_surface",
    "frontage_width",
    "terrain_width_roadside",
    "cadastral_income"
]

for col in df.columns:
    
    if col not in none_categorical_attributes:
        
        df[col] = LabelEncoder().fit_transform(df[col])

df = df.astype({col: 'int64' for col in df.select_dtypes('float64').columns})

##################################################################

if (0):
    print(df.dtypes)

if (1):

    filename = '../data/fitted_dataset_v4.csv'

    df.to_csv(filename, sep=',', index=False)


##################################################################

print('The job have done')
