import pandas as pd

from sklearn.preprocessing import LabelEncoder

################################################

filename = '../data/original_dataset_v4.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

df.drop(df.columns[0], axis=1, inplace=True)  # Unnamed

df.drop(columns=['url', 'locality', 'sale_type', 'log_price_m2', 'cluster'],
        inplace=True)

df.drop(columns=['property_id'], inplace=True)  # Delete index

new_order = [   'price',
                'postal_code',
                'rooms',
                'area',
                'state',
                'has_open_fire',
                'property_type',
                'property_subtype',
                'facades_number',
                'is_furnished',
                'has_terrace',
                'has_garden',
                'has_swimming_pool',
                'has_equipped_kitchen',
                'is_luxurious']
                
df = df[new_order]

##################################################################

df['state'] = LabelEncoder().fit_transform(df['state'].fillna('None'))

df['property_type'] = LabelEncoder().fit_transform(df['property_type'])
df['property_subtype'] = LabelEncoder().fit_transform(df['property_subtype'])

df['has_swimming_pool'] = LabelEncoder().fit_transform(df['has_swimming_pool'])
df['has_equipped_kitchen'] = LabelEncoder().fit_transform(df['has_equipped_kitchen'])

df['is_luxurious'] = df['is_luxurious'].astype(int)

df.fillna(-1, inplace=True)

df = df.astype({col: 'int64' for col in df.select_dtypes('float64').columns})

##################################################################

if (0):
    
    print(df.dtypes)

##################################################################

if (1):

    filename = '../data/fitted_dataset.csv'

    df.to_csv(filename, sep=',', index=False)

##################################################################

print('The job have done')
