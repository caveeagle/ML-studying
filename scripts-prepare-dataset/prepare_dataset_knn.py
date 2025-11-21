import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder

################################################

filename = '../data/original_dataset_v4_caveeagle.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

most_valuable_cols = ['price','rooms','area','property_type',
                      'has_equipped_kitchen','garage','has_garden',
                      'cadastral_income', 'has_swimming_pool','build_year']

df = df[most_valuable_cols]

##################################################################

df['build_year_missing'] = df['build_year'].isna().astype(int)

df['build_year'] = df['build_year'].fillna(df['build_year'].median())

df['cadastral_income_missing'] = df['cadastral_income'].isna().astype(int)

df['cadastral_income'] = df['cadastral_income'].fillna(df['cadastral_income'].median())

df['has_equipped_kitchen'] = df['has_equipped_kitchen'].replace(
    {'Super equipped': 'Fully equipped'}
)

##################################################################

cols_to_encode = ['property_type','has_equipped_kitchen','garage','has_garden','has_swimming_pool' ]

X_cat = df[cols_to_encode]

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

encoded_array = encoder.fit_transform(X_cat)

encoded_cols = encoder.get_feature_names_out(cols_to_encode)

encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

df = df.drop(columns=cols_to_encode)

df = pd.concat([df, encoded_df], axis=1)

##################################################################

df = df.dropna(subset=['rooms'])

df['area'] = df['area'].fillna(df['area'].median())

##################################################################

col_for_scale = ['area','rooms','cadastral_income','build_year']

scaler = StandardScaler()

df[col_for_scale] = scaler.fit_transform(df[col_for_scale])

##################################################################

if (0):
    print(df.dtypes)

if (1):

    filename = '../data/preproc_dataset_knn.csv'

    df.to_csv(filename, sep=',', index=False)


##################################################################

print('The job have done')
