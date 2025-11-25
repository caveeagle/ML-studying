import pandas as pd

from sklearn.preprocessing import LabelEncoder

################################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

### Drop near-constant features

if(0):
    
    for col in df.columns:
        freq = df[col].value_counts(normalize=True, dropna=False)
        top_value = freq.index[0] # Value 
        top_freq = freq.iloc[0] * 100  # and Freq in percents
    
        if top_freq > 80:
            print(f"{col}: {top_freq:.2f}%")

if(1):        
    near_constant_features = [ 'is_furnished','running_water','has_swimming_pool','property_type_other',
                                'has_equipped_kitchen_Not equipped','property_subtype_studio',
                                'property_subtype_duplex','leased']
            
    df.drop(columns=near_constant_features, inplace=True)
    
    cols_to_drop = []
    for col in df.columns:
        if not col.startswith('locality_'):
            continue
        
        counts = df[col].value_counts(dropna=False)
        top_value = counts.index[0]
        top_freq = counts.iloc[0] / len(df) * 100
    
        if top_freq > 90:
            cols_to_drop.append(col)

    df.drop(columns=cols_to_drop, inplace=True)

##################################################################

if (1):

    filename = '../data/preproc_dataset_xgboost.csv'

    df.to_csv(filename, sep=',', index=False)


##################################################################

print('The job have done')
