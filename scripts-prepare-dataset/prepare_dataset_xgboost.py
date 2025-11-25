import pandas as pd

from sklearn.preprocessing import LabelEncoder

################################################

filename = '../data/original_dataset_v4_caveeagle.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

df.drop(columns=['property_id'], inplace=True)  # Delete index

##################################################################

### Drop near-constant features

if(0):
    
    for col in df.columns:
        percent_nan = df[col].isna().mean() * 100
        if percent_nan > 80:
            print(f"{col}: {percent_nan:.1f}% ")

if(1):        
    near_constant_features = [ 'low_energy', 
                                'heat_pump',
                                'air_conditioning', 
                                'alarm', 
                                'garden_surface', 
                                'rain_water_tank', 
                                'planning_permission_granted', 
                                'surroundings_protected', 
                                'security_door', 
                                'frontage_width', 
                                'terrain_width_roadside', 
                                'g_score', 
                                'p_score']
            
    df.drop(columns=near_constant_features, inplace=True)

##################################################################

if(0):
    print(df.dtypes)
    print('\ntype=object:')
    print( df.select_dtypes(include='object').columns.tolist() )

if(0):
    df2 = df.select_dtypes(include='object')
    filename = 'tmp.csv'
    df2.to_csv(filename, sep=',', index=False)


##################################################################


if (1):

    filename = '../data/preproc_dataset_xgboost.csv'

    df.to_csv(filename, sep=',', index=False)


##################################################################

print('The job have done')
