import pandas as pd

from sklearn.preprocessing import LabelEncoder

################################################

filename = '../data/original_dataset_v4.csv'

df = pd.read_csv(filename, delimiter=',')

##################################################################

df.drop(columns=['url',
                 'locality',
                 'water_softener',
                 'parking_places_outdoor',
                 'parking_places_indoor',
                 'garden_orientation',
                 'demarcated_flooding_area',
                 'opportunity_for_professional',
                 'wash_room',
                 'front_facade_orientation',
                 'diningrooms',
                 'maintenance_cost',
                 'vat',
                 'certification_gasoil_tank',
                 'terrace_orientation'],
        inplace=True)

##################################################################

cols_to_front = ['property_id', 'price']

other_cols = [c for c in df.columns if c not in cols_to_front]

new_order = cols_to_front + other_cols

df = df[new_order]

##################################################################

filename = '../data/clear_dataset_v4_caveeagle.csv'

df.to_csv(filename, sep=',', index=False)
    
##################################################################

print(df.dtypes)

##################################################################

print('The job have done')
