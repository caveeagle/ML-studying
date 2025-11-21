"""
Cat boosting model
"""
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

import time
#############################################

filename = '../data/preproc_dataset_v4_amine.csv'

df = pd.read_csv(filename, delimiter=',')

#############################################

X = df.drop("price", axis=1)

y = df["price"]

cat_features = ["rooms", "bathrooms", "toilets", "facades_number", "number_floors", "has_terrace", "has_garden", "has_garage", "elevator", "is_furnished", "leased", "running_water", "primary_energy_consumption", "postal_code", "has_swimming_pool", "property_type_house", "property_type_other", "locality_antwerp", "locality_braine-l-alleud", "locality_brussels", "locality_gent", "locality_laken", "locality_liege", "locality_lier", "locality_mons", "locality_mouscron", "locality_namur", "locality_nivelles", "locality_oostende", "locality_other", "locality_pont-a-celles", "locality_roeselare", "locality_seraing", "locality_tournai", "locality_tubize", "locality_turnhout", "locality_wavre", "property_subtype_duplex", "property_subtype_other", "property_subtype_residence", "property_subtype_studio", "property_subtype_villa", "has_equipped_kitchen_Not equipped", "has_equipped_kitchen_Partially equipped", "has_equipped_kitchen_Super equipped"]

#############################################

float_cat_cols = [col for col in cat_features if X[col].dtype == 'float64']

for col in float_cat_cols:
    X[col] = X[col].fillna(-1).astype(int)

#############################################
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  
#############################################
start_time = time.perf_counter()
################################

model = CatBoostRegressor(
    iterations=3000,           # maximum number of trees
    learning_rate=0.05,        # smaller for higher precision
    depth=6,                   # depth of the trees
    l2_leaf_reg=3,             # regularization
    bagging_temperature=1.5,   # stochasticity
    loss_function='RMSE',      # regression
    random_seed=None,
    early_stopping_rounds=50,  # stop training if no improvement
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test)
)

##############################
end_time = time.perf_counter()
#############################################
elapsed_time = end_time - start_time
print(f"\nTimer: {elapsed_time:.0f} sec\n")
#############################################

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))

#############################################

print('\nTask completed!')

#############################################
