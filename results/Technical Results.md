
## Results


### 1. Results with the dataset **v3** for Random forest

The Random Forest model was trained on the dataset `preproc_dataset_v3.csv`

For the model accuracy metric, I used the **R2** criterion (coefficient of determination).

The tuned model parameters are:
`n_estimators=300`
`max_depth=None`
`min_samples_leaf=2`

The mean accuracy metrics are:
RMSE: 176946 Euro 
MAE: 79775 
R2: **0.683**

### 2. Results with the dataset  preproc_dataset_v4_amine.csv   for Random forest

The Random Forest model was trained on the dataset `preproc_dataset_v4_amine.csv`

For the model accuracy metric, I used the **R2** criterion (coefficient of determination).

The tuned model parameters are:
`n_estimators=300`
`max_depth=None`
`min_samples_leaf=1`

The mean accuracy metrics are:
RMSE: 104538 Euro 
MAE: 55889 
R2: **0.784**

### 3. Results for classic Gradient Boosting

The classic Gradient Boosting model was trained on the dataset `preproc_dataset_v4_amine.csv`

For the model accuracy metric, I used the R2 criterion (coefficient of determination).

The tuned model parameters are:
`n_estimators=1200`
`max_depth=5`
`learning_rate=0.05`
`subsample=0.9`

The mean accuracy metrics are:
R2: **0.810**

The model was tested for overfitting. It is stable and shows no signs of overfitting.

### Other models
KNN model not suitable for this task, R2 < 0.5
CatBoost model gives results worse than classical boosting

### 4. Results for XGBoost
The XGBoost model was tested for two datasets, Amine's dataset was chosen as the best.
The tuned model parameters are:
`n_estimators=800`
`max_depth=6`
`learning_rate=0.05`
`subsample=0.8`
The other parameters also have been tested, but their default values proved to be the best. The following additional parameters were tested:
`min_child_weight, gamma, reg_alpha, reg_lambda`
The mean accuracy metrics are:
R2: **0.800**

### 5. Results for FNN (Neural network)
R2: **0.779**

#### Parameters:
Layers - **4** hidden layers:

`x = Dense(128, activation="relu")(input_layer)`
`x = Dense(64, activation="relu")(x)`
`x = Dense(32, activation="relu")(x)`
`output = Dense(1)(x)`

Tuning parametrs:

`learning_rate = **0.0003**`
`batch_size = **16**`

The best epoch: 14 (with Early stopping)
Best val_loss: 0.2625
Best train_loss: 0.1182

