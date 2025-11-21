
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
`n_estimators=800`
`max_depth=5`
`learning_rate=0.05`
`subsample=0.8`

The mean accuracy metrics are:
R2: **0.804**

The model was tested for overfitting. It is stable and shows no signs of overfitting.

### KNN model
**KNN model not suitable for this task, R2 < 0.3**

