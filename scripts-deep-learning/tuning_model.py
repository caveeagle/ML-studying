"""
Shallow neural network model - FNN (feedforward neural networks)
"""
import pandas as pd
import numpy as np

import time

start_time = time.perf_counter()

import os
import logging

LOG_FILE = './log_FNN_v1.txt'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras

from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#############################################

dataset_filename = '../data/preproc_dataset_v4_amine.csv'

#############################################

# create logger

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# handler for console
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)

# handler for file
file_handler = logging.FileHandler(LOG_FILE)
logger.addHandler(file_handler)

##############################################
##############################################
##############################################
##############################################

#############################################
#                                           #
#           Parameters                      #
#                                           #
#############################################

LEARNING_RATE = 0.001

DEPTH = 2    # Number of hidden layers (2, 3)

BATCH_SIZE = 64

##############################################
##############################################
##############################################
##############################################

R2_LIST = []

ITER_NUM = 3

for i_num in range(ITER_NUM):
    
    start_cycle_timer = time.perf_counter()
    
    logger.info(f'\nITERATION: {i_num}\n')

    logger.info('\nTRAIN WITH PARAMS:')
    
    grid_str = f'LEARNING_RATE = {LEARNING_RATE}\n'
    grid_str += f'DEPTH = {DEPTH}\n'
    grid_str += f'BATCH_SIZE = {BATCH_SIZE}\n'
    
    logger.info(grid_str)
    
    ##############################################
    
    df = pd.read_csv(dataset_filename, delimiter=',')

    ##############################################
    
    # Target variable
    y = df["price"].values.reshape(-1, 1)  # reshape - for StandardScaler
    
    # Remove unnecessary columns
    # Functional dependency ! cadastral_income - with postal_code
    df = df.drop(columns=["price", "cadastral_income"])  
    
    # Embedding column
    postal = df["postal_code"].astype(int).values
    
    # Numeric features
    num_features = ["area", "build_year", "primary_energy_consumption"]
    X_num = df[num_features]
    
    # Other remaining features
    other_cols = [c for c in df.columns if c not in num_features + ["postal_code"]]
    X_other = df[other_cols]
    
    ############################################
    
    ### ===== Train / Validation / Test Split ===== ###
    
    # train (70%) and temp (30%)
    X_train_num, X_temp_num, \
    X_train_postal, X_temp_postal, \
    X_train_other, X_temp_other, \
    y_train, y_temp = train_test_split(
        X_num, postal, X_other, y, test_size=0.3, random_state=None
    )
    
    # from temp (30%) val (15%) and test (15%)
    X_val_num, X_test_num, \
    X_val_postal, X_test_postal, \
    X_val_other, X_test_other, \
    y_val, y_test = train_test_split(
        X_temp_num, X_temp_postal, X_temp_other, y_temp,
        test_size=0.5, random_state=None
    )
    
    ############################################
    
    ###  Scaling ###
    
    num_scaler = StandardScaler()
    
    X_train_num = num_scaler.fit_transform(X_train_num)
    X_val_num = num_scaler.transform(X_val_num)
    X_test_num = num_scaler.transform(X_test_num)
    
    y_scaler = StandardScaler()
    
    y_train = y_scaler.fit_transform(y_train)
    y_val = y_scaler.transform(y_val)
    y_test = y_scaler.transform(y_test)
    
    #############################################
    
    ###  Encode postal_code to continuous range 0..n_categories-1
    unique_postals = np.sort(np.unique(X_train_postal))  # Only from train !
    postal_to_index = {v: i for i, v in enumerate(unique_postals)}
    
    X_train_postal_codes = np.array([postal_to_index[x] for x in X_train_postal])
    # Test set: unknown postal_codes mapped to -1
    X_test_postal_codes = np.array([postal_to_index.get(x, -1) for x in X_test_postal])
    X_val_postal_codes  = np.array([postal_to_index.get(x, -1) for x in X_val_postal])
    
    n_categories = len(unique_postals)
    
    ### Convert other features to numpy ###
    X_train_other = X_train_other.to_numpy()
    X_test_other = X_test_other.to_numpy()
    X_val_other = X_val_other.to_numpy()
    
    X_train_other = X_train_other.astype("float32")
    X_test_other = X_test_other.astype("float32")
    X_val_other = X_val_other.astype("float32")
    
    #############################################
    
    ### There is no R2 metric in TensorFlow! ###
    
    import tensorflow.keras.backend as K
    
    def r2_metric(y_true, y_pred):
        ss_res = K.sum(K.square(y_true - y_pred))
        ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - ss_res / (ss_tot + K.epsilon())
    
    
    #############################################
    #############################################
    #############################################
    
    #############################################
    #                                           #
    #        Build Keras model                  #
    #                                           #
    #############################################
    
    emb_dim = 16  # size of embedding
    
    postal_input = Input(shape=(1, ), name="postal_input")
    # Add +1 to input_dim for unknown category (-1 mapped to 0)
    postal_emb = Embedding(input_dim=n_categories + 1,
                           output_dim=emb_dim)(postal_input)
    postal_emb = Flatten()(postal_emb)
    
    num_input = Input(shape=(X_train_num.shape[1], ), name="num_input")
    other_input = Input(shape=(X_train_other.shape[1], ), name="other_input")
    
    #############################################
    
    early_stop = EarlyStopping(
        monitor='val_loss',        # monitor validation loss
        patience=5,                # stop if no improvement for 3 consecutive epochs
        min_delta=1e-3,            # minimum change to qualify as an improvement
        restore_best_weights=True  # restore the model weights from the best epoch
    )
    
    #############################################
    
    input_layer = Concatenate()([postal_emb, num_input, other_input])
    
    if (DEPTH == 3):
    
        x = Dense(128, activation="relu")(input_layer)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        output = Dense(1)(x)
    
    elif (DEPTH == 2):
    
        x = Dense(64, activation="relu")(input_layer)
        x = Dense(32, activation="relu")(x)
        output = Dense(1)(x)
    
    else:
        raise ValueError("Invalid DEPTH value")
    
    model = Model(inputs=[postal_input, num_input, other_input], outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='mse',
        metrics=[r2_metric]
    )
    
    #############################################
    #                                           #
    #        Train the model                    #
    #                                           #
    #############################################
    
    history = model.fit([X_train_postal_codes, X_train_num, X_train_other],
                          y_train,
                          epochs=30,  # With Early Stopping !
                          batch_size=BATCH_SIZE,
                          callbacks=[early_stop],
                          validation_data=([X_val_postal_codes, X_val_num, X_val_other], y_val),
                          verbose=1)
    
    #############################################
    #                                           #
    #        Evaluate the model                 #
    #                                           #
    #############################################
    
    y_pred_scaled = model.predict([X_test_postal_codes, X_test_num, X_test_other])
    
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_original = y_scaler.inverse_transform(y_test)
    
    r2 = r2_score(y_test_original, y_pred)
    logger.info(f"R2 on test set:{r2:.3f}")
    
    R2_LIST.append(r2)
    
    if(1):
        best_epoch = np.argmin(history.history['val_loss']) + 1
        logger.info(f"The best epoch: {best_epoch}")
    
    
    if(0):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(8, 5))
        ###plt.plot(history.history['loss'], label='Train loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Val.loss (MSE)')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    ######################################
    
    end_cycle_timer = time.perf_counter()
    elapsed_cycle_time = end_cycle_timer - start_cycle_timer
    logger.info(f"\nCycle timer: {elapsed_cycle_time:.0f} sec\n")

        
    ### END OF HYPERCYCLE ###
        
#############################################
#############################################
#############################################

average_r2 = pd.Series(R2_LIST).mean()

logger.info(f"\n\nMean R2 in cycles: {average_r2:.02f}")

    
#############################################

end_time = time.perf_counter()
elapsed_time = end_time - start_time
logger.info(f"\nOverall timer: {elapsed_time:.0f} sec\n")

#############################################

logger.info('\nTask completed!')

#############################################
