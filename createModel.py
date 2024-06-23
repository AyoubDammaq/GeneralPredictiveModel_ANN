from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.optimizers import Adam, SGD, Nadam, Adadelta, Adagrad, RMSprop, Adamax
from tensorflow.keras.optimizers import schedules
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers
import numpy as np
import time




def create_and_train_model(X_train_scaled, Y_train, n_input, n_epochs, batch_size):

    # Define the model
    model = Sequential()
    model.add(Dense(110, activation='relu', input_shape=(n_input,), kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(110, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(90, activation='relu', kernel_initializer='he_uniform', kernel_regularizer='l2'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(12, activation='linear', kernel_initializer='he_uniform', kernel_regularizer='l2'))


    opt1 = Adam(learning_rate=0.001)
    opt2 = Nadam(learning_rate=0.001)
    opt3 = Adadelta(learning_rate=0.1)
    opt4 = Adagrad(learning_rate=0.1)
    opt5 = RMSprop(learning_rate=0.1)
    opt6 = SGD(learning_rate=0.001)
    opt7 = Adamax(learning_rate=0.1)

    model.compile(loss='mean_squared_error', optimizer=opt7, metrics=['mse', 'mae'])

    model.summary()
    
    # Set up early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    t1 = time.perf_counter()
    history = model.fit(X_train_scaled, Y_train, epochs = n_epochs, validation_split = 0.3, batch_size = batch_size , verbose=0)
    t2 = time.perf_counter()

    return model, history, (t2-t1)/60



