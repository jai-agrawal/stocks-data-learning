import pandas as pd
import numpy as np
from datetime import datetime
from nsepy import get_history as gh
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler

# CONSTANTS

N_FEATURES = 1 # number of features
N_STEPS = 60 # number of time-steps
TRAIN_SPLIT = 0.8 # portion of data to be trained

# DATA PREPROCESSING

# Loading Data
data = gh(symbol='BHARTIARTL',start=datetime(2004,1,1),end=datetime(2021,4,13))
data = data[['Close']]
final_data = data.values
train_data = final_data[0:int(len(final_data) * TRAIN_SPLIT),:]
test_data = final_data[int(len(final_data) * TRAIN_SPLIT):,:]

# Scaling Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(final_data)

# Train-Test Split
X_train, y_train = [], []
for i in range(N_STEPS, len(train_data)):
    X_train.append(scaled_data[i - N_STEPS:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], N_FEATURES))

inputs = final_data[len(final_data) - len(test_data) - N_STEPS:]
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(N_STEPS, inputs.shape[0]):
    X_test.append(inputs[i-N_STEPS:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], N_FEATURES))

# BUILDING THE MODEL

# Intialisation and Adding of Layers
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compiling Model:
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
model.save("saved_model.h5")

# Saving Data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

test_data = data[int(len(data) * TRAIN_SPLIT):]
test_data['Predictions'] = predictions
test_data.to_csv('final_data.csv')