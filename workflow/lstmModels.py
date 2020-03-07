from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

def createCompileModelFischer(input_shape, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(25, input_shape=input_shape))
    model.add(Dropout(0.16))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss=loss, optimizer=optimizer)
    return model

def createCompileModelMurtaza(input_shape, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(16, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(1, activation="linear", kernel_initializer="uniform"))
    model.compile(optimizer=optimizer,loss=loss)
    return model

def createCompileModelRaval(input_shape, loss='mse', optimizer='adam'):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Activation('linear'))
    model.compile(loss=loss, optimizer=optimizer)
    return model