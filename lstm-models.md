# LSTM models

```python
def model_fn(params):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.CuDNNLSTM(params["lstm_size"], input_shape=(30, 2)))
    model.add(tf.keras.layers.Dropout(params["dropout"]))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer=tf.keras.optimizers.Adam(params["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_acc", patience=5,
                                                  restore_best_weights=True)]
    history = model.fit_generator(train_generator, validation_data=val_generator,
                                  callbacks=callbacks, epochs=100, verbose=0).history
    return (history, model)
```
From: [Using an LSTM-based model to predict stock returns] (https://jackdry.com/using-an-lstm-based-model-to-predict-stock-returns)

2. 
```python
model = Sequential()
model.add(LSTM(128, input_shape=(layers[1],
layers[0]), return_sequences=True))
model.add(LSTM(64, input_shape=(layers[1],
layers[0]), return_sequences=False))
model.add(Dense(16,init='uniform',activation='relu'))
model.add(Dense(1,init='uniform',activation='linear'))
```
From: [Murtaza Roondiwala et al., 2015, Predicting Stock Prices Using LSTM](https://pdfs.semanticscholar.org/3f5a/cb5ce4ad79f08024979149767da6d35992ba.pdf)

3.
```python
model = Sequential([
    Dense(50, activation='elu', input_shape=input_shape),
    Dropout(0.5),
    Dense(50, activation='elu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
```
From: [Daytrader collab](https://colab.research.google.com/drive/1W6TprjcxOdXsNwswkpm_XX2U_xld9_zZ)

1.
The model is a Sequential model using two LSTM layers
and one Dense layer. The LSTM layers use 50 units each
and hyperbolic tangent as activation function. The model is
currently not using any Dropout and thus runs greater risk of
overfitting, but does perform better than with Dropout on our
test data.  

**Unfortunately hard to reconstruct with too little information**
```python
model = Sequential()
model.add((LSTM(50, input_shape=input_shape, return_sequences=True)
model.add((LSTM(50, input_shape=input_shape, return_sequences=True)

```
From: [Impact of Time Steps on Stock Market Prediction with LSTM](https://kth.diva-portal.org/smash/get/diva2:1361305/FULLTEXT01.pdf)