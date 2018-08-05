#Using Mnist Dataset
import keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
#Data Reshape
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print('Train samples: {}'.format(x_train.shape[0]))
print('Test samples: {}'.format(x_test.shape[0]))
(x_train.shape, x_test.shape)

#Training Parameters

input_dim = x_train.shape[1]
encoding_dim = 32
num_epochs = 50
h1 = 1000
h2 = 500
h3 = 250
h4 = 30


autoencoder = Sequential()
#Encoder 4 hidden layers
autoencoder.add(Dense(h1, input_shape=(input_dim,),activation='relu'))
#autoencoder.add(Dropout(0.25))
autoencoder.add(Dense(h2, activation='relu'))
autoencoder.add(Dense(h3, activation='relu'))
autoencoder.add(Dense(h4, activation='relu'))
#Decoder 4 hidden layers
autoencoder.add(Dense(h4, input_shape=(input_dim,),activation='relu'))
autoencoder.add(Dense(h3, activation='relu'))
autoencoder.add(Dense(h2, activation='relu'))
autoencoder.add(Dense(h1, activation='relu'))
#Output layer
autoencoder.add(Dense(input_dim, activation='sigmoid'))
autoencoder.summary()

autoencoder.compile(optimizer='RMSProp', loss='binary_crossentropy', metrics=["accuracy"])
#model.compile(optimizer, loss, metrics=["accuracy"])
#autoencoder.fit(x_train,x_train, epochs=50, batch_size=256,validation_data=(x_test,x_test))


autoencoder.fit(x_train,x_train, batch_size=128,epochs=num_epochs, verbose=1)

#Evaluate Model (autoencoder) Performance
score = autoencoder.evaluate(x_test,x_test,verbose=1)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
