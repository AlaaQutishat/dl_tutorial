from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import *
from keras.losses import *
from keras.optimizers import *
import keras
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train)  # 60000*28*28
# print(y_train)  # 60000
# print(x_test)  # 10000*28*28
# print(y_test)  # 10000

x_train = x_train.reshape(60000, 784).astype('float32')
x_test = x_test.reshape(10000, 784).astype('float32')
x_train = x_train / 255
x_test = x_test / 255
# print(x_train)
# print(x_test)
# print(y_train[55])

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=784))
model.add(Dropout(0.2))
model.add(Dense(512, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='mse', optimizer=SGD(), metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=1)
print(score)