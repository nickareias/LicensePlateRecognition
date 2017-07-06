# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:41:34 2016

@author: Asus
"""
import numpy
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
import pickle

seed = 7
numpy.random.seed(seed)

#function to get licence plate data
#geared specifically for keras
def get_lp_all(path):
    data = []  #refresh list

    # read data to verify correctness
    infile = open(path, 'rb')
    while 1:
        try:
            data.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()

    #put 4/5 of the data into training
    #and 1/5 into testing
    l = int(len(data) * (4/5))
    
    random.shuffle(data)

    training_data = data[:l]
    testing_data = data[l:]
    
    return (training_data, testing_data)

#MNIST
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

training_data, testing_data = get_lp_all('evenly_distributed_data.pkl')

X_train, y_train = zip(*training_data)
X_test, y_test = zip(*testing_data)

X_train = [numpy.reshape(x, (28,28)) for x in X_train]
X_test = [numpy.reshape(x, (28,28)) for x in X_test]

#shows images of training data

plt.subplot(221)
plt.imshow(numpy.reshape(X_train[0], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(numpy.reshape(X_train[1], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(numpy.reshape(X_train[2], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(numpy.reshape(X_train[3], (28,28)), cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#X_train.shape[0] is how many images
#X_train.shape[1] is how many rows
#X_train.shape[2] is how many columns

#count number of pixels by multiplying rows/columns
num_pixels = len(X_train[0])

num_classes = len(y_train[0])

X_train = numpy.reshape(X_train, (len(X_train),1,28,28))
y_train = numpy.array(y_train)
X_test = numpy.reshape(X_test, (len(X_test),1,28,28))
y_test = numpy.array(y_test)

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    #model.add(Dense(100, input_dim=num_pixels, init='normal', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1,28,28)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, border_mode='same'))
    model.add(Flatten())
    model.add(Dense(100, init='normal', activation='relu'))
    model.add(Dense(num_classes, init='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
 
# build the model
print("creating model")
model = baseline_model()
# Fit the model
print("training model...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=20, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))