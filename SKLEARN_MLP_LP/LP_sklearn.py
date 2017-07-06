# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 12:41:34 2016

@author: Asus
"""
import numpy as np
import scipy
import random
#import multi layer perceptron classifier from scikit learn
from sklearn.neural_network import MLPClassifier

import pickle
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

seed = 7
np.random.seed(seed)

#turn an int into a vector of 0s and 1s
def vectorized_int(num):
    classifier = np.zeros([36])
    for i in range(36):
        if(i == num):
            classifier[i] = 1
                      
    return classifier

#function to convert a vector to an integer
def v2i(vector):
    for (c,i) in enumerate(vector):
        if(i == 1):
            return c
    return -1

#function to get licence plate data
#geared specifically for keras
def get_lp_all(path):
    
    data = []  #refresh list

    #get classifiers file
    classifiers = pd.read_csv(path + "classifiers.txt", names = ['classes'], squeeze = True)
    
    for (i,c) in enumerate(classifiers):
        
        #read img from file
        temp_img = Image.open(str(path)+"lp"+str(i)+".png",mode='r')
        
        #convert image to 2d array
        temp_img = np.asarray(temp_img)
        
        #scale image to 28x28
        temp_img = scipy.misc.imresize(temp_img, (28,28))
        
                
            
        #convert to 1d
        temp_img = np.reshape(temp_img, (28*28))
        
        #normalize image data
        temp_min = np.min(temp_img)
        temp_max = np.max(temp_img)
        temp_img = [((x - temp_min) / (temp_max - temp_min)) for x in temp_img]
        
        #convert classifier into a vector
        class_vector = np.zeros([36])
        for j in range(36):
            if(j == classifiers[i]):
                class_vector[j] = 1


        data.append([temp_img,class_vector])
      
    ## COPY CHARACTERS TO GET EVEN DISTRIBUTION ##
    
    #check frequencies of each character
    frequencies = np.zeros([36])
    
    #create a list of 36 lists
    freq_images = [[] for i in range(36)]
    
    #loop through data and create update list of frequencies
    for i in data:
        temp_index = v2i(i[1])
        frequencies[temp_index] += 1
    
        #add image to list with index equal to its classifier
        freq_images[temp_index].append(i)
    
    #find character with the highest frequency
    max_freq = np.max(frequencies)
    
    #duplicate images until there is even representaton of each image
    for i in range(len(freq_images)):
    
        for j in range(int(max_freq - len(freq_images[i]))):
            if(len(freq_images[i]) != 0):
                freq_images[i].append(freq_images[i][j])
                
    #copy all images into a new list
    even_data = []
    for i in freq_images:
        for j in i:
            even_data.append(j)
            
    #images, classifiers = zip(*even_data)

    #put 4/5 of the data into training
    #and 1/5 into testing
    l = int(len(even_data) * (4/5))
    
    random.shuffle(even_data)

    training_data = even_data[:l]
    testing_data = even_data[l:]
    
    return (training_data, testing_data)


training_data, testing_data = get_lp_all('images/')

X_train, y_train = zip(*training_data)
X_test, y_test = zip(*testing_data)

#shows images of training data

plt.subplot(221)
plt.imshow(np.reshape(X_train[0], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(np.reshape(X_train[1], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(np.reshape(X_train[2], (28,28)), cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(np.reshape(X_train[3], (28,28)), cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#X_train.shape[0] is how many images
#X_train.shape[1] is how many rows
#X_train.shape[2] is how many columns

#count number of pixels by multiplying rows/columns
num_pixels = len(X_train[0])

num_classes = len(y_train[0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)




#number of hidden neurons in the middle layer
hidden_neurons = 750

#maximum number of times to loop through the training data
epochs = 500
 
# build the model
print("creating model")

#create the model, most of the default parameters are good for our problem
#we'll just set the maximum iterations and a random state for repeatability
model = MLPClassifier(hidden_layer_sizes = (hidden_neurons,),
                      max_iter=epochs, random_state = 1, tol = 0,
                      verbose = True)

# Fit the model
print("training model...")
#start training the model
model.fit(X_train, y_train)

#predict test set
predictions = model.predict(X_test)

#check percetage correct and store incorrect images in a list to display
num_correct = 0
num_incorrect = 0
max_incorrect = 4
incorrect_images = []
incorrect_classifiers = []
for (i,p) in enumerate(predictions):
    if(v2i(p) == v2i(y_test[i])):
        num_correct += 1
    elif(num_incorrect < max_incorrect):    
        num_incorrect += 1
        incorrect_images.append(np.reshape(X_test[i],(28,28)))
        incorrect_classifiers.append("Predicted: "+str(v2i(p)) + " / Actual: " + str(v2i(y_test[i])))
                                
        
    
#print accuracy of classification on test set    
accuracy = num_correct/len(predictions)
print("Accuracy: %.4f%%\n" % (accuracy*100))    