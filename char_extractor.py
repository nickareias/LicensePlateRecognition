# -*- coding: utf-8 -*-
"""
Created on Wed Feb 01 11:17:38 2017

@author: Jeffrey M. Paquette
"""

import matplotlib.image as mpimg
import pickle
import os
import scipy
#import common.filters as filters
#import common.rectfinder as rectfinder
#import common.traindata as traindata
#import numpy as np

import filters
import rectfinder

datasetpath = 'dataset2'

images = []         # list of images

# for eacsh file in dataset directory extract rectangles
for fname in os.listdir(datasetpath):
    print("loading " + fname)
    pic = mpimg.imread(datasetpath + "/" + fname)
    pic = filters.rgb2gray(pic)
    print("scaling image...")
    pic = scipy.misc.imresize(pic, (500,250))
    inverted_image = filters.invert2BnW(pic, 0.5)
    print("finding rectangles...")
    rectangles = rectfinder.Tracer(inverted_image, 0.02, 0.15, 0.1)
    #filters.draw(rectangles.highlight_rects())
    #filters.draw(inverted_image)
    rects = rectangles.extract_rects(pic)
    for r in rects:
        images.append(r)
    del pic
    del inverted_image
    del rectangles

    
training_data = []  # list of training data objects
classifiers = []    # list of training data classifiers

# get classifiers for each rectangle
# '-' will ignore rectangle
for i in images:
    filters.draw(i)
    classifier = str.capitalize(str(input("Enter class: ")))
    if (len(classifier) == 1 and (str.isalpha(classifier) or str.isdigit(classifier))):
        training_data.append(i)
        classifiers.append(classifier)
        
#zip training_data and classifiers together
final_data = [(x, y) for x,y in zip(training_data, classifiers)]

#write data to file in format
#final_data -> (image,classifier)
#d[0] -> image
#d[1] -> classifier
output = open('raw_data.pkl', 'w+b')
for d in final_data:
    pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)
output.close()