#licence plate character file manager
import matplotlib.image as mpimg
import pickle
import numpy as np
import scipy

import filters


#turn an int into a vector of 0s and 1s
def vectorized_int(num):
    classifier = np.zeros([36])
    for i in range(36):
        if(i == num):
            classifier[i] = 1
                      
    return classifier

def vectorized_int_digits(num):
    classifier = np.zeros([10])
    for i in range(10):
        if(i == num):
            classifier[i] = 1
                      
    return classifier

def vector_to_int(vector):

    c = 0
    for i in vector:
        if(i == 1):
            return c
        c += 1
    
    return -1

def read_file(path):
    
    training_data = []  #refresh list
    
    # read data to verify correctness
    infile = open(path, 'rb')
    while 1:
        try:
            training_data.append(pickle.load(infile))
        except (EOFError, pickle.UnpicklingError):
            break
    infile.close()
    
    return training_data
        
#scales image, pass in 2d image
def scale_training_data(training_data):
    
    scaled_td = []
    for i in training_data:
        temp_image = scipy.misc.imresize(i, (28,28))
        scaled_td.append(temp_image)
        
    return scaled_td

#pass in 2d images, pre-scaled
def convert_to_bw(training_data):

    #black/white
    bw_td = []
    for i in training_data:
        temp_image = filters.invert2BnW(i, 0.5)
        bw_td.append(temp_image)
        
    return bw_td

#pass in 2d images, pre-scaled, bw-converted
def convert_to_1d(training_data):

    #convert to 1d
    oned_td = []
    for image in training_data:
        oned_td.append(np.ndarray.flatten(image).astype(float))
    
    return oned_td

#pass in 1d image, pre-scaled, bw-converted
def normalize_images(training_data):

    #normalize
    norm_td = []
    for i in training_data:
        temp_min = np.min(i)
        temp_max = np.max(i)
        
        norm_td.append([((x - temp_min) 
                        / (temp_max - temp_min)) 
                            for x in i])
        
    return norm_td

#pass in training_data and classifiers
def vectorize_data_and_zip(training_data, classifiers):
    #vectorize classifiers to include letters
    #vector length of 36
    images = []
    new_classifiers = []
    c = 0
    for i in training_data:
    
        if((ord(classifiers[c]) >= 48) and (ord(classifiers[c]) <= 57)):
            temp_num = ord(classifiers[c]) - 48
        elif((ord(classifiers[c]) >= 65) and (ord(classifiers[c]) <= 90)):
            temp_num = ord(classifiers[c]) - 55         
                          
        temp_class = vectorized_int(temp_num)
        new_classifiers.append(temp_class)
        images.append(i)

        c += 1
            
    #zip image / calssifier into one array
    return [(x, y) for x,y in zip(images, new_classifiers)]

def write_file(path, training_data):
    output = open(path, 'w+b')
    for d in training_data:
        pickle.dump(d, output, pickle.HIGHEST_PROTOCOL)
    output.close()  
  
#makes an even distribution by copying images of a certain type until 
#that type has as many elements as the type with the maximum ammount
def make_even_distribution(td):
    
    #check frequencies of each character
    frequencies = np.zeros([36])
    
    #create a list of 36 lists
    freq_images = [[] for i in range(36)]
    
    for i in td:
        temp_index = vector_to_int(i[1])
        frequencies[temp_index] += 1
                  
        freq_images[temp_index].append(i)
        
    
    
    max_freq = np.max(frequencies)
    
    for i in range(len(freq_images)):
        
        for j in range(int(max_freq - len(freq_images[i]))):
            if(len(freq_images[i]) != 0):
                freq_images[i].append(freq_images[i][j])
        
    even_training_data = []
    for i in freq_images:
        for j in i:
            even_training_data.append(j)
    
    return even_training_data
    
########################
## MAIN FUNCTIONALITY ##
########################
#reads from read_path, converts file accordingly, and write to write_path
def read_convert_write(read_path, write_path):
    
    #read training data from file
    training_data = read_file(read_path)
    
    #unzip training data
    images, classifiers = zip(*training_data)
    
    images2 = scale_training_data(images)
    images3 = convert_to_bw(images2)
    images4 = convert_to_1d(images3)
    images5 = normalize_images(images4)
    final_data = vectorize_data_and_zip(images5, classifiers)
    
    write_file(write_path, final_data)
    

from PIL import Image

#Write lp character images to a folder
#Store classifiers in a csv as the same index as the image they are connected with
data = read_file("data.pkl")

for (c,d) in enumerate(data):
    if(len(d.classifier) > 1):
        data[c].classifier = vector_to_int(d.classifier)
    else:
        if((ord(d.classifier) >= 48) and (ord(d.classifier) <= 57)):
            data[c].classifier = ord(d.classifier) - 48
        elif((ord(d.classifier) >= 65) and (ord(d.classifier) <= 90)):
            data[c].classifier = ord(d.classifier) - 55
        
    c += 1

classifiers = ""
c = 0
for d in data:
    
    #convert numpy array to PIL Image object
    image = Image.fromarray(d.image.astype(np.uint8))
    
    #save image to file
    image.save('images/lp'+str(c)+'.png')
    
    #add class to classifier string for saving to file
    classifiers += (str(d.classifier) + '\n')
    c += 1
    
with open('images/classifiers.txt', 'w+') as f:
    f.write(classifiers)
