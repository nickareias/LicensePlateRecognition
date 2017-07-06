# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 11:04:46 2016

@author: Asus
"""

import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import time
import scipy.misc

def draw(img):
    plt.imshow(img, 'gray')
    plt.show()
    plt.clf()
    return

def rgb2gray(rgb):
    return numpy.dot(rgb[:], [0.299, 0.587, 0.114])

def rgb2grayValue(rgb):
    return [[max(x) for x in y] for y in rgb]

def threshold(img, threshold, t_range):
    img_copy = numpy.array(img)
    
    #img_copy = [(255 - x) for x in img_copy]
    
    
    # find min value
    max_value = numpy.max(img)
    min_value = numpy.min(img)
    rng = max_value - min_value
    max_threshold = min_value + (rng * (threshold + t_range))
    min_threshold = min_value + (rng * (threshold - t_range))
    #mid_range = (max_value + min_value) / 2
    for y in range(0, len(img_copy)):
        for x in range(0, len(img_copy[0])):
            if (img_copy[y][x] >= max_threshold):
                img_copy[y][x] = 0
            elif(img_copy[y][x] <= min_threshold):
                img_copy[y][x] = 255
                        
    return img_copy

def invert2BnW(img, threshold):
    """Takes an image, inverts it and clamps each pixel to either 0 or 255.
    
    Keyword arguments:
    img -- image
    threshold -- rounding threshold (percentage above min)
    """
    img_copy = numpy.array(img)
    
    # find min value
    max_value = numpy.max(img)
    min_value = numpy.min(img)
    rng = max_value - min_value
    rnd_threshold = min_value + (rng * threshold)
    #mid_range = (max_value + min_value) / 2
    for y in range(0, len(img_copy)):
        for x in range(0, len(img_copy[0])):
            if (img_copy[y][x] <= rnd_threshold):
                img_copy[y][x] = 255
            else:
                img_copy[y][x] = 0
    return img_copy
                
def darken(img, value):
    """Darkens a grayscale image by dividing each element by value."""
    
    if (value <= 0):
        return
        
    img_copy = numpy.array(img)
    
    for y in range(0, len(img_copy) - 1):
        for x in range(0, len(img_copy[0]) - 1):
            img_copy[y][x] = img_copy[y][x] / value
    return img_copy
#pic = mpimg.imread('..\\..\\..\\LicencePlateImages\\full_car_1.jpg')
#pic = mpimg.imread('..\\..\\..\\LicencePlateImages\\full_car_2.jpg')
#pic = mpimg.imread('..\\..\\..\\LicencePlateImages\\licence_plate1.jpg')
#pic = mpimg.imread('..\\..\\..\\LicencePlateImages\\licence_plate2.jpg')
#pic = mpimg.imread('..\\..\\..\\LicencePlateImages\\licence_plate3.jpg')
#pic = mpimg.imread('smile.jpg')
#pic = mpimg.imread('som_test.jpg')
#pic = mpimg.imread('som_test2.jpg')
#pic = mpimg.imread('som_test3.jpg')
#pic = mpimg.imread('handwritten.jpg')
#pic = mpimg.imread('licence_plate.jpg')

#pic = mpimg.imread('cropped_canny_licenceplate.jpg')
#gray_pic = rgb2gray(pic)

#gray_pic = mpimg.imread('canny_licenceplate2.jpg')


#reduce image img resolution by x times
def reduce_res(img, x):
    reduced_img = numpy.array(img[0:(int(len(img)/x)),0:(int(len(img[0])/x))])
    
    temp_sum = 0
    temp_avg = 0
    
    r = 0
    for i in reduced_img:
        c = 0
        for j in i:
            
            p = 0
            for a in range(x):
                q = 0
                for b in range(x):
                    if(r+p < len(img) and c+q < len(img[0])):
                        temp_sum += img[r+p][c+q]
                    
                    q += 1
                p += 1
                
            temp_avg = temp_sum / (x*x)
            reduced_img[int(r/x)][int(c/x)] = int(temp_avg)
            temp_sum = 0
            temp_avg = 0
            c += x
        r += x
    
    return reduced_img
    

def gaussian(x, mu, sigma):
    return numpy.exp( -(((x-mu)/(sigma))**2)/2.0 )
        
def blur(img, kernel_radius, sigma):
    
    gray_pic_blur = numpy.array(img[:])
    
    #calculate guassian matrix for blur kernel
    hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
    vkernel = [x for x in hkernel]
    gauss_blur_kernel = [[xh*xv for xh in hkernel] for xv in vkernel]
    
    # normalize the kernel elements (can normalize kernel now or normalize image later, but doing both will cause the image to be darker)
    kernelsum = sum([sum(row) for row in gauss_blur_kernel])
    gauss_blur_kernel = [[x/kernelsum for x in row] for row in gauss_blur_kernel]


    temp_sum = 0       
    temp_avg = 0
    
    #begin blurring image
    r = kernel_radius
    for i in img:
        c = kernel_radius
        for j in i:
            if((r+kernel_radius < len(img)) and (c+kernel_radius < len(img[r])) and (c-kernel_radius >= 0) and (r-kernel_radius >= 0)):
                
                gray_pic_slice = numpy.array(img[r-kernel_radius:r+kernel_radius+1, c-kernel_radius:c+kernel_radius+1])
                
                p = 0
                for w in gauss_blur_kernel:
                    q = 0
                    for z in w:
                        temp_sum += gray_pic_slice[p][q] * gauss_blur_kernel[p][q]
                        q += 1
                    p += 1
                
                gray_pic_blur[r][c] = temp_sum
                
                
                temp_sum = 0
                temp_avg = 0
            c += 1
            
        r += 1    
        
    return gray_pic_blur
    
def scale_image(img, tar_x, tar_y):
    
    temp_sum = 0.0
    temp_avg = 0.0
    
    kernel_x = len(img[0]) / tar_x
    kernel_y = len(img) / tar_y
    
    print(kernel_x)
    print(kernel_y)
    
    #round kernel sizes
    kernel_x = int(numpy.round(kernel_x))
    kernel_y = int(numpy.round(kernel_y))
    
    if(kernel_x == 1 and kernel_y == 1):
        print("no scaling needed")
        return img
    
    reduced_img = numpy.array(img[0:tar_y,0:tar_x])
    
    c = 0
    for i in range(tar_x):
        d = 0
        for j in range(tar_y):
            
            p = 0
            for a in range(int(kernel_x)):
                q = 0
                for b in range(int(kernel_y)):
                    if(c+p < len(img[0]) and d+q < len(img)):
                        temp_sum += img[d+q][c+p]
                    
                    q += 1
                p += 1
                            
            temp_avg = temp_sum / (kernel_x*kernel_y)
            
            if(int(c/kernel_x) < len(reduced_img[0]) and int(d/kernel_y) < len(reduced_img)):
                reduced_img[int(d/kernel_y)][int(c/kernel_x)] = int(temp_avg)
            temp_sum = 0
            temp_avg = 0
            
            d += kernel_y
        c += kernel_x
    
    
    print(str(kernel_x) + ' : ' + str(kernel_y))
    return reduced_img
      
    
#currently this function works well on images without much noise
#it works well on images of higher resolution
#it works well after a slight blur has been applied to it
#blur(img,2,2) is usually optimal, but it depends on the resolution / noise
def sobel_edge(img, threshold):
    
    sobel_kernel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_kernel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    #sobel_kernel_x = [[-1,-2,0,2,1],[-4,-8,0,8,4],[-6,-12,0,12,6],[-4,-8,0,8,4],[-1,-2,0,2,1]]
    #sobel_kernel_y = [[-1,-4,-8,-4,-1],[-4,-8,-12,-8,-4],[0,0,0,0,0],[4,8,12, 8, 4],[1,4,8,4,1]]
    
    kernel_radius = 1
    
    edges = numpy.array(img[:])
    angles = numpy.array(img[:])
    
    gx = 0
    gy = 0
    G = 0
    theta = 0
    
    r = kernel_radius
    for i in img:
        c = kernel_radius
        for j in i:
            if((r+kernel_radius < len(img)) and (c+kernel_radius < len(img[r])) and (c-kernel_radius >= 0) and (r-kernel_radius >= 0)):
                
                img_slice = numpy.array(img[r-kernel_radius:r+kernel_radius+1, c-kernel_radius:c+kernel_radius+1])
                
                p = 0
                for w in sobel_kernel_x:
                    q = 0
                    for z in w:
                        gx += img_slice[p][q] * sobel_kernel_x[p][q]
                        gy += img_slice[p][q] * sobel_kernel_y[p][q]
                        q += 1
                    p += 1
                
                #normal edge detection
                G = (numpy.abs(gx) + numpy.abs(gy)) / 2
                
                
                
                #directional edge detection
                if(G > threshold):
                    theta = (180.0/numpy.pi)*(numpy.pi + numpy.arctan2(gy,gx)) + threshold
                else:
                    theta = 0
                
                    
                edges[r][c] = G
                angles[r][c] = theta
                
                gx = 0
                gy = 0
            c += 1
        r += 1    
        
    fixed_angles = numpy.array(angles[:])
    max_theta = numpy.max(angles)
    print(max_theta)
    min_theta = threshold
    print(min_theta)
    
    angle_increment = (max_theta - min_theta) / 8
    print(angle_increment)
    
    angle_start = (angle_increment / 2) + 5
    print(angle_start)
        
    r = 0
    for i in angles:
        c = 0
        for j in i:
            
            if(angles[r][c] >= angle_start and angles[r][c] < angle_start + angle_increment): #left
                fixed_angles[r][c] = 4
            elif(angles[r][c] >= angle_start+angle_increment and angles[r][c] < angle_start + 2*angle_increment):   #top left
                fixed_angles[r][c] = 3
            elif(angles[r][c] >= angle_start+2*angle_increment and angles[r][c] < angle_start + 3*angle_increment): #top
                fixed_angles[r][c] = 2
            elif(angles[r][c] >= angle_start+3*angle_increment and angles[r][c] < angle_start + 4*angle_increment): #top right
                fixed_angles[r][c] = 1
            elif(angles[r][c] >= angle_start+4*angle_increment and angles[r][c] < angle_start + 5*angle_increment): #right
                fixed_angles[r][c] = 4
            elif(angles[r][c] >= angle_start+5*angle_increment and angles[r][c] < angle_start + 6*angle_increment): #bottom right
                fixed_angles[r][c] = 3
            elif(angles[r][c] >= angle_start+6*angle_increment and angles[r][c] < angle_start + 7*angle_increment): #bottom
                fixed_angles[r][c] = 2
            elif(angles[r][c] >= angle_start+7*angle_increment and angles[r][c] < angle_start + 8*angle_increment): #bottom left
                fixed_angles[r][c] = 1
            elif(angles[r][c] > angle_start+8*angle_increment):  #left
                fixed_angles[r][c] = 4
            else:
                fixed_angles[r][c] = 0
                
            c += 1
        r += 1
        
    #return fixed_angles
    return edges
    
def canny_edge(img, threshold, hist_thresh_high, hist_thresh_low):
    sobel_kernel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_kernel_y = [[-1,-2,-1],[0,0,0],[1,2,1]]
    #sobel_kernel_x = [[-1,-2,0,2,1],[-4,-8,0,8,4],[-6,-12,0,12,6],[-4,-8,0,8,4],[-1,-2,0,2,1]]
    #sobel_kernel_y = [[-1,-4,-8,-4,-1],[-4,-8,-12,-8,-4],[0,0,0,0,0],[4,8,12, 8, 4],[1,4,8,4,1]]
    
    kernel_radius = 1
    
    edges = numpy.array(img[:])
    angles = numpy.array(img[:])
    
    gx = 0
    gy = 0
    G = 0
    theta = 0
    
    r = kernel_radius
    for i in img:
        c = kernel_radius
        for j in i:
            if((r+kernel_radius < len(img)) and (c+kernel_radius < len(img[r])) and (c-kernel_radius >= 0) and (r-kernel_radius >= 0)):
                
                img_slice = numpy.array(img[r-kernel_radius:r+kernel_radius+1, c-kernel_radius:c+kernel_radius+1])
                
                p = 0
                for w in sobel_kernel_x:
                    q = 0
                    for z in w:
                        gx += img_slice[p][q] * sobel_kernel_x[p][q]
                        gy += img_slice[p][q] * sobel_kernel_y[p][q]
                        q += 1
                    p += 1
                
                #normal edge detection
                G = (numpy.abs(gx) + numpy.abs(gy)) / 2
                
                
                
                #directional edge detection
                if(G > threshold):
                    theta = (180.0/numpy.pi)*(numpy.pi + numpy.arctan2(gy,gx)) + threshold
                else:
                    theta = 0
                
                    
                edges[r][c] = G
                angles[r][c] = theta
                
                gx = 0
                gy = 0
            c += 1
        r += 1    
        
    fixed_angles = numpy.array(angles[:])
    max_theta = numpy.max(angles)
    min_theta = threshold
    angle_increment = (max_theta - min_theta) / 8
    angle_start = (angle_increment / 2) + 5
        
    r = 0
    for i in fixed_angles:
        c = 0
        for j in i:
            
            if(angles[r][c] >= angle_start and angles[r][c] < angle_start + angle_increment): #left
                fixed_angles[r][c] = 4
            elif(angles[r][c] >= angle_start+angle_increment and angles[r][c] < angle_start + 2*angle_increment):   #top left
                fixed_angles[r][c] = 3
            elif(angles[r][c] >= angle_start+2*angle_increment and angles[r][c] < angle_start + 3*angle_increment): #top
                fixed_angles[r][c] = 2
            elif(angles[r][c] >= angle_start+3*angle_increment and angles[r][c] < angle_start + 4*angle_increment): #top right
                fixed_angles[r][c] = 1
            elif(angles[r][c] >= angle_start+4*angle_increment and angles[r][c] < angle_start + 5*angle_increment): #right
                fixed_angles[r][c] = 4
            elif(angles[r][c] >= angle_start+5*angle_increment and angles[r][c] < angle_start + 6*angle_increment): #bottom right
                fixed_angles[r][c] = 3
            elif(angles[r][c] >= angle_start+6*angle_increment and angles[r][c] < angle_start + 7*angle_increment): #bottom
                fixed_angles[r][c] = 2
            elif(angles[r][c] >= angle_start+7*angle_increment and angles[r][c] < angle_start + 8*angle_increment): #bottom left
                fixed_angles[r][c] = 1
            elif(angles[r][c] > angle_start+8*angle_increment):  #left
                fixed_angles[r][c] = 4
            else:
                fixed_angles[r][c] = 0
                
            c += 1
        r += 1
        
        
    canny_img = numpy.array(edges[:])   
    
    angle_type = -1
    temp_max = 0
    
    r = 0
    for i in edges:
        c = 0
        for j in i:
            
            if((r-1 > 0) and (c-1 > 0) and (r+1 < len(edges)) and (c+1 < len(edges[0]))):
                
                #horizontal from left
                temp_max = edges[r][c-1]
                if(fixed_angles[r][c] == 4):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r][c+1] == 4):
                    temp_max = numpy.max((temp_max,edges[r][c+1]))
                
                if(edges[r][c-1] < temp_max):
                    if(fixed_angles[r][c-1] == 4):
                        fixed_angles[r][c-1] = 0
                        edges[r][c-1] = 0
                
                #horizontal from right
                temp_max = edges[r][c+1]
                if(fixed_angles[r][c] == 4):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r][c-1] == 4):
                    temp_max = numpy.max((temp_max,edges[r][c-1]))
                
                if(edges[r][c+1] < temp_max):
                    if(fixed_angles[r][c+1] == 4):
                        fixed_angles[r][c+1] = 0
                        edges[r][c+1] = 0                    
                
                #topleft->botright
                temp_max = edges[r-1][c-1]
                if(fixed_angles[r][c] == 3):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r+1][c+1] == 3):
                    temp_max = numpy.max((temp_max,edges[r+1][c+1]))
                
                if(edges[r-1][c-1] < temp_max):
                    if(fixed_angles[r-1][c-1] == 3):
                        fixed_angles[r-1][c-1] = 0
                        edges[r-1][c-1] = 0
                        
                #botright->topleft
                temp_max = edges[r+1][c+1]
                if(fixed_angles[r][c] == 3):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r-1][c-1] == 3):
                    temp_max = numpy.max((temp_max,edges[r-1][c-1]))
                
                if(edges[r+1][c+1] < temp_max):
                    if(fixed_angles[r+1][c+1] == 3):
                        fixed_angles[r+1][c+1] = 0
                        edges[r+1][c+1] = 0
                        
                #vertical from top
                temp_max = edges[r-1][c]
                if(fixed_angles[r][c] == 2):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r+1][c] == 2):
                    temp_max = numpy.max((temp_max,edges[r+1][c]))
                
                if(edges[r-1][c] < temp_max):
                    if(fixed_angles[r-1][c] == 2):
                        fixed_angles[r-1][c] = 0
                        edges[r-1][c] = 0
                        
                #vertical from bot
                temp_max = edges[r+1][c]
                if(fixed_angles[r][c] == 2):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r-1][c] == 2):
                    temp_max = numpy.max((temp_max,edges[r-1][c]))
                
                if(edges[r+1][c] < temp_max):
                    if(fixed_angles[r+1][c] == 2):
                        fixed_angles[r+1][c] = 0
                        edges[r+1][c] = 0
                
                #topright->botleft
                temp_max = edges[r-1][c+1]
                if(fixed_angles[r][c] == 1):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r+1][c-1] == 1):
                    temp_max = numpy.max((temp_max,edges[r-1][c+1]))
                
                if(edges[r-1][c+1] < temp_max):
                    if(fixed_angles[r-1][c+1] == 1):
                        fixed_angles[r-1][c+1] = 0
                        edges[r-1][c+1] = 0
                
                #botleft->topright
                temp_max = edges[r+1][c-1]
                if(fixed_angles[r][c] == 1):
                    temp_max = numpy.max((temp_max,edges[r][c]))
                if(fixed_angles[r-1][c+1] == 1):
                    temp_max = numpy.max((temp_max,edges[r+1][c-1]))
                
                if(edges[r+1][c-1] < temp_max):
                    if(fixed_angles[r+1][c-1] == 1):
                        fixed_angles[r+1][c-1] = 0
                        edges[r+1][c-1] = 0
                
            c += 1
        r += 1
        
    max_val = hist_thresh_high * numpy.max(edges)
    min_val = hist_thresh_low * numpy.max(edges)
    
    #canny edge detection    
    r = 0
    for i in edges:
        c = 0
        for j in i:
            if(edges[r][c] > max_val):
                edges[r][c] = max_val * 2
                
                hysteresis_connection(r, c, edges, min_val, max_val)
                        
            if(edges[r][c] < min_val):
                edges[r][c] = 0
        
            
            c = c + 1
        r = r + 1

    return edges
    
def hysteresis_connection(x, y, edges, min_val, max_val):
    #get neighboring pixels that are above min
    m = x - 1
    for p in range(3):
        n = y - 1
        for q in range(3):
            if(m < len(edges) and n < len(edges[0]) and m > 0 and n > 0 and not(m == x and n == y)):
                if(edges[m][n] < max_val):
                    if(edges[m][n] > min_val):
                        edges[m][n] = max_val * 2
                        hysteresis_connection(m, n, edges, min_val, max_val)
                    else:
                        edges[m][n] = 0
            n += 1
        m += 1
    
def simple_som(image, l, n, tl, tn, epochs, snapshot):
    
    #l is learning rate
    #n is neighborhood
    #tl is a time factor for how fast learning rate decays
    #tn is a time factor for how fast neighborhood decays
    
    #           0 = x value
    #           1 = y value
    #neurons[n][0-1]
    
    #neurons = numpy.array([[100,150.0],[225,150.0],[375.0,150.0],[550.0,150.0],[700.0,150.0],[825.0,150.0]])
    
    neurons = [[random.randint(0,len(image[0])), random.randint(0, len(image))] for q in range(40)]
    
    """
    #auto grid
    xgrid = 20
    ygrid = 6
    yoffset = len(image) / (ygrid+1)
    xoffset = len(image[0]) / (xgrid+1)
    
    neurons = [[0,0] for q in range(xgrid*ygrid)]
    c = 0
    for i in range(xgrid):
        for j in range(ygrid):
            neurons[c][0] = (i+1) * numpy.floor(xoffset)
            neurons[c][1] = (j+1) * numpy.floor(yoffset)
            c += 1
    """
            
    winners = numpy.array(numpy.empty(len(neurons)), int)
    
    highlighted_image = highlight_neurons(image, round_neurons(neurons), 5)
    plt.imshow(highlighted_image, 'gray')
    plt.show()
    plt.clf()
    
    
    for t in range(1, epochs):
        neurons = update_neurons(image, neurons, winners, t)
        
        #check a snapshot of the image and neurons every few iterations
        if(t % snapshot == 0):
            print(t)
            #print(neurons)
            highlighted_image = highlight_neurons(image, round_neurons(neurons), 5)
            plt.imshow(highlighted_image, 'gray')
            plt.show()
            plt.clf()

           
    return highlighted_image

def update_neurons(image, neurons, winners, time):
    winner = 0
    
    sigma_not = 0.8
    tau_sigma = 30000
    
    nu_not = 0.01
    tau_nu = 10000
    
    neighborhood = numpy.array(numpy.empty(len(neurons)), float)
    
    nsize = sigma_not / numpy.exp(time/tau_sigma)
    
    learning_rate = nu_not / numpy.exp(time/tau_nu)
    
    
    #number of times a neuron can win before it yeilds to other neurons
    winner_cap = 9
    
    
    done = False
    while (done == False):
    
        #select random pixel from the input
        x = random.randint(5,(len(image[0]) - 5))
        y = random.randint(5,(len(image) - 5))
        #print("X / Y random point: " + str(x) + "/" + str(y))
        
        #only check pixel if its not background
        if(image[y][x] != 0):
            
            #print("WHITE FOUND!!")
            
            #active pixel was picked
            done = True
            
            #find winner (closest neuron to the current pixel)
            c = 0
            min_distance = 9999
            temp_dist = 0
            for n in neurons:
                
                
                if(winners[c] < winner_cap):
                    #temp_dist = numpy.sqrt(numpy.square(x - neurons[c][0]) + numpy.square(y - neurons[c][1]))
                    temp_dist += numpy.square(x - neurons[c][0])
                    temp_dist += numpy.square(y - neurons[c][1])
                    temp_dist = numpy.sqrt(temp_dist)
                    
                    if(temp_dist < min_distance):
                        winner = c
                        min_distance = temp_dist
                        
                
                c+= 1
                temp_dist = 0
            
            #increment neuron as a winner
            winners[winner] += 1
            #check if winners should be reset
            #if all neurons have won at least once, reset all to 0
            reset = 1
            c = 0
            for w in winners:
                if(winners[c] == 0):
                    reset = 0
                
                c += 1
            #actual reset
            if(reset == 1):
                c = 0
                for w in winners:
                    winners[c] = 0
                    c += 1
                    
                    
            #now that winner is found, determine neighborhood
            c = 0
            for n in neurons:
                
                #excite distance is lateral distance between winning neuron and current neuron.
                #winner coords
                w_x = neurons[winner][0]
                w_y = neurons[winner][1]
                #current neuron coords
                c_x = neurons[c][0]
                c_y = neurons[c][1]
                
                if(w_x != c_x or w_y != c_y):
                    
                    #distance between winner and current neuron
                    d1 = numpy.sqrt(numpy.square(w_x - c_x) + numpy.square(w_y - c_y))
                    
                    #distance between winner and selected point
                    d2 = numpy.sqrt(numpy.square(w_x - x) + numpy.square(w_y - y))
                    
                    #distance between current neuron and selected point
                    d3 = numpy.sqrt(numpy.square(c_x - x) + numpy.square(c_y - y))
                    
                    if(d1 == 0 or d2 == 0):
                        print("----------ERROR----------")
                        print(d1)
                        print(d2)
                        n_val = 0
                    else:
                        value = (numpy.square(d1) + numpy.square(d2) - numpy.square(d3)) / (2*d1*d2)
                        
                        #angle next to winner in the triangle between winner, current neuron, and selected point
                        theta = numpy.arccos(numpy.clip(value,-1,1))
                        
                        #lateral distance between winner and current neuron
                        lateral_distance = d1 * numpy.cos(theta)
                        
                        #scale down excite_distance
                        #lateral_distance /= len(image)
                        
                        n_val = 1/(lateral_distance * time)
                    
                    #n_val = 1 / numpy.exp(numpy.square(lateral_distance) / (2.0*numpy.square(nsize)))
                    
                    """
                    if(time % 1000 == 0):
                        print(lateral_distance)
                        print(neighborhood)
                        print(n_val)
                        print(nsize)
                    """
                else:
                    n_val = 1
                    
                neighborhood[c] = n_val    
                
                    
                c += 1
                
            #adjust neuron coordinates
            c = 0
            for n in neurons:
                
                dx = learning_rate * neighborhood[c] * (x - neurons[c][0])
                dy = learning_rate * neighborhood[c] * (y - neurons[c][1])
                
                """
                if(time % 5000 == 0):
                    print ("[x, y]: [" + str(x) + ", " + str(y) + "]")
                    print ("[dx, dy]: [" + str(dx) + ", " + str(dy) + "]")
                """
                neurons[c][0] = neurons[c][0] + dx
                neurons[c][1] = neurons[c][1] + dy
                
                c += 1
    
    
    
    """
    #go through each pixel one at a time finding closest neuron each time
    x = 0
    for i in image:
        y = 0
        for j in i:
            winner = 0
            
            #only check pixel if its not background
            #and within distance from neuron
            if(image[x][y] != 0):
                
                #find winner (closest neuron to the current pixel)
                c = 0
                min_distance = 9999
                for n in neurons:
                    temp_dist = numpy.sqrt(numpy.square(x - neurons[c][0]) + numpy.square(y - neurons[c][1]))
                    if(temp_dist < min_distance):
                        winner = c
                        min_distance = temp_dist
                    c+= 1
                
                #now that winner is found, determine neighborhood
                c = 0
                for n in neurons:
                    excite_distance = numpy.sqrt(numpy.square(neurons[winner][0] - neurons[c][0]) + numpy.square(neurons[winner][1] - neurons[c][1]))
                    
                    neighborhood[c] = numpy.exp(-numpy.square(excite_distance) / (2*numpy.square(nsize)))
                    
                    c += 1
                    
                #adjust neuron coordinates
                c = 0
                for n in neurons:
                    
                    dx = learning_rate * neighborhood[c] * (x - neurons[c][0])
                    dy = learning_rate * neighborhood[c] * (y - neurons[c][1])
                    
                    neurons[c][0] = neurons[c][0] + dx
                    neurons[c][1] = neurons[c][1] + dy
                    
                    c += 1
            y += 1
        x+= 1
    """
    return neurons
    
def round_neurons(neurons):
    
    c = 0
    for i in neurons:
        neurons[c][0] = numpy.floor(neurons[c][0])
        neurons[c][1] = numpy.floor(neurons[c][1])
        c += 1
    return neurons
    
def highlight_neurons(image, neurons, contrast):
    
    #highlight neurons to show where they are
    intensity = numpy.max(image) * contrast
    
    highlighted_image = numpy.array(image)
    
    c = 0
    for i in neurons:
        
        #highlight a 3x3 around the neuron
        for p in range(10):
            for q in range(10):
                if(neurons[c][0] - 1 + p >= 0 and neurons[c][0] - 1 + p < len(highlighted_image[0]) and neurons[c][1] - 1 + q >= 0 and neurons[c][1] - 1 + q < len(highlighted_image)):
                    highlighted_image[neurons[c][1] - 1 + q][neurons[c][0] - 1 + p] = intensity
                    
        
        c += 1
    
    return highlighted_image

"""    
plt.figure(2)


print("\n\nnormal image full resolution")
plt.imshow(gray_pic, 'gray')
plt.show()
plt.clf()


#gray_pic = reduce_res(gray_pic,5)

#new_pic2 = sobel_edge(gray_pic, 50)


new_pic2 = canny_edge(gray_pic, 50, 0.5, 0.25)

plt.imshow(new_pic2, 'gray')
plt.show()
plt.clf()

scipy.misc.imsave('canny_licenceplate2.jpg', new_pic2)


new_pic3 = simple_som(gray_pic, 1, 1, 1, 1, 20000, 2000)

plt.imshow(new_pic3, 'gray')
plt.show()
plt.clf()

scipy.misc.imsave('som_lp.jpg', new_pic3)



#reduce resolution by 2
print("\n\nreduced res")
gray_pic = reduce_res(gray_pic,5)
plt.imshow(gray_pic, 'gray')
plt.show()
plt.clf()


print("\n\nsobel directional edge detection -- blur:2 -- full res")
new_pic1 = sobel_edge(blur(gray_pic, 2,2), 50)
plt.imshow(new_pic1)
plt.show()
plt.clf()

print("\n\nsobel directional edge detection -- blur:2 -- full res")
new_pic1 = sobel_edge(gray_pic, 50)
plt.imshow(new_pic1)
plt.show()
plt.clf()

new_pic2 = canny_edge(blur(gray_pic, 2,2), 50, 0.5, 0.25)
plt.imshow(new_pic2)
plt.show()
plt.clf()

new_pic2 = canny_edge(gray_pic, 50, 0.5, 0.25)
plt.imshow(new_pic2)
plt.show()
plt.clf()
"""
"""
new_pic1 = numpy.array(gray_pic)
new_pic1 = sobel_edge(blur(gray_pic, 2,2), 50)

#blur:2 optimal for closeup of licence plate
print("\n\nsobel directional edge detection -- blur:2 -- full res")
plt.imshow(new_pic1)
plt.show()
plt.clf()

#ideal values for closeup of a licence plate at res 500x1000
new_pic2 = numpy.array(gray_pic)

#reduce resolution by 2
new_pic2 = reduce_res(gray_pic,2)

new_pic2 = canny_edge(blur(gray_pic, 2,2), 50, 0.5, 0.25)
plt.imshow(new_pic2)
plt.show()
plt.clf()
"""


"""
new_pic3 = numpy.array(gray_pic)
new_pic3 = canny_edge(blur(gray_pic, 2,2), 50, 0.5, 0.25)
plt.imshow(new_pic3)
plt.show()
plt.clf()

scipy.misc.imsave('canny_licenceplate_lowres.jpg', new_pic3)


#reduce resolution by 5 times
#gray_pic = reduce_res(gray_pic,5)

print("\n\nnormal image reduced resolution x 5")
plt.imshow(gray_pic, 'gray')
plt.show()
plt.clf()

plt.imshow(blur(gray_pic,4,4), 'gray')
plt.show()
plt.clf()
"""

