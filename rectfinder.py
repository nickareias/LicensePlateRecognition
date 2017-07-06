# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:02:19 2017

@author: Jeffrey Paquette
"""

import numpy
#import common.filters as filters
import filters

class Point():
    x = 0
    y = 0
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
class Rect():
    top = 0         # top edge of rect
    bottom = 0      # bottom edge of rect
    left = 0        # left edge of rect
    right = 0       # right edge of rect
    area = 0        # area of rect
    
    def __init__(self, top, bottom, left, right):
        self.top = top
        self.bottom = bottom
        self.left = left
        self.right = right
        self.calculate_area()
        return
    
    def calculate_area(self):
        self.area = (self.bottom - self.top + 1) * (self.right - self.left + 1)
        return
        
    def isAdjacent(self, x, y):
        """Returns true if this point is adjacent to this rect"""
        if (x + 1 == self.left and y >= self.top and y <= self.bottom):
            return True
        elif (x - 1 == self.right and y >= self.top and y <= self.bottom):
            return True
        elif (y - 1 == self.bottom and x >= self.left and x <= self.right):
            return True
        elif (y + 1 == self.top and x >= self.left and x <= self.right):
            return True
        else:
            return False
    
    def contains(self, x, y):
        """Returns true if this point is inside this rect"""
        if (x >= self.left and x <= self.right and y >= self.top and y <= self.bottom):
            return True
        else:
            return False
        
    def add(self, x, y):
        """Expands the boundaries of this rect to include this point"""
        if (x < self.left):
            self.left = x
        elif (x > self.right):
            self.right = x
            
        if (y < self.top):
            self.top = y
        elif (y > self.bottom):
            self.bottom = y
        
        # update area calculation
        self.calculate_area()
        return
        
class RectConstructor():
    rects = []      # list of rectangles
    img_area = 0    # area of entire image
    mid_y = 0       # middle y position of pic

    def __init__(self, image, area_min, area_max, max_mid_distance):
        # total area of the image
        self.img_area = len(image) * len(image[0])
        
        # middle y position of image
        self.mid_y = len(image) / 2

        # for each row of pixels (exluding 5 pixels on the edges)
        for y in range(5, len(image) - 5):
            # for each column of pixles (excluding 10 pixels on the edges)
            for x in range(10, len(image[0]) - 10):
                # if the pixel is not black
                if (image[y][x] != 0):
                    rect_found = False
                    # create a new rectangle if none have been created yet
                    if (len(self.rects) == 0):
                        self.rects.append(Rect(y, y, x, x))
                        continue
                    
                    # for each rectangle already created
                    for r in self.rects:
                        # if the pixel is already in this rectangle then disregard pixel
                        if (r.contains(x, y)):
                            rect_found = True
                            break
                        
                        # if pixel is adjacent to this rectangle then add this position to it
                        if (r.isAdjacent(x, y)):
                            r.add(x, y)
                            rect_found = True
                            break
                    # if the pixel has not been added then create a new one
                    if (not rect_found):
                        self.rects.append(Rect(y, y, x, x))
        # merge rectangles that are together
        self.merge_rects()
        
        # delete rectangles that fall within parameters
        self.reject_rects(area_min, area_max, max_mid_distance)
        return
    
    def merge_rects(self):
        rects_merged = True
        # loop until there are no rectangles left to merge
        while (rects_merged):
            rects_merged = False
            to_remove = []

            # loop through all rectangles for each rectangle
            for ra in self.rects:
                for rb in self.rects:
                    # don't compare to self
                    if (ra == rb):
                        continue
                    
                    # if the rectangles overlap, merge them and mark rb for deletion
                    if (ra.left + 1 < rb.right - 1 and ra.right + 1 > rb.left - 1 and
                        ra.top - 1 < rb.bottom + 1 and ra.bottom + 1 > rb.top -1 ):
                        rects_merged = True
                        to_remove.append(rb)
                        if (rb.left < ra.left):
                            ra.left = rb.left
                        if (rb.right > ra.right):
                            ra.right = rb.right
                        if (rb.top < ra.top):
                            ra.top = rb.top
                        if (rb.bottom > ra.bottom):
                            ra.bottom = rb.bottom
                        ra.calculate_area()
                
                # if rectangles have been merged then break to remove rects marked for deletion
                if (rects_merged):
                    break
                
            # delete all rectangles marked for deletion
            for r in to_remove:
                self.rects.remove(r)
    
    def reject_rects(self, area_min, area_max, max_mid_distance):
        to_remove = []
        min_size = area_min * self.img_area
        max_size = area_max * self.img_area
        
        for r in self.rects:
            if (r.area < min_size):
                to_remove.append(r)
            elif (r.area > max_size):
                to_remove.append(r)
            elif (r.top - self.mid_y > max_mid_distance * self.mid_y):
                to_remove.append(r)
            elif (self.mid_y - r.bottom > max_mid_distance * self.mid_y):
                to_remove.append(r)
                
        for r in to_remove:
            self.rects.remove(r)
        return
        
    def highlight_rects(self, image):
        highlighted_image = filters.darken(numpy.array(image), 2)
        for r in self.rects:
            for x in range (r.left, r.right):
                highlighted_image[r.top][x] = 255
                highlighted_image[r.bottom][x] = 255
            for y in range (r.top, r.bottom):
                highlighted_image[y][r.left] = 255
                highlighted_image[y][r.right] = 255
        return highlighted_image

        
class Tracer():
    #rects = []      # list of rectangles
    #img = []        #image to be processed
    #img_area = 0    # area of entire image
    #mid_y = 0       # middle y position of pic
    #width = 0       # width of image
    #height = 0      # height of image
    #visited = []    # array same size as image to mark visited pixels
    
    def __init__(self, image, area_min, area_max, max_mid_distance):        
        # list of rectangles
        self.rects = []
        
        # store image in class variable
        self.img = image
        self.width = len(self.img[0])
        self.height = len(self.img)
        
         # total area of the image
        self.img_area = len(image) * len(image[0])
        
        # min and max sizes of rectangles
        self.min_size = area_min * self.img_area
        self.max_size = area_max * self.img_area
        
        # max distance away from middle horizontal line
        self.max_mid_distance = max_mid_distance
        
        self.visited = numpy.zeros_like(self.img)

        # middle y position of image
        self.mid_y = len(image) / 2
        
        # for each row of pixels (exluding 5 pixels on the edges)
        for y in range(10, len(image) - 10):
            # for each column of pixles (excluding 10 pixels on the edges)
            for x in range(10, len(image[0]) - 10):
                    # if the pixel is not black
                    # and if the pixel is not within a rectangle
                    #self.reject_rects(area_min, area_max, max_mid_distance)
                    
                    if (image[y][x] != 0):
                        if (self.visited[y][x] == 1):
                            continue
                        
                        if (len(self.rects) == 0):
                            new_rectangle = Rect(y, y, x, x)
                            self.rects.append(new_rectangle)
                            self.trace(new_rectangle, x, y)
                            self.reject_rect(new_rectangle)
                        else:
                            in_rect = False
                            for r in self.rects:
                                if (r.contains(x, y)):
                                    in_rect = True
                                    break
                            if (not in_rect):
                                new_rectangle = Rect(y, y, x, x)
                                self.rects.append(new_rectangle)
                                self.trace(new_rectangle, x, y)
                                self.reject_rect(new_rectangle)
        return
    
    def trace(self, r, x, y):
        to_visit = []   #list of points to visit
        to_visit.append(Point(x, y))
        
        while (len(to_visit) > 0):
            pixel = to_visit.pop()
            self.visited[pixel.y][pixel.x] = 1

            if (self.img[pixel.y][pixel.x] != 0):
                r.add(pixel.x, pixel.y)
                if (pixel.x+1 < self.width-10 and self.visited[pixel.y][pixel.x+1] == 0):
                    to_visit.append(Point(pixel.x+1, pixel.y))
                if (pixel.x-1 > 10 and self.visited[pixel.y][pixel.x-1] == 0):
                    to_visit.append(Point(pixel.x-1, pixel.y))
                if (pixel.y+1 < self.height-10 and self.visited[pixel.y+1][pixel.x] == 0):
                    to_visit.append(Point(pixel.x, pixel.y+1))
                if (pixel.y-1 > 10 and self.visited[pixel.y-1][pixel.x] == 0):
                    to_visit.append(Point(pixel.x, pixel.y-1))
        return
    
    def reject_rect(self, rect):
        if (rect.area < self.min_size):
            self.rects.remove(rect)
        elif (rect.area > self.max_size):
            self.rects.remove(rect)
        elif (rect.top - self.mid_y > self.max_mid_distance * self.mid_y):
            self.rects.remove(rect)
        elif (self.mid_y - rect.bottom > self.max_mid_distance * self.mid_y):
            self.rects.remove(rect)
        return
            
    def reject_rects(self, area_min, area_max, max_mid_distance):
        to_remove = []
        min_size = area_min * self.img_area
        max_size = area_max * self.img_area
        
        for r in self.rects:
            if (r.area < min_size):
                to_remove.append(r)
            elif (r.area > max_size):
                to_remove.append(r)
            elif (r.top - self.mid_y > max_mid_distance * self.mid_y):
                to_remove.append(r)
            elif (self.mid_y - r.bottom > max_mid_distance * self.mid_y):
                to_remove.append(r)
                
        for r in to_remove:
            self.rects.remove(r)
        return
    
    def highlight_rects(self):
        highlighted_image = filters.darken(numpy.array(self.img), 2)
        for r in self.rects:
            for x in range (r.left, r.right):
                highlighted_image[r.top][x] = 255
                highlighted_image[r.bottom][x] = 255
            for y in range (r.top + 1, r.bottom - 1):
                highlighted_image[y][r.left] = 255
                highlighted_image[y][r.right] = 255
        return highlighted_image
        
    def extract_rects(self, original):
        images = []     # list of images cut out from origin
        for r in self.rects:
            image = numpy.array(original[r.top:r.bottom, r.left:r.right])
            images.append(image)
        return images