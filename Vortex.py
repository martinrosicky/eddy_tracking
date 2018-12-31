# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 2018

@author: Martin

version: 1.0

Implement a class object "Node" and doubly-linked list "Doublelist" 

"""
# libraries
import numpy as np
from utils import dist
from netCDF4 import Dataset


# Load latitude and longitude
with Dataset('gom_reanalysis.nc', 'r') as dataset:
    Y, X = dataset['lat/GOM'][0:325], dataset['lon/GOM'][0:430] 

class Node(object):
    """ Detected center of an eddy """ 
    def __init__(self, day, lat, lon, rot, prev, next):
        self.day = day
        self.lat = lat
        self.lon = lon
        self.rot = rot    # type of eddy: +1 for cyclonic/ -1 for anti-cyclonic
        self.prev = prev
        self.next = next
 
 
class Doublelist(object):
    """ Tracking centers of an eddy during its lifetime """
    head = None
    tail = None
 
    def append(self, day, lat, lon, rot):
        new_node = Node(day, lat, lon, rot, None, None)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.prev = self.tail
            new_node.next = None
            self.tail.next = new_node
            self.tail = new_node
 
    def remove(self, day, lat, lon, rot):
        current_node = self.head
 
        while current_node is not None:
            if current_node.day == day and current_node.lat == lat and current_node.lon == lon and current_node.rot == rot:
                # if it's not the first element
                if current_node.prev is not None:
                    current_node.prev.next = current_node.next
                    current_node.next.prev = current_node.prev
                else:
                    # otherwise we have no prev (it's None), head is the next one, and prev becomes None
                    self.head = current_node.next
                    current_node.next.prev = None
 
            current_node = current_node.next
 
    def show(self):
        print "Show list data: (day, lat, lon, rot)"
        current_node = self.head
        while current_node is not None:
            print "({}, {}, {}, {})".format(current_node.day, current_node.lat, current_node.lon, current_node.rot)
            current_node = current_node.next
        print "*"*50
    
    def getdata(self):
        current_node = self.head
        data = np.array([]).reshape(0, 3)
        while current_node is not None:
            data = np.vstack([data, np.array([current_node.day, current_node.lat, current_node.lon])])
            current_node = current_node.next
        return data
        print "*"*50
    
    def lifetime(self):
        return (getattr(self.tail, "day") - getattr(self.head, "day")) / 24 + 1
        
    def speed(self):
        """ Computes eddy's instantaneous speed """
        current_node = self.head
        speed = np.array([]).reshape(0, 2)
        while current_node is not None:
            if current_node == self.head:
                temp = np.array([current_node.day, None])
            else:
                temp = np.array([current_node.day, dist(Y[current_node.lat], X[current_node.lon], Y[current_node.prev.lat], X[current_node.prev.lon])/((current_node.day - current_node.prev.day)*3600 * 24.0)])
            speed = np.vstack([speed, temp])
            current_node = current_node.next
        return speed

    def mode(self):
        """ display eddy's mode: cyclonic or anti-cyclonic """
        current_node = self.head
        return "cyclonic" if current_node.rot == 1 else "anti-cyclonic"
        