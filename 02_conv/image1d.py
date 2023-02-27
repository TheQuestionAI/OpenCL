# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 10:29:56 2023

@author: gcd_c
"""

import numpy as np

class Image1D:
    def __init__(self, width, data = None, data_type = None):        
        
        assert isinstance(width, int) and width > 0, "Input size is not valid!"
        
        self._w = width
        self._dtype = data_type if data_type is not None else np.float32
        self._data = np.zeros((self._w, 4), dtype = self._dtype)
        
        if data is not None:
            assert isinstance(data, np.ndarray), "Input data type is not valid!"
            assert len(data.shape) == 1, "Only support 1D flatten data ndarray!"
            
            size = data.size                    
            idx = 0
            for i in range(self._w):
                if idx >= size:
                    continue
                    
                if idx < size:
                    self._data[i][0] = data[idx]
                if idx+1 < size:
                    self._data[i][1] = data[idx+1]
                if idx+2 < size:
                    self._data[i][2] = data[idx+2]
                if idx+3 < size:
                    self._data[i][3] = data[idx+3]
                    
                idx += 4
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def size(self):
        return self._data.size
                    
    def read_image(self, coord):
        assert isinstance(coord, int), "coord must be scalar integer!"
        
        x = coord
        if x < 0 or x >= self._w:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        return self._data[x]
    
    def write_image(self, coord, value):
        
        assert isinstance(coord, int), "coord must be scalar integer!"
        
        x = coord
        if x < 0 or x >= self._w:        
            print("(x,y) is out of image boundary.")
            return

        assert isinstance(value, (list, tuple, np.ndarray)), "value type is not valid!"
        assert len(value) == 4, "value size is not valid!"
        
        self._data[x] = value
        
    def __getitem__(self, key):
        assert key >= 0 or key < self._h, "key is out of image Height boundary!"
        
        return self._data[key]
    
    def __repr__(self):
        return str(self._data)
    
    @property
    def values(self):
        return self._data
    
    def to_numpy(self, shape = None):
        assert isinstance(shape, tuple), "shape must be tuple!"
        
        if shape is not None:
            return self._data.flatten().reshape(*shape)
        return self._data
    

if __name__ == "__main__":
    width = 5
        
    raw_data = np.arange(5*5)
    
    weights = Image1D(width = width, data = raw_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    