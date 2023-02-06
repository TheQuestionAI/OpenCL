# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 15:10:08 2023

@author: gcd_c
"""

import numpy as np

class Image2D:
    def __init__(self, shape = None, width = None, height = None, data = None, data_type = None):        
        assert shape is not None or (width is not None and height is not None), "Input shape or width and height are not valid!"
        
        if shape is not None:
            types = (list, tuple, np.ndarray)
            assert isinstance(shape, types), "Input shape type is not valid!"
            assert len(shape) == 2, "Input shape size is not valid!"
            assert isinstance(shape[0], int) and isinstance(shape[1], int), "Input element data type is not valid!"
            assert shape[0] > 0 and shape[1] > 0, "Input element data value is not valid!"
            
            self._w, self._h = shape
        
        if width is not None and height is not None:
            assert isinstance(width, int) and width > 0, "Input width is not valid!"
            assert isinstance(height, int) and height > 0, "Input height is not valid!"
            
            self._w = width
            self._h = height
            
        self._dtype = data_type if data_type is not None else np.float32
        
        self._data = np.zeros((self._h, self._w, 4),dtype = self._dtype())
        if data is not None:
            assert isinstance(data, np.ndarray), "Input data type is not valid!"
            assert len(data.shape) == 1, "Only support 1D flatten data ndarray!"
            
            size = data.size                    
            idx = 0
            for i in range(self._h):
                for j in range(self._w):
                    if idx >= size:
                        continue
                    
                    if idx < size:
                        self._data[i][j][0] = data[idx]
                    if idx+1 < size:
                        self._data[i][j][1] = data[idx+1]
                    if idx+2 < size:
                        self._data[i][j][2] = data[idx+2]
                    if idx+3 < size:
                        self._data[i][j][3] = data[idx+3]
                        
                    idx += 4
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def size(self):
        return self._data.size
                    
    def read_image(self, coord):
        assert isinstance(coord, tuple) and len(coord) == 2, "coord must be tuple of size 2!"
        
        x, y = coord
        assert isinstance(x, int) and isinstance(y, int), "x,y data type is not valid!"
        
        if x < 0 or x >= self._w or y < 0 or y >= self._h:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        return self._data[y][x]
    
    def write_image(self, coord, value):
        
        assert isinstance(coord, tuple) and len(coord) == 2, "coord must be tuple of size 2!"
        
        x, y = coord
        assert isinstance(x, int) and isinstance(y, int), "x,y data type is not valid!"
        
        if x < 0 or x >= self._w or y < 0 or y >= self._h:        
            print("(x,y) is out of image boundary.")
            return

        assert isinstance(value, (list, tuple, np.ndarray)), "value type is not valid!"
        assert len(value) == 4, "value size is not valid!"
        
        self._data[y][x] = value
        
    def __getitem__(self, key):
        assert key >= 0 or key < self._h, "key is out of image Height boundary!"
        
        return self._data[key]
    
    def __repr__(self):
        return str(self._data)
    
    @property
    def values(self):
        return self._data
    

if __name__ == "__main__":
    width = 3
    height = 2
        
    raw_data = np.arange(5*5)
    
    weights = Image2D(shape = (width, height), data = raw_data)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    