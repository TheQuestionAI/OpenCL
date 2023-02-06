# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:22:17 2023

@author: gcd_c
"""

import numpy as np

# We assume the input src_data is 4D with data layout dim = (D, H, W, C), C is aligned with 4.
# We want to return dst_data which is 4D with data yout dim = (C/4, D, H, W*4c)
def DHWC_to_CgDHWC4(src_data):      
    """checked"""
    
    assert isinstance(src_data, np.ndarray), "input data array must be numpy array!"
    assert len(src_data.shape) == 4, "The dimension of the input array must be 4!"
    
    D, H, W, C = src_data.shape
    
    newC = int(C / 4)
    newD = D
    newH = H
    newW = W * 4
    
    dst_data = np.zeros((newC, newD, newH, newW), dtype = src_data.dtype)
    
    # 我们loop dst_data来fill它. 用src_data对应的value来fill每一个位置(实际上是每4个位置)
    for c in range(newC):       
        # dst_data有newC = C/4个group channels, 也即每次loop我们一次性从src_data提取4 channel数据.
        old_c = 4*c         # 找到当前group channel对应要提取的4个channel数据的starting channel location.
        # 当前loop, 要提取数据的channels是 [4*c, 4*c+1, 4*c+2, 4*c+3]
        for d in range(newD):
            old_d = d       # dst_data和src_data在depth维度上大小相同.
            for h in range(newH):
                old_h = h   # dst_data和src_data在height维度上大小相同.
                for w in range(0, newW, 4):     # 我们每个location(d,h,w')存储4个channel的在这一location对应的数据
                    old_w = int(w/4)    # dst_data有newW = W*4, 也即width大小是src_data的4倍. 
                    # 我们除以4以找到对应的src_data width location.
                    
                    # 每次loop, 每个location(d,h,w'), w'=w:w+4, 我们一次性从src_data对应位置提取4 channel数据.
                    dst_data[c][d][h][w:w+4] = src_data[old_d][old_h][old_w][old_c:old_c+4]
        
    return dst_data


# We assume the input src_data is 4D with data layout dim = (C/4, D, H, W*4c), C is aligned with 4.
# We want to return dst_data which is 4D with data yout dim = (D, H, W, C)
def CgDHWC4_to_DHWC(src_data):      
    """checked"""
    
    assert isinstance(src_data, np.ndarray), "input data array must be numpy array!"
    assert len(src_data.shape) == 4, "The dimension of the input array must be 4!"
    
    C, D, H, W = src_data.shape
    
    newC = C * 4
    newD = D
    newH = H
    newW = int(W / 4)
    
    dst_data = np.zeros((newD, newH, newW, newC), dtype = src_data.dtype)
    
    # 我们loop dst_data来fill它. 用src_data对应的value来fill每一个位置(实际上是每4个位置)
    for d in range(newD):
        old_d = d
        for h in range(newH):
            old_h = h
            for w in range(newW):
                old_w = 4 * w
                for c in range(0, newC, 4):
                    old_c = int(c / 4)
                    
                    dst_data[d][h][w][c:c+4] = src_data[old_c][old_d][old_h][old_w:old_w+4
                                                                             ]
    return dst_data
    

# We assume the input src_data is 5D with data layout dim = (Dk, Hk, Wk, Cin, Cout), Cout and Cin is aligned with 4.
# We want to return dst_data which is 5D with data yout dim = (Cin/4, Dk, Hk, Wk, Cout*4cin)
def DHWCiCo_to_CgiDHWCoCi4(src_data):
    """checked"""
    
    assert isinstance(src_data, np.ndarray), "input data array must be numpy array!"
    assert len(src_data.shape) == 5, "The dimension of the input array must be 4!"
    
    # convert DHWCiCo to DHWCoCi first.
    permutation = (0, 1, 2, 4, 3)            # 01234 -> 01243
    src_data_tmp = np.transpose(src_data, axes = permutation)
    
    D, H, W, Co, Ci = src_data_tmp.shape
    
    newCi = int(Ci / 4)
    newD = D
    newH = H
    newW = W
    newCo = Co * 4
    
    dst_data = np.zeros((newCi, newD, newH, newW, newCo), dtype = src_data.dtype)
    
    # 我们loop dst_data来fill它. 用src_data对应的value来fill每一个位置(实际上是每4个位置)
    for ci in range(newCi):       
        # dst_data有newC = C/4个group channels, 也即每次loop我们一次性从src_data提取4 channel数据.
        old_ci = 4*ci                # 找到当前group channel对应要提取的4个channel数据的starting channel location.
        # 当前loop, 要提取数据的channels是 [4*c, 4*c+1, 4*c+2, 4*c+3]
        for d in range(newD):
            old_d = d               # dst_data和src_data在depth维度上大小相同.
            for h in range(newH):
                old_h = h           # dst_data和src_data在height维度上大小相同.
                for w in range(newW):
                    old_w = w       # dst_data和src_data在width维度上大小相同.
                    for co in range(0, newCo, 4): # 我们每个location (d,h,w,co')存储4个channel的在这一location对应的数据
                        old_co = int(co/4)    # dst_data有newCo = Co*4, 也即Cout大小是src_data的4倍. 
                        # 我们除以4以找到对应的src_data Cout location.
                        # 每次loop, 每个location (d,h,w,co'), co'=co:co+4, 我们一次性从src_data对应位置提取4 channel数据.
                        dst_data[ci][d][h][w][co:co+4] = src_data_tmp[old_d][old_h][old_w][old_co][old_ci:old_ci+4]
    
    return dst_data



if __name__ == '__main__':
    
    depth = 2
    height = 5
    width = 3
    channel = 8
    
    src_data = np.arange(depth*height*width*channel).reshape(depth, height, width, channel)
    
    dst_data = DHWC_to_CgDHWC4(src_data)
    
    
    Cout = 1
    Cin = 4
    Dk = 2
    Hk = 2
    Wk = 2
    
    size = Cout*Cin*Dk*Hk*Wk
    
    src_data = np.arange(size).reshape(Dk, Hk, Wk, Cin, Cout)
    
    dst_data = DHWCiCo_to_CgiDHWCoCi4(src_data)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    