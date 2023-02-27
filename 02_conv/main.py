# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:33:20 2023

@author: gcd_c
"""

import numpy as np
import tensorflow as tf

import utility

############################## 3D convolution #####################################################################

################## Initialize inputs and weights ##################################################################
np.random.seed(64)

Nin = 1

Din = 3
Hin = 5
Win = 5
Cin = 4

Dk = 3
Hk = 3
Wk = 3
Cout = 54

inputs = np.random.randn(Din, Hin, Win, Cin)        # inputs layout: Din x Hin x Win x Cin
filters = np.random.randn(Dk, Hk, Wk, Cin, Cout)    # filters layout: Dk x Hk x Wk x Cin x Cout
biases = np.random.randn(Cout)                      # biases layout: Cout

tf_strides = [1, 1, 1, 1, 1]            # batch x depth x height x width x in_channels. strides[batch] and strides[in_channels] must be 1.
padding_mode = 'VALID'                   # 'VALID' or 'SAME'
data_format = 'NDHWC'
tf_dilations = [1, 1, 1, 1, 1]          # batch x depth x height x width x in_channels. dilations[batch], dilations[depth] and dilations[in_chnanels] must be 1.

in_shape = (Din, Hin, Win)
kernel_shape = (Dk, Hk, Wk)
strides = tuple(tf_strides[1:4])        # depth x height x width                   
dilations = tuple(tf_dilations[1:4])    # depth x height x width
paddings = utility.calculate_padding_size(in_shape, kernel_shape, strides, dilations, padding_mode) # depth x height x width

Dout = int((Din + 2*paddings[0] - (dilations[0]*(Dk - 1) + 1))/strides[0]) + 1
Hout = int((Hin + 2*paddings[1] - (dilations[1]*(Hk - 1) + 1))/strides[1]) + 1
Wout = int((Win + 2*paddings[2] - (dilations[2]*(Wk - 1) + 1))/strides[2]) + 1
out_shape = (Dout, Hout, Wout)

################## tensorflow conv3d ####################################################################################
tf_inputs = tf.convert_to_tensor(inputs.reshape(Nin, Din, Hin, Win, Cin))
tf_filters = tf.convert_to_tensor(filters)  # filters = np.random.randn(Dk, Hk, Wk, Cin, Cout) 

# tf_inputs  -> Nin x Din x Hin x Win x Cin -> batch x in_depth x in_height x in_width x in_channels
# tf_filters -> Dk x Hk x Wk x Cin x Cout   -> filter_depth x filter_height x filter_width x in_channels x out_channels
tf_outputs = tf.nn.conv3d(tf_inputs, tf_filters, tf_strides, padding_mode, data_format, tf_dilations)
tf_outputs = tf.nn.bias_add(tf_outputs, biases)
tf_outputs = tf_outputs.numpy().reshape(Dout, Hout, Wout, Cout).astype(np.float32)

################## self-implementation conv3d ###########################################################################
from converter import DHWC_to_CgDHWC4, DHWCiCo_to_CgiDHWCoCi4, CgDHWC4_to_DHWC
from image1d import Image1D
from image2d import Image2D
from image2darray import Image2DArray
from selfconv3d import SelfConv3D

# align the inputs with input channels multiple of 4
aligned_inputs = utility.align_array(inputs, align_base = (1, 1, 1, 4))
# Din x Hin x Win x Cin -> Cgin x Din x Hin x Win*4cin
flat_inputs = DHWC_to_CgDHWC4(aligned_inputs).flatten()

# align the filters with input channels and output channels of multiple of 4
aligned_filters = utility.align_array(filters, align_base = (1, 1, 1, 4, 4))
# Dk x Hk x Wk x Cin x Cout -> Cgin x Dk x Hk x Wk x Cout*4cin
flat_filters = DHWCiCo_to_CgiDHWCoCi4(aligned_filters).flatten()
Cain, Caout = aligned_filters.shape[-2:]

# align the biases with output channels of multiple of 4
aligned_biases = utility.align_array(biases, align_base = (4,))

# input image2darray: Cgin x Din x Hin x Win*4cin -> Cgin*Din x Hin x Win
img_input_width = Win
img_input_height = Hin
img_input_depth = int(Cain/4) * Din
img_input_shape = (img_input_width, img_input_height, img_input_depth)
img2darr_inputs = Image2DArray(shape = img_input_shape, data = flat_inputs)      

# filters image2d: Cgin x Dk x Hk x Wk x Cout*4cin -> Cgin*Dk*Hk*Wk x Cout
img_filters_width = Caout
img_filters_height = int(Cain/4) * Dk * Hk * Wk
img_filters_shape = (img_filters_width, img_filters_height)
img2d_filters = Image2D(shape = img_filters_shape, data = flat_filters)  

# biases image1d: Cout*4cout -> Cout
img_biases_width = int(Caout/4)
img1d_biases = Image1D(width = img_biases_width, data = aligned_biases)

input_shape = (Din, Hin, Win, Cain)
#output_shape = (Dout, Hout, Wout, Cout)
filter_shape = (Dk, Hk, Wk)
num_filters = Caout

"""
image2darray_outputs = SelfConv3D(img2darr_inputs, img2d_filters, img1d_biases,
                                  input_shape, filter_shape, num_filters,
                                  strides, paddings, dilations)
flat_image2darray_outputs = image2darray_outputs.to_numpy().flatten()

aligned_self_outputs = CgDHWC4_to_DHWC(image2darray_outputs.to_numpy(shape = (int(Caout/4), Dout, Hout, Wout*4)))

target_output_shape = (Dout, Hout, Wout, Cout)
self_outputs = utility.extract_array(aligned_self_outputs, target_output_shape)
"""
###################### save data to file ###########################################################################
# aligned_inputs: Din x Hin x Win x Cain
# flat_inputs = DHWC_to_CgDHWC4(aligned_inputs).flatten()
float_flat_inputs = flat_inputs.astype(np.float32)
float_flat_inputs.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\input.raw")

# aligned_filters: Dk x Hk x Wk x Cain x Caout
# flat_filters = DHWCiCo_to_CgiDHWCoCi4(aligned_filters).flatten()
float_flat_filters = flat_filters.astype(np.float32)
float_flat_filters.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\filters.raw")

# aligned_biases: Caout
float_flat_biases = aligned_biases.astype(np.float32)
float_flat_biases.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\biases.raw")

"""
# image2darray_outputs: Cgout*Dout x Hout x Wout*4cout
flat_image2darray_outputs = image2darray_outputs.to_numpy().flatten()
float_flat_image2darray_outputs = flat_image2darray_outputs.astype(np.float32)
float_flat_biases.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\groundtruth_outputs.raw")
"""

# tf_outputs: Dout x Hout x Wout x Cout
aligned_tf_outputs = utility.align_array(tf_outputs, align_base = (1, 1, 1, 4))
# CgDHWC4_tf_outputs: Cgout x Dout x Hout x Wout*4cout
CgDHWC4_tf_outputs = DHWC_to_CgDHWC4(aligned_tf_outputs)
# flat_tf_outputs: Cgout*Dout*Hout*Wout*4cout
flat_tf_outputs = DHWC_to_CgDHWC4(aligned_tf_outputs).flatten()
float_flat_tf_outputs = flat_tf_outputs.astype(np.float32)
float_flat_tf_outputs.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\tf_output.raw")

aligned_input_shape = list(input_shape)                 # Din x Hin x Win x Cain
aligned_output_shape = [Dout, Hout, Wout, Caout]        # Dout x Hout x Wout x Caout
aligned_filter_shape = list(filter_shape[::-1])         # original Dk x Hk x Wk, transform to (X, Y, Z) -> (Wk, Hk, Dk)
aligned_strides = list(strides[::-1])                   # original Sd x Sh x Sw, transform to (X, Y, Z) -> (Sw, Sh, Sd) 
aligned_paddings = [int(val) for val in paddings[::-1]] # original Pd x Ph x Pw, transform to (X, Y, Z) -> (Pw, Ph, Pd) 
aligned_dilations = list(dilations[::-1])               # original Dd x Dh x Dw, transform to (X, Y, Z) -> (Dw, Dh, Dd)              
parameters = np.array(aligned_input_shape + aligned_output_shape + aligned_filter_shape + aligned_strides + aligned_paddings + aligned_dilations)
parameters.tofile(r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data\parameters.raw")

###################### reuslt validation ###########################################################################

#print("tf_outputs\n", tf_outputs[0][0])
print()
#print("self_outputs\n", self_outputs[0][0])
print()

"""
flat_final_tf_outputs = tf_outputs.ravel().astype(np.float32)
flat_final_self_outputs = self_outputs.ravel().astype(np.float32)

D, H, W, C = tf_outputs.shape

def validation_results_v1(tf_outputs, self_outputs):
    for d in range(D):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    diff = abs(tf_outputs[d][h][w][c] - self_outputs[d][h][w][c])
                    ans = "correct" if diff < 0.001 else "wrong"
                    print(ans, "({},{},{},{})".format(d,h,w,c), tf_outputs[d][h][w][c], self_outputs[d][h][w][c], diff)
    

def validation_results_v2(tf_outputs, self_outputs):
    results = np.abs(tf_outputs - self_outputs) < 0.001
    
    print("tf_outputs == self_outputs:", np.sum(results).astype(int) == tf_outputs.size)

def validation_results_v3(tf_outputs, self_outputs):
    for d in range(D):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    diff = abs(tf_outputs[d][h][w][c] - self_outputs[d][h][w][c])
                    ans = "correct" if diff < 0.001 else "wrong"
                    if diff > 0.001:    
                        print(ans, "({},{},{},{})".format(d,h,w,c), tf_outputs[d][h][w][c], self_outputs[d][h][w][c], diff)
    

validation_results_v1(tf_outputs, self_outputs)
validation_results_v2(tf_outputs, self_outputs)
"""







