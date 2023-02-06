# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 12:33:20 2023

@author: gcd_c
"""

import numpy as np

import tensorflow as tf

############################## 3D convolution #####################################################################

################## Initialize inputs and weights ##################################################################
np.random.seed(32)

Nin = 1

Din = 16
Hin = 9
Win = 9
Cin = 16

Dk = 3
Hk = 3
Wk = 3
Cout = 4

inputs = np.random.randn(Din, Hin, Win, Cin)        # inputs layout: Din x Hin x Win x Cin
filters = np.random.randn(Dk, Hk, Wk, Cin, Cout)    # filters layout: Dk x Hk x Wk x Cin x Cout

tf_strides = [1, 1, 1, 1, 1]            # batch x depth x height x width x in_channels. strides[batch] and strides[in_channels] must be 1.
#padding = [0, 0, 0, 0, 0]              # i.e. no padding 'VALID'.
data_format = 'NDHWC'
tf_dilations = [1, 1, 1, 1, 1]          # batch x depth x height x width x in_channels. dilations[batch], dilations[depth] and dilations[in_chnanels] must be 1.

strides = tuple(tf_strides[1:4])        # depth x height x width 
paddings = (0, 0, 0)                    # depth x height x width
dilations = tuple(tf_dilations[1:4])    # depth x height x width

Dout = int((Din + 2*paddings[0] - (dilations[0]*(Dk - 1) + 1))/strides[0]) + 1
Hout = int((Hin + 2*paddings[1] - (dilations[1]*(Hk - 1) + 1))/strides[1]) + 1
Wout = int((Win + 2*paddings[2] - (dilations[2]*(Wk - 1) + 1))/strides[2]) + 1

################## tensorflow conv3d ####################################################################################
tf_inputs = tf.convert_to_tensor(inputs.reshape(Nin, Din, Hin, Win, Cin))
tf_filters = tf.convert_to_tensor(filters)  # filters = np.random.randn(Dk, Hk, Wk, Cin, Cout) 

# tf_inputs  -> Nin x Din x Hin x Win x Cin -> batch x in_depth x in_height x in_width x in_channels
# tf_filters -> Dk x Hk x Wk x Cin x Cout   -> filter_depth x filter_height x filter_width x in_channels x out_channels
tf_outputs = tf.nn.conv3d(tf_inputs, tf_filters, tf_strides, 'VALID', data_format, tf_dilations)
tf_outputs = tf_outputs.numpy().reshape(Dout, Hout, Wout, Cout)

################## self-implementation conv3d ###########################################################################
from converter import DHWC_to_CgDHWC4, DHWCiCo_to_CgiDHWCoCi4, CgDHWC4_to_DHWC
from image2d import Image2D
from image2darray import Image2DArray
from selfconv3d import SelfConv3D

# Din x Hin x Win x Cin -> Cgin x Din x Hin x Win*4cin
flat_inputs = DHWC_to_CgDHWC4(inputs).flatten()
# Dk x Hk x Wk x Cin x Cout -> Cgin x Dk x Hk x Wk x Cout*4cin
flat_filters = DHWCiCo_to_CgiDHWCoCi4(filters).flatten()

# input image2darray: Cgin x Din x Hin x Win*4cin -> Cgin*Din x Hin x Win
img_input_width = Win
img_input_height = Hin
img_input_depth = int(Cin/4) * Din
img_input_shape = (img_input_width, img_input_height, img_input_depth)
img2darr_inputs = Image2DArray(shape = img_input_shape, data = flat_inputs)      

# filters image2d: Cgin x Dk x Hk x Wk x Cout*4cin -> Cgin*Dk*Hk*Wk x Cout
img_filters_width = Cout
img_filters_height = int(Cin/4) * Dk * Hk * Wk
img_filters_shape = (img_filters_width, img_filters_height)
img2d_filters = Image2D(shape = img_filters_shape, data = flat_filters)  

"""
def SelfConv3D(image2darray_inputs,         # 3D Image2DArray: D x H x W -> Cgin*Din x Hin x Win x Win*4cin
               image2d_filters,             # 2D Image2D: H x W*4 -> Cgin*Dk*Hk*Wk x Cout*4cin
               input_shape,                 # 4D DHWC -> Din x Hin x Win x Cin
               #output_shape,               # 4D DHWC -> Dout x Hout x Wout x Cout
               filter_shape,                # 3D DHW  -> Dk x Hk x Wk
               num_filters,                 # 1D C    -> Cout
               strides,                     # 3D DHW  -> strideD x strideH x strideW
               paddings,                    # 3D DHW  -> paddingsD x paddingsH x paddingsW
               dilations):                  # 3D DHW  -> dilationsD x dilationsH x dilationsW
"""
input_shape = (Din, Hin, Win, Cin)
#output_shape = (Dout, Hout, Wout, Cout)
filter_shape = (Dk, Hk, Wk)
num_filters = Cout

image2darray_outputs = SelfConv3D(img2darr_inputs, img2d_filters,
                                  input_shape, filter_shape, num_filters,
                                  strides, paddings, dilations)

self_outputs = CgDHWC4_to_DHWC(image2darray_outputs.to_numpy(shape = (int(Cout/4), Dout, Hout, Wout*4)))






















