# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 17:49:25 2023

@author: gcd_c
"""

import numpy as np

from image2d import Image2D
from image2darray import Image2DArray

def SelfConv3D(image2darray_inputs,         # 3D Image2DArray: D x H x W -> Cgin*Din x Hin x Win x Win*4cin
               image2d_filters,             # 2D Image2D: H x W*4 -> Cgin*Dk*Hk*Wk x Cout*4cin
               image1d_biases,              # 1D Image1D: W*4 -> Cgout*4cout
               input_shape,                 # 4D DHWC -> Din x Hin x Win x Cin
               #output_shape,               # 4D DHWC -> Dout x Hout x Wout x Cout
               filter_shape,                # 3D DHW  -> Dk x Hk x Wk
               num_filters,                 # 1D C    -> Cout
               strides,                     # 3D DHW  -> strideD x strideH x strideW
               paddings,                    # 3D DHW  -> paddingsD x paddingsH x paddingsW
               dilations):                  # 3D DHW  -> dilationsD x dilationsH x dilationsW
    
    inputs = image2darray_inputs
    filters = image2d_filters
    biases = image1d_biases
    
    Din, Hin, Win, Cin = input_shape
    Dk, Hk, Wk = filter_shape
    Cout = num_filters
    
    Dout = int((Din + 2*paddings[0] - (dilations[0]*(Dk - 1) + 1))/strides[0]) + 1
    Hout = int((Hin + 2*paddings[1] - (dilations[1]*(Hk - 1) + 1))/strides[1]) + 1
    Wout = int((Win + 2*paddings[2] - (dilations[2]*(Wk - 1) + 1))/strides[2]) + 1
    output_shape = (Dout, Hout, Wout, Cout)    
    
    """
    Dout, Hout, Wout, Cout = output_shape
    Dk, Hk, Wk = filter_shape
    Cgin = int(Cin/4) 
    Cgout = int(Cout/4)
    
    strideX, strideY, strideZ = strides
    padX, padY, padZ = paddings
    dilationX, dilationY, dilationZ = dilations
    """
    
    # outputs: Dout x Hout x Wout x Cout -> Cgout=Cout/4 x Dout x Hout x Wout*4cout
    # image2darray: Cgout x Dout x Hout x Wout*4cout -> Cgout*Dout x Hout x Wout
    img2darr_output_width = Wout
    img2darr_output_height = Hout
    img2darr_output_depth = int(Cout/4) * Dout
    img2darr_output_shape = (img2darr_output_width, img2darr_output_height, img2darr_output_depth)
    outputs = Image2DArray(shape = img2darr_output_shape)
    
    for d in range(Dout):
        for h in range(Hout):
            for w in range(Wout):
                out_loc = (d, h, w)
                SelfConv3D_helper(inputs, filters, biases, outputs, 
                                  input_shape, output_shape, filter_shape, 
                                  strides, paddings, dilations, out_loc)
    
    return outputs


def SelfConv3D_helper(inputs,               # image2dArray: Cgin*Din x Hin x Win*4cin
                      filters,              # image2d     : Cgin*Dk*Hk*Wk x Cout*4cin
                      biases,               # image1d     : Cgout*4cout
                      outputs,              # image2dArray: Cgout*Dout x Hout x Wout*4cout
                      input_shape,
                      output_shape,
                      filter_shape,
                      strides,
                      paddings,
                      dilations,
                      out_loc):

    Din, _, _, Cin = input_shape
    Dout, Hout, Wout, Cout = output_shape
    Dk, Hk, Wk = filter_shape
    Cgin = int(Cin / 4)
    Cgout = int(Cout / 4)
    # remember coordinate mapping: (x, y, z) -> (w, h, d) or (D, H, W) -> (Z, Y, X)
    strideX, strideY, strideZ = strides[::-1]
    padX, padY, padZ = paddings[::-1]
    dilationX, dilationY, dilationZ = dilations[::-1]
    
    # Each work-item will work on Cout conv3D output points, one output point per one output channel.
    # To be clear, the output points that current work-item work on, will have the exactly same (Dout, Hout, Wout)
    # coordinates but different in output channel.
    # Also, the input patch that the work-item uses for conv3D operation is exactly the same for all output points,
    # just different in output filter.
    
    # Get the current work-item location in host-output (Dout, Hout, Wout, Cout)
    outputX, outputY, outputZ = out_loc[::-1]   
    # Calculate the Conv3D starting location (x = win, y = hin, z = din) in host-input
    inputX = outputX * strideX - int(padX)      # padding_mode = 'SAME', the padding value can be decimal!
    inputY = outputY * strideY - int(padY)      # the paddings we derived will exactly reflect the target value
    inputZ = outputZ * strideZ - int(padZ)      
    # Here we only need p_left, that is we alwasy assume (p_left + 1) = p_right and p_left + p_right = 2*Pvalue.
    # Thus, we take the floor(Pvalue) to get p_left. p_left = floor(Pvalue) = int(Pvalue)

    # Figure out the output coordinate on output image2darray, in which we will store the conv3D results there.
    # Remember, output points that current work-item work on have the exactly same (Dout, Hout, Wout) coordinates 
    # but different in output channel.
    # output image2darray: Cgout*Dout x Hout x Wout*4cout
    image2dArrayOutputX = outputX                   # outputX -> [0, Wout - 1]
    image2dArrayOutputY = outputY                   # outputY -> [0, Hout - 1]
    image2dArrayOutputZ = None                      # outputZ -> [0, Dout - 1]
    # work-item works on Cout output points form Cout output channels, 
    # thus image2dArrayOutputZ varies with differnet output channel.
    
    # Figure out the starting coordinate on input image2darray, in which we obtain the input patch for conv3D.
    # input image2darray: Cgin*Din x Hin x Win*4cin
    image2dArrayInputStartX = inputX        # inputX -> [0, Win - 1]
    image2dArrayInputStartY = inputY        # inputY -> [0, Hin - 1]
    image2dArrayInputStartZ = None          # inputZ -> [0, Din - 1]
    
    # Figure out the target bias in image1d-biases. 
    # According to the 1D image layout we select, there are Cgout group biases.
    img1dBx = None     
    # img1dBx (img1dBx in [0, Cgout - 1]) varies by different group filter selected. We determine it in for loop. 
    
    # 1 work-item is responsible for Cout = Cgout*4 output points, i.e. Cgout output pixels.
    # We loop all grouped output channels, and each group output channels we calculate 1 output pixel (i.e. 4 output points).
    for cog in range(Cgout):
        res = np.zeros(4)   # Initialize the 4 output points in 1 output pixel.
        
        # Figure out the output coordinate on output image2darray, in which we will store the conv3D results there.
        # image2dArrayOutputZ varies with differnet output channel, thus we calculate it here inside Cgout loop.
        image2dArrayOutputZ = cog * Dout + outputZ      # outputZ -> [0, Dout - 1]

        # Figure out the bias X coordinate in image1d-biases.
        # bias X coordinate in image1d-biases is exactly the same of the group filter selected.
        img1dBx = cog 
        
        # 2-step conv3d for 4 output points: 1. group-channel-level conv3D; 2. data-point-level conv3D results add up.
        # group-channel-level conv3D, each output point needs a vector4 to store conv3D results.
        conv3dValues = np.zeros((4, 4))     
        
        # perform conv3d operation for each output point between the same target input patch but different target filter
        for idx in range(4):
            # Pick up the target filter for conv3d in this run.
            # filters image2d: H x W -> Cgin*Dk*Hk*Wk x Cout*4cin. i.e. every x/column corresponds to a filter.
            image2dFilterCurrentX = cog*4 + idx
            
            # loop the input patch and filter for conv3D.
            # inputs image2dArray: Cgin*Din x Hin x Win*4cin
            for cig in range(Cgin): 
                # Figure out the starting coordinate on input image2darray, in which we obtain the input patch for conv3D.
                # image2dArrayInputStartZ varies with differnet input channel, thus we calculate it here inside Cgin loop.
                # input image2darray: Cgin*Din x Hin x Win*4cin
                image2dArrayInputStartZ = cig * Din + inputZ
                
                # Figure out the starting coordinate on filter image2d, in which we obtain the filter patch for conv3D
                # filter image2d: Cgin*Dk*Hk*Wk x Cout*4cin
                image2dFilterStartY = cig * Dk * Hk * Wk
                
                # loop all points in selected filter channel and input channel patch for conv3d operation.
                for kz in range(Dk):   # filter_shape = (Dk, Hk, Wk)
                    image2dArrayInputCurrentZ = image2dArrayInputStartZ + kz * dilationZ
                    if image2dArrayInputCurrentZ < cig * Din or image2dArrayInputCurrentZ >= (cig+1)*Din:
                        continue
                    for ky in range(Hk):
                        image2dArrayInputCurrentY = image2dArrayInputStartY + ky * dilationY
                        for kx in range(Wk):
                            image2dArrayInputCurrentX = image2dArrayInputStartX + kx * dilationX
                            image2dFilterCurrentY = image2dFilterStartY + kz * Hk * Wk + ky * Wk + kx
                            
                            inputCoord = (image2dArrayInputCurrentX, image2dArrayInputCurrentY, image2dArrayInputCurrentZ)
                            filterCoord = (image2dFilterCurrentX, image2dFilterCurrentY)
                            
                            inputValue = inputs.read_image(coord = inputCoord)
                            filterValue = filters.read_image(coord = filterCoord)
                            
                            conv3dValues[idx] += inputValue * filterValue
                            
                            #if outputX == 0 and outputY == 0 and outputZ == 0 and cog == 0 and idx == 0:
                            #    print("output=({},{},{},{})".format(outputZ, outputY, outputX, cog),  "input=({},{},{})".format(image2dArrayInputCurrentX, image2dArrayInputCurrentY, image2dArrayInputCurrentZ), inputValue)
                            #    print("output=({},{},{},{})".format(outputZ, outputY, outputX, cog),  "filter=({},{})".format(image2dFilterCurrentX, image2dFilterCurrentY), filterValue)

                            
                                #print("output=({},{},{},{})".format(outputZ, outputY, outputX, cog),  "input=({},{},{})".format(inputZ, inputY, inputX), inputValue)
        
        # sum up all group channel conv results to get the single output point convolution results.
        res = conv3dValues.sum(axis = 1)
        
        # also add the bias
        biasValue = biases.read_image(coord = img1dBx)
        res += biasValue
        
        outputCoord = (image2dArrayOutputX, image2dArrayOutputY, image2dArrayOutputZ)
        outputs.write_image(coord = outputCoord, value = res)
        
        #print("output=({},{},{},{})".format(outputZ, outputY, outputX, cog),  "input=({},{},{})".format(inputZ, inputY, inputX), res)
        


































