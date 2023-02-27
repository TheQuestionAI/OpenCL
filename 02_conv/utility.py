# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 12:31:19 2023

@author: gcd_c
"""
import numpy as np

# input_shape、filter_shape、strides、dilations must be in SAME shape!
def calculate_padding_size(input_shape, filter_shape, strides, dilations, padding_mode):
    ############################## validaton checking ##########################################
    assert padding_mode in ['VALID', 'SAME'], "Currently only supports two type of padding mode: VALID and SAME!"
    
    types = (list, tuple, np.ndarray)
    assert isinstance(input_shape, types), "input_shape type is not valid!"
    assert isinstance(filter_shape, types), "filter_shape type is not valid!"
    assert isinstance(strides, types), "stride type is not valid!"
    assert isinstance(dilations, types), "dilations type is not valid!"
    
    assert all(x == len(input_shape) for x in [len(input_shape), len(filter_shape), len(strides), len(dilations)]), \
    "input_shape/filter_shape/strides/dilations size is not the same!"
    
    for val1, val2, val3, val4 in zip(input_shape, filter_shape, strides, dilations):
        assert isinstance(val1, int) and val1 > 0, "input_shape data type is not valid!"
        assert isinstance(val2, int) and val2 > 0, "filter_shape data type is not valid!"
        assert isinstance(val3, int) and val3 > 0, "stride data type is not valid!"
        assert isinstance(val4, int) and val4 > 0, "dilations data type is not valid!"
    ############################################################################################
    
    #################################### main codes ############################################
    """
    padding -> 值图像边缘image boundary填充像素的模式.
    (1) VALID padding: VALID填充模式, 实际上是不进行任何边缘像素填充操作no padding operation. pad = [0]*len(input_shape).
                       卷积计算只使用original input image, 并且不允许卷积核filter在和input image进行卷积计算时超出input 
                       image的边界. 也即original input image多余的没法进行卷积计算的部分将被直接忽略. VALID模式下进行的
                       卷积计算输出的output image的大小必然小于input image大小.
    (2) SAME padding: SAME填充模式, 是对input image的边界进行扩充填充, 使得当stride = 1时, original input image上的
                      所有点都能够作为卷积计算中心点, 如此Lout = Lin即output_image和input_image大小完全相同; 当stride > 1
                      时, 由计算公式Lout = ceil(Lin / S)确定output image的shape.
                      
    不同padding模式下, Lout 和 paddings的取值.
    (1) VALID padding: 此时paddings = [0]*len(input_shape), 因此快速可知Lout = floor((Lin - K') / S) + 1
                                                                         K' = D*(K-1) + 1
    (2) SAME padding: 依据计算公式   Lout = ceil(Lin / S)
                                    Lout = floor((Lin + 2P - K') / S) + 1
                      反推出padding size.
                      floor()是向下取整函数, 所以即使Lin已知, 因为floor()的存在我们想要求的P也会有多个解.
                      也即可以有多个P的取值满足上面两个计算公式, 因此我们可以选择最简单最易于计算的case,
                      也即 Lin + 2P - K'能够整除S的case, 即(Lin + 2P - K') % S = 0, 
                      则有 Lout = (Lout - 2P + K') / S + 1     =>     (Lout - 1)*S = Lin + 2P - K'   
                      => 2P = (Lout - 1)*S + K' - Lin
                      => P = ( (Lout - 1)*S + K' - Lin ) / 2
                      这里就存在除以2不是整数的情况.
                      case 1) 能够整除2     -> 则显然 p_left = p_right = ( (Lout - 1)*S + K' - Lin ) // 2
                      case 2) 不能整除2     -> 则 p_left = ( (Lout - 1)*S + K' - Lin ) // 2
                                                 p_right = ( (Lout - 1)*S + K' - Lin ) // 2 + 1  
    """
    paddings = []
    
    if padding_mode == 'VALID':
        paddings = [0]*len(input_shape)
    
    if padding_mode == 'SAME':
    
        for Lin, K, S, D in zip(input_shape, filter_shape, strides, dilations):
            # 使用公式Lout = ceil(Lin / S)计算Lout
            Lout = int(np.ceil(Lin / S))
            # 计算dilated kernel size.
            DD = D*(K - 1) + 1
            # 使用公式 P = ( (Lout - 1)*S + K' - Lin ) // 2 计算padding size
            #pad = int( ((Lout - 1)*S + DD - Lin) // 2)
            pad = ((Lout - 1)*S + DD - Lin) / 2     # 这里我们算pad, 存在不整除2的情形. 我们不做判断, 精确返回结果.
            
            paddings.append(pad)
        
    return paddings


def align_array(src_array, align_base, pad_value = 0):
    ############################## validaton checking ##########################################
    assert isinstance(src_array, np.ndarray) , "Currently only supports align the np.ndarray!"
    
    types = (list, tuple, np.ndarray)
    assert isinstance(align_base, types), "align_base type is not valid!"
    
    assert len(src_array.shape) == len(align_base), "scr_array and align_base must be the same dimension!"
    assert all([isinstance(val, int) for val in align_base]), "align_array data type is not valid!"
    assert all(np.array(align_base) > 0), "align_array data value is not valid!"
    
    #################################### main codes ############################################
    src_array_shape = src_array.shape
    aligned_dims = tuple(int((src_dim + align_dim - 1) / align_dim) * align_dim \
                         for src_dim, align_dim in zip(src_array_shape, align_base))
    aligned_src_array = np.zeros(aligned_dims)
    aligned_src_array.fill(pad_value)
    aligned_src_array = aligned_src_array.astype(src_array.dtype)
    
    _recursive_fill_align_array(aligned_src_array, src_array)
    
    return aligned_src_array

def _recursive_fill_align_array(aligned_src_array, src_array):
    dim1 = src_array.shape[0]
    if len(aligned_src_array.shape) == 1:   
        aligned_src_array[:dim1] = src_array
    else:
        for idx in range(dim1):     # dfs递归
            _recursive_fill_align_array(aligned_src_array[idx], src_array[idx])
            
    
def extract_array(src_array, target_shape):
    ############################## validaton checking ##########################################
    assert isinstance(src_array, np.ndarray) , "Currently only supports align the np.ndarray!"
    
    types = (list, tuple)
    assert isinstance(target_shape, types), "target_shape type is not valid!"
    
    assert len(src_array.shape) == len(target_shape), "scr_array and target_shape must be the same dimension!"
    assert all([isinstance(val, int) for val in target_shape]), "target_shape data type is not valid!"
    assert all(np.array(target_shape) > 0), "target_shape data value is not valid!"
    assert all([target_dim <= src_dim for src_dim, target_dim in zip(src_array.shape, target_shape)]), "target_shape must within the shape of src_array!"

    #################################### main codes ############################################    
    extracted_src_array = np.zeros(target_shape).astype(src_array.dtype)
    
    _recursive_extract_array(extracted_src_array, src_array)
    
    return extracted_src_array

def _recursive_extract_array(extracted_src_array, src_array):
    dim1 = extracted_src_array.shape[0]
    if len(extracted_src_array.shape) == 1:   
        extracted_src_array[:] = src_array[:dim1]
    else:
        for idx in range(dim1):     # dfs递归
            _recursive_extract_array(extracted_src_array[idx], src_array[idx])
    

if __name__ == '__main__':
    
    """
    input_shape = (32, 32)
    filter_shape = (13, 13)
    strides = (1, 1)
    dilations = (2, 2)
    
    paddings = calculate_padding_size(input_shape, filter_shape, strides, dilations, padding_mode = "SAME")

    output_shape = []
    for Lin, K, P, S, D in zip(input_shape, filter_shape, paddings, strides, dilations):
        DD = D*(K - 1) + 1
        Lout = int((Lin + 2*P -DD ) / S) + 1
        output_shape.append(Lout)
    output_shape = tuple(output_shape)
    
    print("input_shape:", input_shape)
    print("paddings:", paddings)
    print("output_shape:", output_shape)
    """
    
    """
    input_shape = (15, 13, 14, 12)
    src_array = np.random.randn(*input_shape)
    align_base = (2, 4, 5, 7)
    """
    
    input_shape = (7, 6)
    src_array = np.random.randn(*input_shape)
    align_base = (2, 4)

    """
    src_array = np.random.randn(7, 6, 5)
    align_base = (2, 4, 3)
    """
    
    pad_value = 0
    
    aligned_src_array = align_array(src_array, align_base)
    
    dst_array = extract_array(aligned_src_array, input_shape)
    
    
    print("Is dst_array equal to src_array:", np.array_equal(src_array, dst_array))

    test_array = dst_array.flatten()[:10].astype(np.float32)
    
    file_dir = r"D:\Dropbox\OMSCS_Yangzi\CS textbook\Computer Graphics for Video Games\ClionOpenCL\OpenCL_Kernel_Test_Framework\data"
    file_name = r"test_data.raw"
    file_path = file_dir + "\\" + file_name
    test_array.tofile(file_path)












