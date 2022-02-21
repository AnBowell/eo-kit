import numpy as np
from EOkit.EOkit import lib
from EOkit.array_utils import (check_type, check_contig)
from cffi import FFI
ffi = FFI()



def single_whittaker(y_input, weights_input, lambda_, d):


    data_len = len(y_input)
    
    result = np.empty(data_len, dtype=np.float64)
    result = check_contig(result)
    
    y_input = check_contig(y_input)
    weights_input = check_contig(weights_input)

    y_input = check_type(y_input)
    weights_input = check_type(weights_input)
    
    y_input_ptr = ffi.cast("double *", y_input.ctypes.data)
    weights_input_ptr = ffi.cast("double *", weights_input.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data) 
    
           
    lib.rust_single_whittaker(
        y_input_ptr,
        weights_input_ptr,
        result_ptr,
        result.size,
        lambda_,
        d,
    )

    return result



def multiple_whittakers(y_inputs, weights_inputs, lambda_, d):
    
    
    index_runner = 0

    start_indices = [0]

    for y_input in y_inputs[:-1]:
        length_of_input = len(y_input)
        index_runner += length_of_input
        start_indices.append(index_runner)
        
    start_indices = np.array(start_indices, dtype=np.uint64)
    
    y_input_array = np.concatenate(y_inputs).ravel().astype(np.float64)
    weight_input_array = np.concatenate(weights_inputs).ravel().astype(np.float64)
    
    result = np.empty(y_input_array.size,dtype=np.float64)
    
    
    y_input_array = check_contig(y_input_array)
    weight_input_array = check_contig(weight_input_array)
    result = check_contig(result)
    start_indices = check_contig(start_indices)

    y_input_ptr = ffi.cast("double *", y_input_array.ctypes.data)
    weights_input_ptr = ffi.cast("double *", weight_input_array.ctypes.data)
    result_ptr = ffi.cast("double *", result.ctypes.data) 
    start_indices_ptr = ffi.cast("uintptr_t *", start_indices.ctypes.data)


    lib.rust_multiple_whittakers(
        y_input_ptr,
        weights_input_ptr,
        start_indices_ptr,
        start_indices.size,
        result_ptr,
        result.size,
        lambda_,
        d,
    
    )

    
    results = []
    
    for i in range(0, len(start_indices)):
        
        if i + 1 >= len(start_indices):

            single_result = result[start_indices[int(i)] :]
        else:

            single_result = result[start_indices[int(i)] : int(start_indices[i + 1])]

        results.append(single_result)
     
    return results




    

