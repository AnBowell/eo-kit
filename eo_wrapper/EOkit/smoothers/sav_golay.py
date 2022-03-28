import numpy as np
from EOkit.EOkit import lib
from EOkit.array_utils import check_type, check_contig
from cffi import FFI


ffi = FFI()


def single_sav_golay(y_input, window_size, order, deriv=0, delta=1):

    # TODO! Condense all this stuff into a function accross the smoothers.
    data_len = len(y_input)

    result = np.empty(data_len, dtype=np.float64)
    result = check_contig(result)

    y_input = check_contig(y_input)

    y_input_ptr = ffi.cast("double *", y_input.ctypes.data)

    result_ptr = ffi.cast("double *", result.ctypes.data)

    lib.rust_single_sav_golay(
        y_input_ptr, result_ptr, result.size, window_size, order, deriv, delta
    )

    return result


def multiple_sav_golay(y_inputs, window_size, order, deriv=0, delta=1, n_threads=-1):
    pass
