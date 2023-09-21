
from cython import boundscheck, wraparound
from numba import njit

import numpy as np
from numpy import empty
from numpy import power, divide, add, multiply
from numpy import float32 as npfloat32

# from commons import *
import commons
from commons import _is_rfft2_size_argument_valid
from img_type import KER_SIZE, ARR_32F1D, ARR_32F2D


"""
Calculate 2-D Butterworth filter by product two 1-D Butterworth filter.

This method is 10x faster than standard method when order!=1.
An image after low-pass filter will become more smooth on edges.
"""


_SIGNATURE_BUTTERWORTH_KERNEL_SHIFT = [
    "float32[:](int64,float32,float32,float32)", 
    "float32[:](int64,float32,float32,Omitted(0))", 
]
@njit(
    _SIGNATURE_BUTTERWORTH_KERNEL_SHIFT, 
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_butterworth_kernel_shift(
        size: KER_SIZE, cutoff: float, order: float, center: float = 0
    ) -> ARR_32F1D:
    """
    Return an 1-D Butterworth kernel. This function is present for filtering a
    data after fft but not apply fftshift yet.
    """
    kernel = empty((size), dtype=npfloat32)
    const = power(cutoff, multiply(2, order))
    order2 = multiply(2, order)
    if (size % 2):
        quant = size//2 + 1
    else:
        quant = size//2
        dist = power(quant, order2)
        kernel[quant] = divide(const, add(const, dist))
    if (center == 0):
        for y in range(1, quant):
            dist = power(y, order2)
            val = divide(const, add(const, dist))
            kernel[y] = val
            kernel[-y] = val
        kernel[0] = 1
    else:
        for y in range(-quant+1, quant):
            dist = power(abs(y-center), order2)
            kernel[y] = divide(const, add(const, dist))
    return kernel


_SIGNATURE_BUTTERWORTH_KERNEL_HALF = [
    "float32[:](int64,float32,float32,float32,float32)", 
    "float32[:](int64,float32,float32,Omitted(0),float32)", 
    "float32[:](int64,float32,float32,float32,Omitted(1))", 
    "float32[:](int64,float32,float32,Omitted(0),Omitted(1))", 
]
@njit(
    _SIGNATURE_BUTTERWORTH_KERNEL_HALF, 
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_butterworth_kernel_half(
        size: KER_SIZE, cutoff_frequency: float, order: float,
        center: float = 0, c: float = 1
    ) -> ARR_32F1D:
    """
    Return an 1-D Butterworth kernel with positive frequency only. This
    function is present for filtering a data after rfft but not apply rfftshift
    yet.
    """
    kernel = empty((size), dtype=npfloat32)
    const = power(cutoff_frequency, multiply(2, order))
    order2 = multiply(2, order)
    if center:
        for y in range(size):
            dist = power(abs(y-center), order2)
            val = divide(const, add(const, dist))
            kernel[y] = multiply(c, val)
    else:
        for y in range(size):
            dist = power(y, order2)
            val = divide(const, add(const, dist))
            kernel[y] = multiply(c, val)
    return kernel


_SIGNATURE_BUTTERWORTH_LOWPASS = [
    "float32[:,:](UniTuple(int64, 2), float32, float32)", 
]
@njit(
    _SIGNATURE_BUTTERWORTH_LOWPASS, 
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def _approx_butterworth_lowpass_nonparallel(
        size: KER_SIZE, cutoff: float, order: float
    ) -> ARR_32F2D:
    """Return an approximate Butterworth low-pass filter."""
    mask_y = get_butterworth_kernel_shift(size[0], cutoff, order)
    mask_x = get_butterworth_kernel_half(size[1], cutoff, order)
    return np.outer(mask_y, mask_x)


@njit(
    _SIGNATURE_BUTTERWORTH_LOWPASS, 
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def _approx_butterworth_lowpass_parallel(
        size: KER_SIZE, cutoff: float, order: float
    ) -> ARR_32F2D:
    """Return an approximate Butterworth low-pass filter."""
    mask_y = get_butterworth_kernel_shift(size[0], cutoff, order)
    mask_x = get_butterworth_kernel_half(size[1], cutoff, order)
    return commons.nb_outer(mask_y, mask_x)


@boundscheck(False)
@wraparound(False)
def approx_butterworth_lowpass(
        size: KER_SIZE, cutoff: float, order: float,
        parallel: bool = False
    ) -> ARR_32F2D:
    """
    Return an approximate Butterworth low-pass filter.
    
    If parallel=True, then apply numba parallel computation.
    """
    if parallel:
        return _approx_butterworth_lowpass_parallel(
            size, cutoff, order)
    else:
        return _approx_butterworth_lowpass_nonparallel(
            size, cutoff, order)


@njit([
    "float32[:,:](UniTuple(int64,2),float32,float32)", 
], nogil=True, cache=True, fastmath=True)
@wraparound(False)
def _approx_butterworth_highpass_nonparallel(
        size: KER_SIZE, cutoff: float, order: float,
    ) -> ARR_32F2D:
    """Return an approximate Butterworth high-pass filter."""
    mask_y = get_butterworth_kernel_shift(size[0], cutoff, order)
    mask_x = get_butterworth_kernel_half(size[1], cutoff, order)
    return commons.outer_bias(mask_y, mask_x, 1)


@njit([
    "float32[:,:](UniTuple(int64,2),float32,float32)", 
], nogil=True, cache=True, fastmath=True)
@wraparound(False)
def _approx_butterworth_highpass_parallel(
        size: KER_SIZE, cutoff: float, order: float
    ) -> ARR_32F2D:
    """Return an approximate Butterworth high-pass filter."""
    mask_y = get_butterworth_kernel_shift(size[0], cutoff, order)
    mask_x = get_butterworth_kernel_half(size[1], cutoff, order)
    return commons.outer_bias_parallel(mask_y, mask_x, 1.)


@boundscheck(False)
@wraparound(False)
def approx_butterworth_highpass(
        size: KER_SIZE, cutoff: float, order: float,
        parallel: bool = False
    ) -> ARR_32F2D:
    """
    Return an approximate Butterworth high-pass filter.
    
    If parallel=True, then apply numba parallel computation.
    """
    if parallel:
        return _approx_butterworth_highpass_parallel(
            size, cutoff, order)
    else:
        return _approx_butterworth_highpass_nonparallel(
            size, cutoff, order)