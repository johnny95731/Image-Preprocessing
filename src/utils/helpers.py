from typing import Any, overload

__all__ = [
    'is_iterable',
    'is_valid_filename',
    'add_weighted',
    'nb_add',
    'nb_subtract',
    'nb_multiply',
    'nb_power',
    'nb_divide',
    'bias_reciprocal',
    'elementwise_mean',
    'nb_outer',
    'outer_bias',
    'outer_bias_parallel',
    'lowpass_outer',
    'lowpass_outer_parallel',
    'highpass_outer',
    'highpass_outer_parallel',
    'transform',
    'transform_1D',
    'pad_reflect101',
    'normalize_uint8_parallel',
    'normalize_uint8',
]

import numpy as np


from numba import njit, vectorize, prange
from src.utils.img_type import (
    Arr16S1D,
    Arr32F1D,
    Arr64F1D,
    Arr64S1D,
    Arr8U1D,
    Arr8U2D,
    Arr32F2D,
)

# This document includes some basic function.
# The operator with prefix 'nb_', e.g., nb_add and nb_multiply, is present for
# restricting the output dtype.


def is_iterable(obj: Any) -> bool:
    """Check whether an object is iterable or not."""
    try:
        iter(obj)
    except:  # noqa: E722
        return False
    else:
        return True


@njit('boolean(unicode_type)', nogil=True, cache=True, fastmath=True)
def is_valid_filename(text: str) -> bool:
    """Check whether a text is valid as a filename."""
    if not text:
        return False
    keys = ('\\', '/', ':', '*', '?', '"', '|', '<', '>')
    for key in keys:
        if key in text:
            return False
    return True


# Basic operators for controling dtype.
def conv2D(img: Arr32F2D, kernel: Arr32F2D) -> Arr32F2D:
    padded_img = pad_reflect101(img, kernel.shape)
    return convolution_gray(padded_img.astype(np.float32), kernel)


@njit(
    ['float32[:,:](float32[:,:],float32[:,:])'],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def convolution_gray(padded_img: Arr32F2D, kernel: Arr32F2D) -> Arr32F2D:
    padded_size = padded_img.shape
    ksize = kernel.shape
    orig_size = (padded_size[0] - ksize[0] + 1, padded_size[1] - ksize[1] + 1)
    output = np.empty(orig_size, dtype=np.float32)
    for y in prange(orig_size[0]):
        for x in range(orig_size[1]):
            window = padded_img[y : y + ksize[0], x : x + ksize[1]]
            conv_val = 0
            for ly, row in enumerate(kernel):
                for lx, val in enumerate(row):
                    conv_val += window[ly, lx] * val
            output[y, x] = conv_val
    return output


@njit(nogil=True, cache=True, fastmath=True, parallel=True)
def add_weighted(a, amount, b, amount2):
    return np.multiply(a, amount) + np.multiply(b, amount2)


# fmt: off
__SIGNATURE_BASIC_OPERATORS = [
    'int16(uint8,int64)',
    'float32(uint8,float32)',
    'float32(uint8,float64)',
    'float32(float32,float32)',
    'float32(float32,float64)',
    'float32(float64,float64)',
]
@overload
def nb_add(a: Arr8U1D, b: Arr64S1D) -> Arr16S1D: pass
@overload
def nb_add(a: Arr8U1D, b: Arr32F1D | Arr64F1D) -> Arr32F1D: pass
@overload
def nb_add(a: Arr32F1D, b: Arr32F1D | Arr64F1D) -> Arr32F1D: pass
@overload
def nb_add(a: Arr64F1D, b: Arr64F1D) -> Arr32F1D: pass
@overload
def nb_add(a: list[float | int], b: list[float | int]) -> Arr32F1D: pass
# fmt: on
@vectorize(__SIGNATURE_BASIC_OPERATORS, cache=True, fastmath=True)
def nb_add(a, b):
    """Element-wise add two areguments."""
    return np.add(a, b)


# fmt: off
@overload
def nb_subtract(a: Arr8U1D, b: Arr64S1D) -> Arr16S1D: pass
@overload
def nb_subtract(a: Arr8U1D, b: Arr32F1D | Arr64F1D) -> Arr32F1D: pass
@overload
def nb_subtract(a: Arr32F1D, b: Arr32F1D | Arr64F1D) -> Arr32F1D: pass
@overload
def nb_subtract(a: Arr64F1D, b: Arr64F1D) -> Arr32F1D: pass
@overload
def nb_subtract(a: list[float | int], b: list[float | int]) -> Arr32F1D: pass
# fmt: on
@vectorize(__SIGNATURE_BASIC_OPERATORS, cache=True, fastmath=True)
def nb_subtract(a, b):
    """Element-wise subtract two areguments."""
    return np.subtract(a, b)


# fmt: off
__SIGNATURE_MULTIPLY = [
    'int16(uint8,int64)',
    'float32(uint8,float32)',
    'float32(uint8,float64)',
    'float32(float32,float32)',
    'float32(float64,float32)',
    'complex64(complex64,float32)',
    'complex128(complex128,float32)',
]


# fmt: on
@vectorize(__SIGNATURE_MULTIPLY, cache=True, fastmath=True)
def nb_multiply(a, b):
    """Element-wise multiply two areguments."""
    return np.multiply(a, b)


@vectorize(__SIGNATURE_BASIC_OPERATORS, cache=True, fastmath=True)
def nb_power(base, exponent):
    """
    Firse argument(base) raised to the power of second argument(exponent),
    element-wise.
    """
    return np.power(base, exponent)


@vectorize(__SIGNATURE_BASIC_OPERATORS, cache=True, fastmath=True)
def nb_divide(a, b):
    return np.divide(a, b)


__SIGNATURE_BIAS_RECIPROCAL = [
    'float32(uint8,float32)',
    'float32(int16,float32)',
    'float32(float32,float32)',
]


@vectorize(__SIGNATURE_BIAS_RECIPROCAL, cache=True, fastmath=True)
def bias_reciprocal(x, bias):
    """Add a bias to x and then get the reciprocal."""
    return np.divide(1, np.add(x, bias))


__SIGNATURE_ELEMENTWISE_MEAN = [
    'float32(uint8,uint8)',
    'float32(float32,float32)',
]


@vectorize(__SIGNATURE_ELEMENTWISE_MEAN, cache=True, fastmath=True)
def elementwise_mean(a, b):
    """Element-wise calculate mean of two areguments."""
    return np.multiply(0.5, np.add(a, b))


# Operations for filtering.
__SIGNATURE_OUTER = [
    'float32[:,:](float32[:],float32[:])',
]


@njit(__SIGNATURE_OUTER, nogil=True, cache=True, fastmath=True, parallel=True)
def nb_outer(a: Arr32F1D, b: Arr32F1D) -> Arr32F2D:
    """
    Calculate the outer product of two 1-D array.

    This function is same as numpy.outer, but this is a parallel computation
    version with numba. numpy.outer is faster than nonparallel computation
    version.
    """
    output = np.empty((a.size, b.size), dtype=np.float32)
    for y in prange(a.size):
        output[y] = np.multiply(a[y], b)
    return output


__SIGNATURE_OUTER_BIAS = [
    'float32[:,:](float32[:],float32[:],float32)',
]


@njit(__SIGNATURE_OUTER_BIAS, nogil=True, cache=True, fastmath=True)
def outer_bias(a, b, bias):
    """
    Calculate bias minus the outer product of two 1-D array, bias - outer(a,b).

    This function is present for calculate highpass filter which lowpass filter
    is separable.
    """
    output = np.empty((a.size, b.size), dtype=np.float32)
    for y, val in enumerate(a):
        output[y] = bias - np.multiply(val, b)
    return output


@njit(__SIGNATURE_OUTER_BIAS, nogil=True, cache=True, fastmath=True, parallel=True)
def outer_bias_parallel(a, b, bias):
    """
    Calculate bias minus the outer product of a and b, bias - outer(a,b),
    with numba parallel computation.

    This function is present for calculate highpass filter which lowpass filter
    is separable.
    """
    output = np.empty((a.size, b.size), dtype=np.float32)
    for y in prange(a.size):
        output[y] = bias - np.multiply(a[y], b)
    return output


__SIGNATURE_LOWPASS_OUTER = [
    'complex128[:,:](complex128[:,:],float32[:],float32[:])',
    'complex64[:,:](complex64[:,:],float32[:],float32[:])',
    'uint8[:,:](uint8[:,:],float32[:],float32[:])',
]


@njit(__SIGNATURE_LOWPASS_OUTER, nogil=True, cache=True, fastmath=True)
def lowpass_outer(fftimg, a, b):
    """
    Multiply a lowpass filter which is outer(a,b) to fftimg.
    """
    output = np.empty_like(fftimg)
    for y, row_fft in enumerate(fftimg):
        output[y] = np.multiply(np.multiply(a[y], b), row_fft)
    return output


@njit(__SIGNATURE_LOWPASS_OUTER, nogil=True, cache=True, fastmath=True, parallel=True)
def lowpass_outer_parallel(fftimg, a, b):
    """
    Multiply a lowpass filter which is outer(a,b) to fftimg with numba parallel
    computation.
    """
    output = np.empty_like(fftimg)
    for y in prange(fftimg.shape[0]):
        output[y] = (a[y] * b) * fftimg[y]
    return output


__SIGNATURE_HIGHPASS_OUTER = [
    'complex128[:,:](complex128[:,:],float32[:],float32[:],float32)',
    'complex64[:,:](complex64[:,:],float32[:],float32[:],float32)',
    'float32[:,:](float32[:,:],float32[:],float32[:],float32)',
    'uint8[:,:](uint8[:,:],float32[:],float32[:],float32)',
]


@njit(__SIGNATURE_HIGHPASS_OUTER, nogil=True, cache=True, fastmath=True)
def highpass_outer(fftimg, a, b, bias):
    """
    Multiply a highpass filter which the lowpass filter is outer(a,b),
    to fftimg.
    """
    output = np.empty_like(fftimg)
    for y, row_img in enumerate(fftimg):
        output[y] = (bias - a[y] * b) * row_img
    return output


@njit(__SIGNATURE_HIGHPASS_OUTER, nogil=True, cache=True, fastmath=True, parallel=True)
def highpass_outer_parallel(fftimg, a, b, bias):
    """
    Multiply a highpass filter which the lowpass filter is outer(a,b),
    to fftimg with numba parallel computation.
    """
    ksize = fftimg.shape
    output = np.empty_like(fftimg)

    for y in prange(ksize[0]):
        output[y] = (bias - a[y] * b) * fftimg[y]
    return output


# # Norms
# _SIGNATURE_2_NORM = [
#     "float32[:,:](float32[:,:],float32[:,:])",
#     "float64[:,:](float64[:,:],float64[:,:])",
# ]
# @njit(
#     _SIGNATURE_2_NORM,
#     nogil=True, cache=True,  fastmath=True, parallel=True)
#
# def euclidean_norm(y, x):
#     """
#     Return the Euclidean norm (or, 2-norm) of in an 2-D space.

#     The first argument is the y-axis component, and the second argument is the
#     x-axis component.
#     """
#     return np.sqrt(np.add(np.power(y,2), np.power(x,2)))


# _SIGNATURE_P_NORM = [
#     "float32[:,:](float32[:,:],float32[:,:],float32)",
#     "float64[:,:](float64[:,:],float64[:,:],float32)",
# ]
# @njit(
#     _SIGNATURE_P_NORM,
#     nogil=True, cache=True, fastmath=True, parallel=True)
#
# def p_metric(y, x, p):
#     """
#     Return the p-norm of in an 2-D space.

#     The first argument is the y-axis component, and the second argument is the
#     x-axis component.
#     """
#     return np.power( np.add(np.power(y,p), np.power(x,p)), np.divide(1,p))


# _SIGNATURE_SUP_METRIC = [
#     "float64[:,:](float64[:,:],float64[:,:])",
# ]
# @njit(
#     _SIGNATURE_SUP_METRIC,
#     nogil=True, cache=True, fastmath=True, parallel=True)
#
# def sup_metric(y,x):
#     """
#     Return the sup-norm of in an 2-D space, that is, return y is y > x and
#     return x is x > y.

#     The first argument is the y-axis component, and the second argument is the
#     x-axis component.
#     """
#     return np.maximum(y, x)


# _SIGNATURE_DIST_MATRIX = [
#     "float32[:,:](UniTuple(uint16,2))"
# ]
# @njit(
#     _SIGNATURE_DIST_MATRIX,
#     nogil=True, cache=True, fastmath=True)
#
# def dist_matrix(size):
#     """
#     Return an 2-D matrix whose (y,x) entry is the dist between (y,x) and
#     center, where center = size//2.
#     """
#     output = np.empty(size, dtype=np.float32)
#     center_y = size[0] // 2
#     center_x = size[1] // 2

#     for y in range(size[0]):
#         dist_y = np.power(center_y-y, 2)
#         for x in range(size[1]):
#             output[y,x] = np.sqrt(dist_y + np.power(center_x-x, 2))
#     return output


# _SIGNATURE_DIST_SQUARE_MATRIX = [
#     "float32[:,:](UniTuple(uint16,2))"
# ]
# @njit(
#     _SIGNATURE_DIST_SQUARE_MATRIX,
#     nogil=True, cache=True, fastmath=True)
#
# def dist_square_matrix(size):
#     """
#     Return an 2-D matrix whose (y,x) entry is the square of dist between (y,x)
#     and center, where center = ceil((size-1) / 2).
#     """
#     output = np.empty(size, dtype=np.float32)
#     center_y = size[0] // 2
#     center_x = size[1] // 2

#     for y in range(size[0]):
#         dist_y = np.power(center_y-y, 2)
#         for x in range(size[1]):
#             output[y,x] = dist_y + np.power(center_x-x, 2)
#     return output


# Thansform uint8 data by table[data].
# fmt: off
@overload
def transform(img: Arr8U2D, table: Arr8U1D) -> Arr8U2D: pass
@overload
def transform(img: Arr8U2D, table: Arr32F1D) -> Arr32F2D: pass
# fmt: on
@njit(
    [
        'uint8[:,:](uint8[:,:],uint8[:])',
        'float32[:,:](uint8[:,:],float32[:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def transform(img, table):
    """
    Transforms intensity of an image with a table:
        output = table[img].

    Parameters
    ----------
    img : Arr8U2D
        An 8UC1 image.
    table : ARR_1D
        A 1-D array with length 256.

    Returns
    -------
    output : IMG_GRAY
        New image. The dtype depend on the dtype of table.
    """
    output = np.empty_like(img, dtype=table.dtype)
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            output[y, x] = table[val]
    return output


# fmt: off
@overload
def transform_1D(seq: Arr8U2D, table: Arr8U1D) -> Arr8U1D: pass
@overload
def transform_1D(seq: Arr8U2D, table: Arr32F1D) -> Arr32F1D: pass
# fmt: on
@njit(
    [
        'uint8[:](uint8[:],uint8[:])',
        'float32[:](uint8[:],float32[:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def transform_1D(seq: Arr8U2D, table: Arr8U1D | Arr32F1D) -> Arr8U1D | Arr32F1D:
    """Transforms intensity of a sequence with a table:
    output = table[seq].
    """
    output = np.empty_like(seq, dtype=table.dtype)
    for y, val in enumerate(seq):
        output[y] = table[val]
    return output  # type: ignore


# Padding Border. See OpenCV BorderTypes:
# https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5


def pad_reflect101(img: np.ndarray, ksize=(3, 3)) -> np.ndarray:
    """Padding Border with REFLECT101: dcb | abcdef | edc."""
    size = img.shape
    khalf_y = ksize[0] // 2
    khalf_x = ksize[1] // 2
    output = np.empty((size[0] + ksize[0] - 1, size[1] + ksize[1] - 1), dtype=img.dtype)

    output[khalf_y:-khalf_y, khalf_x : size[1] + khalf_x] = img  # Central=img
    output[khalf_y:-khalf_y, :khalf_x] = img[:, khalf_x:0:-1]  # Left
    output[khalf_y:-khalf_y, size[1] + khalf_x :] = img[
        :, size[1] - 2 : size[1] - khalf_x - 2 : -1
    ]  # Right

    output[:khalf_y, :] = output[ksize[0] - 1 : khalf_y : -1]  # Top
    output[-1 : -khalf_y - 1 : -1, :] = output[
        size[0] - 1 : size[0] + khalf_y - 1
    ]  # Bottom
    return output


# Normalize
@njit(
    [
        'float32[:,:](uint8[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def normalize_uint8_parallel(img: Arr8U2D) -> Arr32F2D:
    """Compress the range of values of an 8UC1 image to [0,1] with
    parallel=True.

    Parameters
    ----------
    img : Arr8U2D
        An 8UC1 image.

    Returns
    -------
    output : Arr32F2D
        Compressed image.
    """
    output = np.empty_like(img, dtype=np.float32)
    tabel = np.empty(256, dtype=np.float32)
    for i in range(256):
        tabel[i] = np.divide(i, 255)

    for y in prange(img.shape[0]):
        for x, val in enumerate(img[y]):
            output[y, x] = tabel[val]
    return output  # type: ignore


@njit(
    [
        'float32[:,:](uint8[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def normalize_uint8(img: Arr8U2D) -> Arr32F2D:
    """Compress the range of values of an 8UC1 image to [0,1].

    Parameters
    ----------
    img : Arr8U2D
        An 8UC1 image.

    Returns
    -------
    output : Arr32F2D
        Compressed image.
    """
    output = np.empty_like(img, dtype=np.float32)
    tabel = np.empty(256, dtype=np.float32)
    for i in range(256):
        tabel[i] = np.divide(i, 255)

    for y, row in enumerate(img):
        for x, val in enumerate(row):
            output[y, x] = tabel[val]
    return output  # type: ignore
