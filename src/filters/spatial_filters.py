__all__ = [
    'auto_gaussian_kernel_size',
    'auto_gaussian_kernel_sigma',
    'check_gaussian_kernel_arg',
    'get_mean_kernel',
    'get_1d_gaussian',
    'get_2d_gaussian_kernel',
    'mean_blur',
    'gaussian_blur',
    'bilateral_blur',
    'median_blur',
    'maximum_blur',
    'minimum_blur',
    'geometric_mean_blur',
    'reciprocal_8UC1',
    'reciprocal_8UC3',
    'harmonic_mean_blur',
    'contraharmonic_mean_blur',
    'midpoint_filter',
    'alpha_trimmed_mean_blur',
    'adaptive_mean_blur',
    'adaptive_median_filter',
    'get_gradient_operator_names',
    'get_gradient_operator',
    'get_line_operator_name',
    'get_line_operator',
    'gradient_norm',
    'gradient_uint8_overflow',
    'get_derivatives_of_gaussian',
    'get_difference_gaussian',
    'get_laplacian_gaussian',
    'laplacian',
    'laplacian_all',
    'laplacian2',
    'laplacian4',
    'sobel',
    'kirsch',
    'gaussian_gradient',
    'marr_hildreth',
    'unsharp_masking',
]
from typing import Literal
from math import ceil, exp
import textwrap

from cython import boundscheck, wraparound
from numba import (
    njit,
    prange,
    typeof,
    int64,
    int32,
    int16,
    int8,
    uint64,
    uint32,
    uint16,
    uint8,
)

import numpy as np
from numpy import pi

import cv2
from cv2 import CV_32F

from src.utils.helpers import is_iterable, pad_reflect101
from src.utils.stats import quick_sort, mean_seq, max_min, median
from src.utils.img_type import (
    IMG_ARRAY,
    Arr8U2D,
    Arr32F2D,
    Arr32F1D,
    IMG_32F,
    KER_SIZE,
    IMG_8U,
)


# Checking parameters
@boundscheck(False)
@wraparound(False)
def __valid_convolution_ksize(ksize: int | KER_SIZE = 3) -> KER_SIZE:
    """Check the validity of size of convolution kernel.
    If ksize is even, then ksize += 1.

    Parameters
    ----------
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    ksize : Tuple[int, int]
        New size of kernel. A cuple of positive odd integer that both greater
        than 1.

    Raises
    ------
    TypeError
        ksize is not int or a cuple of int.
    ValueError
        ksize is even or is less than 1.
        ksize is tuple and its length is not 2.
        One of element of ksize is even or is less than 1.
    """
    error_message = ' '.join([
        'ksize should be one (or two) positive odd integer(s) that',
        'greater than 1.',
    ])

    integer_type = [  # numba type
        int64,
        int32,
        int16,
        int8,
        uint64,
        uint32,
        uint16,
        uint8,
    ]
    if not is_iterable(ksize):
        if typeof(ksize) not in integer_type:
            raise TypeError(error_message)
        if not ksize % 2 or ksize < 1:
            raise ValueError(error_message)
        ksize = (ksize, ksize)
    else:
        if len(ksize) != 2:
            raise ValueError(error_message)
        for val in ksize:
            if typeof(val) not in integer_type:
                raise TypeError(error_message)
            if not val % 2 or val < 1:
                raise ValueError(error_message)
        ksize = tuple(ksize)
    return ksize


@njit(
    [
        'int64(float32)',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
@boundscheck(False)
@wraparound(False)
def auto_gaussian_kernel_size(sigma: float) -> int:
    """Calculate ksize for gaussian blur by the following formula:
        min{ i | i is odd and i >= max(3, (20*sigma-7) / 3) }

    Parameters
    ----------
    sigma : float
        The parameter sigma of gaussian function. Must be positive.

    Returns
    -------
    ksize : int
        Size of kernel. An odd number.
    """
    y = ceil(6.6666667 * sigma - 2.3333333)
    if y < 3:
        y = 3
    elif not y % 2:
        y += 1
    return y


@njit(
    [
        'float32(int64)',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
@boundscheck(False)
@wraparound(False)
def auto_gaussian_kernel_sigma(ksize: int) -> float:
    """Computed sigma for gaussian blur by the following formula:
        sigma = 0.3*((ksize-1)*0.5-1) + 0.8 = 0.15*ksize - 0.35
    Check the OpenCV document of getGaussianKernel: https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gac05a120c1ae92a6060dd0db190a61afa

    Parameters
    ----------
    ksize : int
        Size of kernel. Must be odd number or a tuple of two odd numbers.

    Returns
    -------
    sigma : float
        The parameter sigma of gaussian function.
    """
    return 0.15 * ksize - 0.35


@boundscheck(False)
@wraparound(False)
def check_gaussian_kernel_arg(
    ksize: int | KER_SIZE, sigma_y: float, sigma_x: float
) -> tuple[KER_SIZE, float, float]:
    """Check the validity of gaussian kernel arguments. If ksize and sigma_y

    Parameters
    ----------
    ksize : int | tuple[int, int]
        Size of kernel. Must be odd number or a tuple that contains two odd
        numbers. If ksize < 1 and sigma_y or sigma_x <= 0, then set ksize = 3.
    sigma_y : float
        The parameter sigma of gaussian function along y-direction. If
        sigma_y <= 0, then sigma_y will be computed from ksize.
    sigma_x : float
        The parameter sigma of gaussian function along x-direction. If
        sigma_x <= 0, then sigma_x will be computed from ksize.

    Returns
    -------
    ksize : tuple[int, int]
        Size of kernel. A tuple that contains two odd numbers.
    sigma_y : float
        The parameter sigma of gaussian function along y-direction.
    sigma_x : float
        The parameter sigma of gaussian function along x-direction.

    Raises
    ------
    TypeError
        ksize is neather int nor a cuple of int.
    ValueError
        ksize is tuple and its length is not 2.
        ksize is even number(s).
    """
    # Choose Suitable Ksize if ksize is negative
    if not is_iterable(ksize) and ksize < 1:
        ksize = (ksize, ksize)
        if sigma_y <= 0:
            ksize = (3, ksize[1])
            sigma_y = auto_gaussian_kernel_sigma(3)
        if sigma_x <= 0:
            ksize = (ksize[0], 3)
            sigma_x = auto_gaussian_kernel_sigma(3)
        if sigma_y > 0 and sigma_x > 0:
            ksize = (
                auto_gaussian_kernel_size(sigma_y),
                auto_gaussian_kernel_size(sigma_x),
            )
    elif is_iterable(ksize):
        if len(ksize) != 2:
            __valid_convolution_ksize(ksize)  # Raise ValueError
        if ksize[0] < 1 and sigma_y <= 0:
            ksize = (3, ksize[1])
            sigma_y = auto_gaussian_kernel_sigma(3)
        if ksize[1] < 1 and sigma_x <= 0:
            ksize = (ksize[0], 3)
            sigma_x = auto_gaussian_kernel_sigma(3)
        if ksize[0] < 1:
            ksize = (auto_gaussian_kernel_size(sigma_y), ksize[1])
        if ksize[1] < 1:
            ksize = (
                ksize[0],
                auto_gaussian_kernel_size(sigma_y),
            )

    # Check ksize
    ksize = __valid_convolution_ksize(ksize)
    # Choose Suitable Sigma if ksize is negative
    if sigma_y <= 0:
        sigma_y = auto_gaussian_kernel_sigma(ksize[0])
    if sigma_x <= 0:
        sigma_x = auto_gaussian_kernel_sigma(ksize[1])
    return ksize, sigma_y, sigma_x


## Blurring
# Kernel getters
__SIGNATURE_GET_MEAN_KERNEL = ['float32[:,:](UniTuple(uint32,2))']


@njit(__SIGNATURE_GET_MEAN_KERNEL, nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_mean_kernel(ksize) -> Arr32F2D:
    """Return the mean blur kernel.

    Parameters
    ----------
    ksize : Tuple[int, int]
        Size of kernel. Must be a tuple of two odd numbers.

    Returns
    -------
    kernel : KER_32F2D
        Mean blurring kernel.
    """
    kernel = np.empty(ksize, dtype=np.float32)
    value = 1 / kernel.size
    for y in range(ksize[0]):
        for x in range(ksize[1]):
            kernel[y, x] = value
    return kernel


@njit(['float32[:](int64,float32)'], nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_1d_gaussian(ksize: int, sigma: float) -> Arr32F1D:
    """Return the 1D gaussian kernel. The center of kernel is (ksize//2). The
    kernel will be normalized to sum(kernel) = 1.

    Parameters
    ----------
    ksize : int
        Size of kernel.
    sigma : float
        The parameter sigma of gaussian function.

    Returns
    -------
    kernel : KER_32F1D
        1D gaussian kernel.
    """
    kernel = np.empty((ksize), dtype=np.float32)
    const = np.divide(-0.5, np.power(sigma, 2))
    half = ksize // 2
    summation = 0
    for y in range(ksize):
        val = exp(np.multiply((half - y) ** 2, const))
        kernel[y] = val
        summation += val
    # Normalized
    kernel *= 1 / summation
    return kernel


@njit(
    [
        'float32[:,:](UniTuple(int64,2),float32,float32)',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
@wraparound(False)
def get_2d_gaussian_kernel(ksize: KER_SIZE, sigma_y: float, sigma_x: float) -> Arr32F2D:
    """Create an 2D gaussian kernel by outer product two 1D gaussian kernel.

    Parameters
    ----------
    ksize : tuple[int, int]
        Size of kernel.
    sigma_y : float
        The parameter sigma of 2D gaussian function along y-direction.
    sigma_x : float
        The parameter sigma of 2D gaussian function along x-direction.

    Returns
    -------
    kernel : KER_32F2D
        2D gaussian kernel.
    """
    kernel_y = get_1d_gaussian(ksize[0], sigma_y)
    kernel_x = get_1d_gaussian(ksize[1], sigma_x)
    return np.outer(kernel_y, kernel_x)


# Blurring Operators
@boundscheck(False)
@wraparound(False)
def mean_blur(img: IMG_ARRAY, ksize: int | KER_SIZE = (3, 3)) -> IMG_ARRAY:
    """Applies the mean filter to an image.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : IMG_ARRAY
        Output image.
    """
    ksize = __valid_convolution_ksize(ksize)
    kernel = get_mean_kernel(ksize)
    return cv2.filter2D(img, CV_32F, kernel)


@boundscheck(False)
@wraparound(False)
def gaussian_blur(
    img: IMG_ARRAY,
    ksize: int | KER_SIZE = 3,
    sigma_y: float = 0,
    sigma_x: float = 0,
) -> IMG_ARRAY:
    """Applies the gaussian filter to an image.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | tuple[int, int], default=3
        Size of kernel. Must satisfies one of following conditions:
        1. int and odd. If ksize < 1 and sigma <= 0, then set ksize = 3. If
           ksize < 1 sigma > 0, then ksize will be computed from sigma.
        2. Two numbers that both satisfy condition 1.
    sigma_y : float, default=0
        The parameter sigma of gaussian function along y-direction. If
        sigma_y <= 0, then sigma_y will be computed from ksize.
    sigma_x : float, default=0
        The parameter sigma of gaussian function along x-direction. If
        sigma_x <= 0, then sigma_x will be computed from ksize.

    Returns
    -------
    output : IMG_ARRAY
        Output image.
    """
    ksize, sigma_y, sigma_x = check_gaussian_kernel_arg(ksize, sigma_y, sigma_x)
    kernel = get_2d_gaussian_kernel(ksize, sigma_y, sigma_x)
    return cv2.filter2D(img, CV_32F, kernel)


@boundscheck(False)
@wraparound(False)
def bilateral_blur(
    img: IMG_ARRAY, ksize: int, sigma_color: float = 50, sigma_space: float = 50
) -> IMG_ARRAY:
    """Applies the bilateral filter to an image.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int
        Size of kernel. Must be odd number or a tuple that contains two odd
        numbers. If ksize < 1 and sigma_y or sigma_x <= 0, then set ksize = 3.
    sigma_color : float
        Filter sigma in the color space. A larger value of the parameter means
        that farther colors within the pixel neighborhood (see sigma_space)
        will be mixed together, resulting in larger areas of semi-equal color.
    sigma_space : float
        Filter sigma in the coordinate space. A larger value of the parameter
        means that farther pixels will influence each other as long as their
        colors are close enough (see sigma_color).

    Returns
    -------
    output : IMG_ARRAY
        Output image.
    """
    if not isinstance(ksize, int):
        raise TypeError('ksize must be int.')
    return cv2.bilateralFilter(img, ksize, sigma_color, sigma_space)


@boundscheck(False)
@wraparound(False)
def median_blur(img: IMG_ARRAY, ksize: int) -> IMG_ARRAY:
    """Blurs an image using the median filter.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int
        Size of kernel. Must be a positive odd number.
        If ksize == 1, return img.

    Returns
    -------
    output : IMG_ARRAY
        Output image.

    Raises
    ------
    TypeError
        ksize is not int.
    ValueError
        ksize is even or ksize <= 1.
    """
    if not isinstance(ksize, int):
        raise TypeError('ksize must be int.')
    if not ksize % 2 or ksize < 1:
        raise ValueError('ksize must be a positive odd number.')
    if ksize == 1:
        return img
    else:
        return cv2.medianBlur(img, ksize)


@boundscheck(False)
@wraparound(False)
def maximum_blur(img: IMG_ARRAY, ksize: int | KER_SIZE) -> IMG_ARRAY:
    """Blurs an image using the maximum filter.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : IMG_ARRAY
        Output image.
    """
    if not isinstance(ksize, int):
        raise TypeError('ksize must be int.')
    if not ksize % 2 or ksize <= 1:
        raise TypeError('ksize must be a positive odd number that greater than 1.')
    ksize = __valid_convolution_ksize(ksize)
    kernel = np.ones(ksize, dtype=np.uint8)
    return cv2.dilate(img, kernel)


@boundscheck(False)
@wraparound(False)
def minimum_blur(img: IMG_ARRAY, ksize: int | KER_SIZE) -> IMG_ARRAY:
    """Blurs an image using the minimum filter.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : IMG_ARRAY
        Output image.
    """
    ksize = __valid_convolution_ksize(ksize)
    kernel = np.ones(ksize, dtype=np.uint8)
    return cv2.erode(img, kernel)


@njit('float32[:,:](uint8[:,:])', nogil=True, cache=True, fastmath=True)
@wraparound(False)
def logarithm_8UC1(img: Arr8U2D):
    output = np.empty_like(img, dtype=np.float32)
    table = np.empty(256, dtype=np.float32)
    for i in range(256):
        table[i] = np.log(i)
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            output[y, x] = table[val]
    return output


@njit('float32[:,:,:](uint8[:,:,:])', nogil=True, cache=True, fastmath=True)
@wraparound(False)
def logarithm_8UC3(img: Arr8U2D):
    output = np.empty_like(img, dtype=np.float32)
    table = np.empty(256, dtype=np.float32)
    for i in range(256):
        table[i] = np.log(i)
    for y, row in enumerate(img):
        for x, channels in enumerate(row):
            for c, val in enumerate(channels):
                output[y, x, c] = table[val]
    return output


@boundscheck(False)
@wraparound(False)
def geometric_mean_blur(img: IMG_8U, ksize: int | KER_SIZE = (3, 3)) -> IMG_8U:
    """Blurs an image using the geometric mean filter.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : ARR_8U
        Output image.
    """
    ksize = __valid_convolution_ksize(ksize)
    if img.dtype == np.uint8:
        if img.ndim == 2:
            log_ = logarithm_8UC1(img)
        elif img.ndim == 3:
            log_ = logarithm_8UC3(img)
    else:
        log_ = np.log(img, dtype=np.float32)
    log_mean = mean_blur(log_, ksize)
    blurred = np.exp(log_mean, dtype=np.float32)
    return cv2.convertScaleAbs(blurred)


@njit('float32[:,:](uint8[:,:])', nogil=True, cache=True, fastmath=True)
@wraparound(False)
def reciprocal_8UC1(img: Arr8U2D):
    output = np.empty_like(img, dtype=np.float32)
    table = np.empty(256, dtype=np.float32)
    for i in range(256):
        table[i] = np.divide(1, i)
    for y, row in enumerate(img):
        for x, val in enumerate(row):
            output[y, x] = table[val]
    return output


@njit('float32[:,:,:](uint8[:,:,:])', nogil=True, cache=True, fastmath=True)
@wraparound(False)
def reciprocal_8UC3(img: Arr8U2D):
    output = np.empty_like(img, dtype=np.float32)
    table = np.empty(256, dtype=np.float32)
    for i in range(256):
        table[i] = np.divide(1, i)
    for y, row in enumerate(img):
        for x, channels in enumerate(row):
            for c, val in enumerate(channels):
                output[y, x, c] = table[val]
    return output


@boundscheck(False)
@wraparound(False)
def harmonic_mean_blur(img: IMG_8U, ksize: int | KER_SIZE = (3, 3)) -> IMG_8U:
    """Blurs an image using the harmonic mean filter.

    Parameters
    ----------
    img : ARR_8U
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : ARR_8U
        Output image.
    """
    ksize = __valid_convolution_ksize(ksize)
    if img.dtype == np.uint8:
        if img.ndim == 2:
            recip_ = reciprocal_8UC1(img)
        elif img.ndim == 3:
            recip_ = reciprocal_8UC3(img)
    else:
        recip_ = np.divide(1, img, dtype=np.float32)
    recip_mean = mean_blur(recip_, ksize)
    blurred = np.divide(1, recip_mean, dtype=np.float32)
    return cv2.convertScaleAbs(blurred)


@boundscheck(False)
@wraparound(False)
def contraharmonic_mean_blur(
    img: IMG_8U, ksize: int | KER_SIZE = (3, 3), q: int | float = 1
) -> IMG_8U:
    """Blurs an image using the contraharmonic mean filter with parameter q.
        blurred = mean_blur(img**(q+1)) / mean_blur(img**q)

    Parameters
    ----------
    img : ARR_8U
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.
    q : int | float
        The order q of the filter. For positive values of q, the filter
        eliminates pepper noise. For negative values of q it eliminates salt
        noise. If q = 0, the contraharmonic mean equals the arithmetic mean.
        If Q = −1, the contraharmonic mean equals the harmonic mean.

    Returns
    -------
    output : ARR_8U
        Output image.
    """
    # Contraharmonic Mean Filter
    if q == 0:
        return mean_blur(img, ksize)
    elif q == -1:
        return harmonic_mean_blur(img, ksize)
    ksize = __valid_convolution_ksize(ksize)
    kernel = get_mean_kernel(ksize)

    img_q1 = np.power(img, q + 1, dtype=np.float32)
    img_q = np.power(img, q, dtype=np.float32)
    filter_q1 = cv2.filter2D(img_q1, CV_32F, kernel)
    filter_q = cv2.filter2D(img_q, CV_32F, kernel)
    blurred = np.divide(filter_q1, filter_q, dtype=np.float32)
    return cv2.convertScaleAbs(blurred)


@boundscheck(False)
@wraparound(False)
def midpoint_filter(img: IMG_8U, ksize: int | KER_SIZE = (3, 3)) -> IMG_8U:
    """Blurs an image using the midpoint filter.
        blurred = (max_blur(img)+min_blur(img)) / 2

    Parameters
    ----------
    img : ARR_8U
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.

    Returns
    -------
    output : ARR_8U
        Output image.
    """
    ksize = __valid_convolution_ksize(ksize)
    kernel = np.ones(ksize, dtype=np.float32)
    max_blurred = cv2.dilate(img, kernel).astype(np.float32)
    min_blurred = cv2.erode(img, kernel).astype(np.float32)
    blurred = cv2.addWeighted(max_blurred, 0.5, min_blurred, 0.5)
    return cv2.convertScaleAbs(blurred)


__SIGNATURE_ALPHA_TRIMMED_MEAN = [
    'float32[:,:](uint8[:,:],UniTuple(uint16,2),UniTuple(uint16,2),uint16)'
]


@njit(
    __SIGNATURE_ALPHA_TRIMMED_MEAN, nogil=True, cache=True, fastmath=True, parallel=True
)
@wraparound(False)
def __alpha_trimmed_mean(pad_img, originalSize, ksize, alpha):
    output = np.empty(originalSize, dtype=np.float32)
    half_alpha = alpha // 2
    for y in prange(originalSize[0]):
        for x in prange(originalSize[1]):
            sort_ = quick_sort(pad_img[y : y + ksize[0], x : x + ksize[1]].flatten())[
                half_alpha:-half_alpha
            ]
            output[y, x] = mean_seq(sort_)
    return output


@boundscheck(False)
@wraparound(False)
def alpha_trimmed_mean_blur(
    img: IMG_8U, ksize: int | KER_SIZE = (3, 3), alpha: int = 2
) -> IMG_8U:
    """Blurs an image using the alpha-trimmed mean filter. The filter at index
    [y,x] of image will do following steps:
        1. Pick up elements.
        2. Order elements.
        3. Trim (alpha//2) elements at the beginning and at the end of order
           elements.
        4. Computing mean value of remains.
        5. (Go to next index)

    Parameters
    ----------
    img : ARR_8U
        Input image.
    ksize : int | Tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.
    alpha : int
        Trimmed number of filter. Should be even.

    Returns
    -------
    blurred : ARR_8U
        Output image.
    """
    # Alpha-Trimmed Mean Filter, very slow
    ksize = __valid_convolution_ksize(ksize)

    pad_img = pad_reflect101(img, ksize)
    return __alpha_trimmed_mean(pad_img, img.shape, ksize, alpha)


__SIGNATURE_ADAPTIVE_NOISE_REDUCTION = [
    'float32[:,:](uint8[:,:],float32[:,:],float32[:,:],float32[:,:],float32)'
]


@njit(
    __SIGNATURE_ADAPTIVE_NOISE_REDUCTION,
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=False,
)
@wraparound(False)
def __adaptive_mean(img, local_mean, trans, local_var, var_eta):
    output = np.empty_like(img, dtype=np.float32)
    for y, row_var in enumerate(local_var):
        for x, var_ in enumerate(row_var):
            if var_ < var_eta:
                output[y, x] = local_mean[y, x]
            else:
                c = np.divide(var_eta, var_)  # c < 1
                output[y, x] = img[y, x] + np.multiply(c, trans[y, x])
    return output


@boundscheck(False)
@wraparound(False)
def adaptive_mean_blur(
    img: IMG_8U, ksize: int | KER_SIZE = (3, 3), var_eta: int = 1
) -> IMG_8U:
    """Blurs an image using the adaptive mean filter. The filter at index [y,x]
    of image will do following steps:
        1. Pick up elements.
        2. Calculate variance of elements, var.
        3. If var < var_eta, put blurred[y,x] = mean[y,x], where mean[y,x] is
           the mean filter of img at index [y,x].
           Otherwise, put
               blurred[y,x] = img[y,x] + (var_eta/var) * (mean[y,x]-img[y,x]).
        4. (Go to next index)

    Parameters
    ----------
    img : ARR_8U
        Input image.
    ksize : int | Tuple[int, int], fefault=(3, 3)
        Size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.
    var_eta : float
        The threshold value of variance.

    Returns
    -------
    blurred : ARR_8U
        Output image.
    """
    kernel = get_mean_kernel(ksize)
    mean_blurred = cv2.filter2D(img, cv2.CV_32F, kernel)
    # Calculate variance
    translation = np.subtract(img, mean_blurred, dtype=np.float32)
    local_var = cv2.filter2D(translation**2, cv2.CV_32F, kernel)
    # Adaptive decision
    blurred = __adaptive_mean(img, mean_blurred, translation, local_var, var_eta)
    return cv2.convertScaleAbs(blurred)


__SIGNATURE_ADAPTIVE_MEDIAN = [
    'uint8[:,:](uint8[:,:],UniTuple(uint16,2),UniTuple(uint16,2))',
]


@njit(__SIGNATURE_ADAPTIVE_MEDIAN, nogil=True, cache=True, fastmath=True, parallel=True)
@wraparound(False)
def __adaptive_median(pad_img, initial, final):
    c_init_y = (final[0] - initial[0]) // 2
    c_init_x = (final[1] - initial[1]) // 2
    c_final_y = (final[0] + initial[0]) // 2
    c_final_x = (final[1] + initial[1]) // 2
    center = (final[0] // 2, final[1] // 2)
    iter_max = 1 + (final[0] - initial[0]) // 2  # Maximum of iteration number.
    original_size = (  # Size before padding.
        pad_img.shape[0] - final[0] + 1,
        pad_img.shape[1] - final[1] + 1,
    )
    output = np.empty(original_size, dtype=pad_img.dtype)
    for y in prange(original_size[0]):
        for x in prange(original_size[1]):
            # Indices of slicing
            i_y = y + c_init_y
            f_y = y + c_final_y
            i_x = x + c_init_x
            f_x = x + c_final_x
            #
            img_val = pad_img[y + center[0], x + center[1]]
            is_assigned = False
            for _ in range(iter_max):
                region = pad_img[i_y:f_y, i_x:f_x]  # Step 1 and step 2.
                # Step 3.
                max_, min_ = max_min(region)
                median_value = median(region)
                if min_ < median_value < max_:  # Step 4.
                    # Step 5.
                    if min_ < img_val < max_:
                        output[y, x] = img_val
                    else:
                        output[y, x] = median_value
                    is_assigned = True
                    break
                else:
                    # Step 6.
                    i_y -= 1
                    i_x -= 1
                    f_y += 1
                    f_x += 1
            if not is_assigned:  # Did not enter step 5. # Step 6
                output[y, x] = median_value
    return output


@boundscheck(False)
@wraparound(False)
def adaptive_median_filter(
    img: IMG_8U,
    kinitial: int | KER_SIZE = (3, 3),
    kfinal: int | KER_SIZE = (7, 7),
) -> IMG_8U:
    """Blurs an image using the adaptive median filter. The filter at index
    [y,x] of image will do following steps:
        1. Pick up elements with kernel size: kinitial. Enter step 3.
        2. Pick up elements.
        3. Find maximum, minimum, and median of elements.
        4. If median not in {maximum, minimum}, then enter step 5. Otherwise,
           enter step 6.
        5. If img[y,x] not in {maximum, minimum}, put blurred[y,x] = img[y,x].
           Otherwise, put blurred[y,x] = median. Enter step 7.
        6. If kernel size less than kfinal, expanding kernel size with width+2
           and height+2 and then enter step 2. Otherwise, put
               blurred[y,x] = median and enter step 7.
        7. (Go to next index)

    Parameters
    ----------
    img : ARR_8U
        Input image.
    kinitial : int | Tuple[int, int], fefault=(3, 3)
        Initial size of kernel. Must satisfies one of following conditions:
        1. int type, odd, and positive.
        2. Two numbers that both satisfy condition 1.
    kfianl : int | Tuple[int, int], default=(7, 7)
        Fianl size of kernel. Must satisfies condition 1 or condition 3:
        1. int type, odd, positive, greater than initial.
        2. kfianl.width-kinitial.width == kfianl.height-kinitial.height.
        3. Two numbers that both satisfy condition 1. Satisfy condition 2.

    Returns
    -------
    blurred : ARR_8U
        Output image.
    """
    kinitial = __valid_convolution_ksize(kinitial)
    kfinal = __valid_convolution_ksize(kfinal)
    pad_img = pad_reflect101(img, kfinal)
    return __adaptive_median(pad_img, kinitial, kfinal)


## Edge Detection 測邊
# Kernel getters
@boundscheck(False)
@wraparound(False)
def get_gradient_operator_names():
    """Get a list of availible gradient operator name.

    Returns
    -------
    output : tuple[str]
        List of availible gradient operator name.
    """
    return (
        'laplacians',
        'laplacians_all',
        'laplacian2',
        'laplacian4',
        'sobel',
        'prewitt',
        'robinson',
        'roberts',
        'kirsch',
        'scharr',
        'LoG',
    )


@boundscheck(False)
@wraparound(False)
def get_gradient_operator(grad_name: str) -> list[Arr32F2D]:
    """Get a list of availible gradient operator name.

    Parameters
    ----------
    grad_name : str
        gradient operator name

    Returns
    -------
    output : list[KER_32F2D]
        List of gradient kernel.

    Raises
    ------
    ValueError
        grad_name not in get_gradient_operator_names().
    """
    # Lapacian operator
    if grad_name == 'laplacians':
        kernel = [
            # Y
            [[1], [-2], [1]],
            # X
            [[1, -2, 1]],
        ]
    elif grad_name == 'laplacians_all':
        kernel = [
            # Y
            [[1], [-2], [1]],
            # X
            [[1, -2, 1]],
            # NW
            [[1, 0, 0], [0, -2, 0], [0, 0, 1]],
            # NE
            [[0, 0, 1], [0, -2, 0], [1, 0, 0]],
        ]
    elif grad_name == 'laplacian2':  # X,Y Direction
        kernel = [[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]
    elif grad_name == 'laplacian4':  # All Direction
        kernel = [[[1, 1, 1], [1, -8, 1], [1, 1, 1]]]
    # Sobel operator
    elif grad_name == 'sobel':
        kernel = [
            # Y
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            # X
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        ]
    # Prewitt operator
    elif grad_name == 'prewitt':
        kernel = [
            # Y
            [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
            # X
            [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
        ]
    # Robinson operator
    elif grad_name == 'robinson':
        kernel = [
            # Y
            [[1, 1, 1], [1, -2, 1], [-1, -1, -1]],
            # X
            [[1, 1, -1], [1, -2, -1], [1, 1, -1]],
        ]
    # Roberts
    elif grad_name == 'roberts':
        kernel = [
            # NW
            [[-1, 0], [0, 1]],
            # NE
            [[0, -1], [1, 0]],
        ]
    # Kirsch operator
    elif grad_name == 'kirsch':
        kernel = [
            # Y
            [[-5, -5, -5], [3, 0, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 0, 3], [-5, -5, -5]],
            # X
            [[-5, 3, 3], [-5, 0, 3], [-5, 3, 3]],
            [[3, 3, -5], [3, 0, -5], [3, 3, -5]],
            # NW
            [[-5, -5, 3], [-5, 0, 3], [3, 3, 3]],
            [[3, 3, 3], [3, 0, -5], [3, -5, -5]],
            # NE
            [[3, -5, -5], [3, 0, -5], [3, 3, 3]],
            [[3, 3, 3], [-5, 0, 3], [-5, -5, 3]],
        ]
    # Scharr operator
    elif grad_name == 'scharr':
        # kernel = [  ]
        kernel = [
            # Y
            [[-3, -10, -3], [0, 0, 0], [-3, 10, -3]],
            # X
            [[-3, 0, -3], [-10, 0, -10], [-3, 0, -3]],
        ]
    # Laplacian of Gaussian
    elif grad_name == 'LoG':
        kernel = [
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0],
            ],
        ]
    else:
        s = ', '.join(get_gradient_operator_names())
        alert = ''.join((
            f'"{grad_name}" is not valid.',
            f'Check valid method with get_gradient_operator_names(): {s}.',
        ))
        raise ValueError('\n'.join(textwrap.wrap(alert, 80)))
    return [np.array(ker, dtype=np.float32) for ker in kernel]


@boundscheck(False)
@wraparound(False)
def get_line_operator_name():
    """Get a list of availible line-detection operator name.

    Returns
    -------
    output : tuple[str]
        List of availible line-detection operator name.
    """
    return (
        'vertical',
        'horizontal',
        '+45',
        '-45',
    )


@boundscheck(False)
@wraparound(False)
def get_line_operator(op_name: str) -> Arr32F2D:
    """Get a list of availible line-detection operator name.

    Parameters
    ----------
    op_name : str
        gradient operator name

    Returns
    -------
    output : KER_32F2D
        Line-detection operator

    Raises
    ------
    ValueError
        grad_name not in get_line_operator_name().
    """
    # vertical
    if op_name == 'vertical':
        kernel = [
            [-1, -1, -1],
            [2, 2, 2],
            [-1, -1, -1],
        ]
    elif op_name == 'horizontal':
        kernel = [
            [-1, 2, -1],
            [-1, 2, -1],
            [-1, 2, -1],
        ]
    elif op_name == '+45':
        kernel = [
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2],
        ]
    elif op_name == '-45':
        kernel = [
            [-1, -1, 2],
            [-1, 2, -1],
            [2, -1, -1],
        ]
    else:
        s = ', '.join(get_line_operator_name())
        alert = ''.join((
            f'"{op_name}" is not valid.',
            f'Check valid method with get_line_operator_name(): {s}.',
        ))
        raise ValueError('\n'.join(textwrap.wrap(alert, 80)))
    return np.array(kernel, dtype=np.float32)


@boundscheck(False)
@wraparound(False)
def gradient_norm(
    edges: list[IMG_ARRAY], ntype: int = 0, pos_only: bool = False
) -> IMG_32F:
    """Combining multiple gradients by taking norm. The norm of image edges
    will calculate along axis=0.

    Parameters
    ----------
    edges : list[IMG_ARRAY]
        A list or array of results of gradient.
        [
            convolve(img, kernel1), convolve(img, kernel2), ...,
            convolve(img, kerneln)
        ]
    ntype : {0, 1, 2, 3}, default=0
        Type of norm.
            0 = sup-norm.
            1 = 1-norm.
            2 = 2-norm(Euclidean norm).
            3 = Average of absolute value.
    pos_only : bool, default=False
        Only take the positive value of edges.

    returns
    -------
    output : ARR_32F
    """
    if isinstance(edges, list):
        edges = np.array(edges, dtype=np.float32)
    if pos_only:  # Only take the part that edges > 0.
        edges[edges < 0] = 0
    if ntype == 0:
        return np.max(np.abs(edges), axis=0)
    if ntype == 1:
        return np.sum(np.abs(edges), axis=0, dtype=np.float32)
    if ntype == 2:
        return np.sqrt(
            np.sum(np.power(edges, 2, dtype=np.float32), axis=0, dtype=np.float64),
            dtype=np.float32,
        )
    elif ntype == 3:
        return np.mean(np.abs(edges), axis=0, dtype=np.float32)


@boundscheck(False)
@wraparound(False)
def gradient_uint8_overflow(
    norm: IMG_32F,
    otype: Literal[0, 1] = 0,
    percent: int | float = 1,
    dtype: np.dtype = np.float32,
) -> IMG_ARRAY:
    """Deal with out of range of [0,255].

    Parameters
    ----------
    norm : ARR_32F
        The image after taking norm by `gradient_norm`. The value should be
        non-negative.
    otype : {0, 1}, default=0
        How to deal with out of range.
        0 = Clip the image to [0,255] directly.
        1 = Normalization to [0, 255] by following steps:
            1. output = 255 * (norm-minimum)/((maximum-minimum)*p), where
               p = percent / 100.
            2. output[output>255] = 255.
            Values greater than p*maximum + (1-p)*minimum will be set to 255
    percent : float
        Convex combination weight.
    dtype : np.dtype, default=np.float32
        output dtype.

    Returns
    -------
    output : IMG_ARRAY
    """
    if otype == 0:  # Clip to [0, 255]
        if dtype == np.uint8:
            return cv2.convertScaleAbs(norm)
        else:
            norm[norm > 255] = 255
            return norm
    percent /= 100
    if otype == 1:  # Compress the dynamic range
        maximum = np.max(norm)
        minimum = np.min(norm)
        ratio = 255 / ((maximum - minimum) * percent)
        if dtype == np.uint8:
            return cv2.convertScaleAbs(norm, alpha=ratio, beta=-ratio * minimum)
        else:
            scaled = np.multiply(norm - minimum, ratio, dtype=dtype)
            scaled[scaled > 255] = 255
            return scaled
    elif otype == 2:  # Scaling the range to [0,255].
        maximum = np.max(norm)
        minimum = np.min(norm)
        if dtype == np.uint8:
            return cv2.convertScaleAbs(
                norm,
                alpha=255 / (maximum - minimum),
                beta=-255 * minimum / (maximum - minimum),
            )
        else:
            return np.multiply(norm - minimum, 255 / (maximum - minimum), dtype=dtype)


@njit(
    ['UniTuple(float32[:,:],2)(UniTuple(uint16,2),float32,float32)'],
    nogil=True,
    cache=True,
    fastmath=True,
)
@wraparound(False)
def get_derivatives_of_gaussian(
    ksize: KER_SIZE, sigma_y: float, sigma_x: float
) -> tuple[Arr32F2D, Arr32F2D]:
    """The kernels of partial derivatives of an 2D gaussian function.

    Parameters
    ----------
    ksize : tuple[int, int]
        Size of kernel. Both number must be odd.
    sigma_y : float
        The parameter sigma of 2D gaussian function along y-direction.
    sigma_x : float
        The parameter sigma of 2D gaussian function along x-direction.

    Returns
    -------
    kernel_y : KER_32F2D
        The partial derivative with respect to y.
    kernel_x : KER_32F2D
        The partial derivative with respect to x.
    """
    kernel_y = np.empty(ksize, dtype=np.float32)  # ∂G/∂y
    kernel_x = np.empty(ksize, dtype=np.float32)  # ∂G/∂x

    khalf_y = ksize[0] // 2
    khalf_x = ksize[1] // 2
    # coefficient outside exp()
    c1_y = np.divide(-1, sigma_y**2)
    c1_x = np.divide(-1, sigma_x**2)
    # coefficient inside exp()
    c2_y = np.divide(-0.5, sigma_y**2)
    c2_x = np.divide(-0.5, sigma_x**2)

    for y in range(ksize[0]):
        exp_temp = (khalf_y - y) ** 2 * c2_y  # Inside exp()
        part_y = (khalf_y - y) * c1_y  # The value relate to ∂G/∂y
        for x in range(khalf_x):
            part_x = (khalf_x - x) * c1_x  # The value relate to ∂G/∂x
            exp_ = np.exp(((khalf_x - x) ** 2) * c2_x + exp_temp)
            kernel_y[y, x] = part_y * exp_
            kernel_x[y, x] = part_x * exp_
    return kernel_y, kernel_x


@njit(
    ['float32[:,:](UniTuple(uint16,2),float32,float32)'],
    nogil=True,
    cache=True,
    fastmath=True,
)
@wraparound(False)
def get_difference_gaussian(ksize: KER_SIZE, sigma1: float, sigma2: float) -> Arr32F2D:
    """The difference of two 2D gaussian function.
        D(y,x) = G_sigma1(y,x) - G_sigma2(y,x),
    where G_sigma is an 2D gaussian function with parameter sigma.

    Parameters
    ----------
    ksize : tuple[int, int]
        Size of kernel. Both number must be odd.
    sigma1 : float
        The parameter of first gaussian function.
    sigma2 : float
        The parameter of second gaussian function.

    Returns
    -------
    kernel : KER_32F2D
        The difference of two 2D gaussian function.
    """
    kernel = np.empty(ksize, dtype=np.float32)

    khalf_y = ksize[0] // 2
    khalf_x = ksize[1] // 2
    # (outside, inside) constants of first/second gaussian function
    consts1 = (-1 / (2 * pi * sigma1**2), -1 / (2 * sigma1**2))
    consts2 = (-1 / (2 * pi * sigma2**2), -1 / (2 * sigma2**2))
    for y in range(ksize[0]):
        dist_y = (y - khalf_y) ** 2  # y-component of dist
        for x in range(ksize[1]):
            dist = dist_y + (x - khalf_x) ** 2  # Term about dist
            first = consts1[0] * np.exp(dist * consts1[1])
            kernel[y, x] = first - consts2[0] * np.exp(dist * consts2[1])
    kernel -= np.sum(kernel) / kernel.size  # The sum of edge detection kernel is zero.
    return kernel


@njit(
    ['float32[:,:](UniTuple(uint16,2),float32)'], nogil=True, cache=True, fastmath=True
)
@wraparound(False)
def get_laplacian_gaussian(ksize: KER_SIZE, sigma: float):
    """Return the negative laplacian of an 2D gaussian function (LoG), -ΔG.

    Parameters
    ----------
    ksize : tuple[int, int]
        Size of kernel. Both number must be odd.
    sigma : float
        The parameter of gaussian function.

    Returns
    -------
    kernel : KER_32F2D
        The difference of two 2D gaussian function.
    """
    kernel = np.empty(ksize, dtype=np.float32)

    khalf_y = ksize[0] // 2
    khalf_x = ksize[1] // 2
    sigma2 = 1 / (2 * sigma**2)
    sigma4 = 1 / (pi * sigma**4)
    total = 0
    for y in range(ksize[0]):
        dist_y = (y - khalf_y) ** 2
        for x in range(ksize[1]):
            dist = (dist_y + (x - khalf_x) ** 2) * sigma2
            kernel[y, x] = sigma4 * (1 - dist) * np.exp(-dist)
            total += kernel[y, x]
    total /= ksize[0] * ksize[1]
    total = np.float32(total)
    return kernel - total


@boundscheck(False)
@wraparound(False)
def laplacian(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0, 1, 2, 3] | None = 2,
) -> IMG_32F | list[IMG_32F]:
    """Taking gradient by convolving with Laplace operators of two directions:
        [
            [[ 1], [-2], [ 1]], # Vertical
            [[1, -2,  1]], # Horizontal
        ].

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.

    Returns
    -------
    output : ARR_32F | list[ARR_32F]
    """
    shape = img.shape
    edges = np.empty([2, *shape], dtype=np.float32)

    for i, kernel in enumerate(get_gradient_operator('laplacians')):
        edges[i] = cv2.filter2D(img, CV_32F, kernel)
    if norm_type is not None:
        output = gradient_norm(edges, norm_type)

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def laplacian_all(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0, 1, 2, 3] | None = 2,
) -> IMG_32F | list[IMG_32F]:
    """Taking gradient by convolving with Laplace operators of four directions:
        [
            [[ 1], [-2], [ 1]], # Vertical
            [[1, -2,  1]], # Horizontal
            # -45
            [[1,  0,  0],
             [0, -2,  0],
             [0,  0,  1]],
            # +45
            [[0,  0,  1],
             [0, -2,  0],
             [1,  0,  0]]
        ].

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.

    Returns
    -------
    output : ARR_32F | list[ARR_32F]
    """
    shape = img.shape
    edges = np.empty([2, *shape], dtype=np.float32)

    for i, kernel in enumerate(get_gradient_operator('laplacians_all')):
        edges[i] = cv2.filter2D(img, CV_32F, kernel)
    if norm_type is not None:
        output = gradient_norm(edges, norm_type)

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def laplacian2(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0] | None = 0,
) -> IMG_32F:
    """Taking gradient by convolving with an 2D Laplace operator:
        [
            [[0,  1,  0],
             [1, -4,  1],
             [0,  1,  0]]
        ].

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, None}, default=0
        Taking absolute value if norm_type == 0. Doing nothin if
        norm_type == None.

    Returns
    -------
    output : ARR_32F
    """
    # Only Y,X Direction # fastest of laplacian
    kernel = get_gradient_operator('laplacian2')[0]
    edge = cv2.filter2D(img, CV_32F, kernel)
    if norm_type == 0:
        output = np.abs(edge)
    else:
        output = edge

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def laplacian4(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0] | None = 0,
) -> IMG_32F:
    """Taking gradient by convolving with an 2D Laplace operator with diagonals
    included:
        [
            [[1,  1,  1],
             [1, -8,  1],
             [1,  1,  1]]
        ].

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, None}, default=0
        Taking absolute value if norm_type == 0. Doing nothin if
        norm_type == None.

    Returns
    -------
    output : ARR_32F
    """
    # All Direction
    kernel = get_gradient_operator('laplacian4')[0]
    edge = cv2.filter2D(img, CV_32F, kernel)
    if norm_type == 0:
        output = np.abs(edge)
    else:
        output = edge

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def sobel(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0, 1, 2, 3] | None = 2,
) -> IMG_32F | list[IMG_32F]:
    """Taking gradient by convolving with 2 Sobel operators.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.

    Returns
    -------
    output : ARR_32F | list[ARR_32F]
    """
    shape = img.shape
    edges = np.empty([2, *shape], dtype=np.float64)

    kernels = get_gradient_operator('sobel')

    for i, ker in enumerate(kernels):
        edges[i] = cv2.filter2D(img, CV_32F, ker)
    if norm_type is not None:
        output = gradient_norm(edges, norm_type)

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def kirsch(
    img: IMG_ARRAY,
    threshold: int | float = 0,
    norm_type: Literal[0, 1, 2, 3] | None = 2,
) -> IMG_32F | list[IMG_32F]:
    """Taking gradient by convolving with 8 Kirsch operators.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.

    Returns
    -------
    output : ARR_32F | list[ARR_32F]
    """
    shape = img.shape

    kernels = get_gradient_operator('kirsch')
    edges = np.empty([len(kernels), *shape], dtype=np.float64)

    for i, ker in enumerate(kernels):
        edges[i] = cv2.filter2D(img, CV_32F, ker)

    if norm_type is not None:
        output = gradient_norm(edges, norm_type)

    if threshold > 0:
        np.copyto(output, 0, where=output < threshold)
    return output


@boundscheck(False)
@wraparound(False)
def gaussian_gradient(
    img: IMG_ARRAY,
    ksize: int | KER_SIZE = 3,
    sigma_y: float = 1,
    sigma_x: float = 1,
    threshold: int | float = 0,
    norm_type: Literal[0, 1, 2, 3] | None = 2,
) -> IMG_32F | list[IMG_32F]:
    """Edge detection by convolving image with partial derivatives of
    gaussian function.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | tuple[int, int]
        Size of kernel. Must be odd number or a tuple of two odd numbers.
    sigma_y : float
        The parameter sigma of 2D gaussian function along y-direction.
    sigma_x : float
        The parameter sigma of 2D gaussian function along x-direction.
    threshold : int | float, default=0
        Threshld value. Thresholding will apply after norm.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.

    Returns
    -------
    output : KER_32F2D | list[KER_32F2D]
    """
    ksize = __valid_convolution_ksize(ksize)
    sigma_y, sigma_x = auto_gaussian_kernel_sigma(ksize, sigma_y, sigma_x)
    kernel_y, kernel_x = get_derivatives_of_gaussian(ksize, sigma_y, sigma_x)
    edges = np.array(
        (cv2.filter2D(img, CV_32F, kernel_y), cv2.filter2D(img, CV_32F, kernel_x)),
        np.float32,
    )
    if norm_type is not None:
        output = gradient_norm(edges, norm_type)

    if threshold > 0:
        np.copyto(output, 0, where=output <= threshold)
    return output


@boundscheck(False)
@wraparound(False)
def marr_hildreth(
    img: IMG_ARRAY,
    ksize: int | KER_SIZE = 3,
    sigma: float = 0,
    norm_type: Literal[0, 1, 2, 3] = 2,
    dtype: np.dtype = np.float32,
):
    """Edge detection by Marr-Hildreth's method.

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int | tuple[int, int]
        Size of kernel. Must satisfies one of following conditions:
        1. int and odd. If ksize < 1 and sigma <= 0, then set ksize = 3. If
           ksize < 1 sigma > 0, then ksize will be computed from sigma.
        2. Two numbers that both satisfy condition 1.
    sigma : float
        The parameter sigma of gaussian function along y-direction. If
        sigma_y <= 0, then sigma_y will be computed from ksize.
    norm_type : {0, 1, 2, 3, None}, default=2
        Combining multiple gradients by taking norm if norm_type is not None.
    dtype : np.dtype, default=np.float32
        Output dtype.

    Returns
    -------
    output : KER_32F2D
    """
    # gaussian_blur will check arguments
    blured = gaussian_blur(img, ksize, sigma, sigma)

    edges = sobel(blured, norm_type=norm_type)
    return gradient_uint8_overflow(edges, otype=0, dtype=dtype)


## Image sharpening
# Unsharp Mask
@boundscheck(False)
@wraparound(False)
def unsharp_masking(
    img: IMG_ARRAY,
    ksize: int | KER_SIZE = (5, 5),
    blurring: Literal['mean', 'gaussian'] = 'gaussian',
    amount: float = 1.0,
    sigma_y: float = 1,
    sigma_x: float = 1,
) -> IMG_ARRAY:
    """Unsharp masking (USM). Sharpening an image by using the difference
    between the original image and its blurred version.
        USM(img) = img + amount*(img-blurred)

    Parameters
    ----------
    img : IMG_ARRAY
        Input image.
    ksize : int
        Size of kernel. Must be a positive odd number.
    blurring : {'mean', 'gaussian'}, default='gaussian'.
        Method of image bluring.
    amount : float, default=1
        Increasing amount of edge. Usually less than 4.5.
    sigma_y : float
        sigma_y in gaussian blur. Only works when blurring == 'gaussian'.
    sigma_x : float
        sigma_x in gaussian blur. Only works when blurring == 'gaussian'.

    Returns
    -------
    output : IMG_ARRAY
        output image.
    """
    if blurring == 'mean':
        ksize = __valid_convolution_ksize(ksize)
        kernel = get_mean_kernel(ksize)
    elif blurring == 'gaussian':
        ksize, sigma_y, sigma_x = check_gaussian_kernel_arg(ksize, sigma_y, sigma_x)
        kernel = get_2d_gaussian_kernel(ksize, sigma_y, sigma_x)
    # Convolution is a linear operator. Hence the formula can be rewrite as
    # convolution with the following kernel.
    impulse = np.zeros(ksize, dtype=np.float32)
    impulse[ksize[0] // 2, ksize[1] // 2] = 1
    kernel = (1 + amount) * impulse - amount * kernel
    #
    sharpened = cv2.filter2D(img, CV_32F, kernel)
    # clip to [0,255]
    sharpened = gradient_uint8_overflow(sharpened, otype=0, dtype=img.dtype)
    return sharpened


# def sharpening(img: IMG_ARRAY, grad: str | KER_2D):
#     kernels = get_gradient_operator(grad)

#     edges = np.empty([len(kernels), *img.shape], dtype=np.float32)
#     for i, ker in enumerate(kernels):
#         edges[i] =  cv2.filter2D(img, CV_32F, ker)
