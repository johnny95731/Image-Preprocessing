__all__ = [
    'get_freq_gaussian_kernel',
    'get_freq_gaussian_kernel_posgaussian_lowpass',
    'gaussian_highpass',
    'gaussian_bandpass',
    'gaussian_bandreject',
    'gaussian_notch_reject',
]

from math import exp
from typing import Tuple, Union


from numba import njit, prange

import numpy as np

# from commons import *
import src.utils.helpers as helpers
from src.utils.helpers import is_iterable
from src.utils.img_type import KER_SIZE, Arr32F1D, Arr32F2D


# In this document, we present some basic Gaussian filters such as Gaussian
# Lowpass Filter, Gaussian highpass filter, Gaussian bandpass Filter,
# Gaussian notch(reject) filter.


# Frequency domain Gaussian filter
__SIGNATURE_GAUSSIAN_KERNEL_SHIFT = [
    'float32[:](int64,float32,float32)',
    'float32[:](int64,float32,Omitted(0))',
]


@njit(__SIGNATURE_GAUSSIAN_KERNEL_SHIFT, nogil=True, cache=True, fastmath=True)
def get_freq_gaussian_kernel(
    size: KER_SIZE, sigma: float, center: float = 0
) -> Arr32F1D:
    """Return the 1D Gaussian filter.
        `filter(u) = exp(-(u-center)**2 / (2*sigma**2))`, where
    u is frequency (not index), sigma is the measure of dispersion, and center
    is the translation coefficient.

    The relation between indice and frequencies is:
    -    [0, 1, ...,   size/2-1,     -size/2, ..., -1] if size is even;
    -    [0, 1, ..., (size-1)/2, -(size-1)/2, ..., -1] if size is odd.

    Parameters
    ----------
    size : int
        The size of filter.
    sigma : float
        Measure of dispersion.
    center : float
        Horicontal translation of gaussian function.

    Returns
    -------
    output : KER_32F1D
        The 1D Gaussian filter for frequency domain.
    """
    kernel = np.empty((size), dtype=np.float32)
    const = np.divide(-0.5, np.power(sigma, 2))  # -1 / (2*sigma**2)
    if size % 2:
        quant = size // 2 + 1
    else:
        quant = size // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        kernel[quant] = exp(np.power(quant - center, 2) * const)
    if center == 0:
        for y in range(1, quant):
            val = exp(np.power(y, 2) * const)
            kernel[y] = val
            kernel[-y] = val
        kernel[0] = 1
    else:
        for y in range(-quant + 1, quant):
            kernel[y] = exp(np.power(y - center, 2) * const)
    return kernel


__SIGNATURE_GAUSSIAN_KERNEL_HALF = [
    'float32[:](int64,float32,float32,float32)',
    'float32[:](int64,float32,Omitted(0),float32)',
    'float32[:](int64,float32,float32,Omitted(1))',
    'float32[:](int64,float32,Omitted(0),Omitted(1))',
]


@njit(__SIGNATURE_GAUSSIAN_KERNEL_HALF, nogil=True, cache=True, fastmath=True)
def get_freq_gaussian_kernel_pos(
    size: KER_SIZE, sigma: float, center: float = 0, c: float = 1
) -> Arr32F1D:
    """Return the 1D Gaussian filter with positive frequency only.
        `filter(u) = c * exp(-(u-center)**2 / (2*sigma**2))`, where
    u is frequency, sigma is the measure of dispersion, center is the
    translation coefficient, and c is the scaling coefficient.

    The relation between indice and frequencies is:
    -    [0, 1, ...,   size/2-1] if size is even;
    -    [0, 1, ..., (size-1)/2] if size is odd.

    Parameters
    ----------
    size : int
        The size of filter.
    sigma : float
        Measure of dispersion.
    center : float
        Horicontal translation of gaussian function.
    c : float
        Scaling coefficient.

    Returns
    -------
    output : KER_32F1D
        The 1D Gaussian filter for frequency domain.
    """
    kernel = np.empty((size), dtype=np.float32)
    const = np.divide(-0.5, np.power(sigma, 2))
    for x in range(size):
        kernel[x] = c * exp(np.power(x - center, 2) * const)
    return kernel


def gaussian_lowpass(
    size: KER_SIZE, sigma_y: float, sigma_x: float, parallel: bool = False
) -> Arr32F2D:
    """Return the 2D Gaussian low-pass filter.
        `filter(u,v) = exp(-u**2/(2*sigma_y**2) - v**2/(2*sigma_x**2))`,
    where u,v are frequencies (not indices) and sigma_y and sigma_x are
    measures of dispersion along y-direction and x-direction, respectively.

    Parameters
    ----------
    size : int
        The size of filter.
    sigma_y : float
        Measure of dispersion along y-direction.
    sigma_x : float
        Measure of dispersion along x-direction.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian low-pass filter.
    """
    kernel_y = get_freq_gaussian_kernel(size[0], sigma_y)
    kernel_x = get_freq_gaussian_kernel_pos(size[1], sigma_x)
    if parallel:
        return helpers.nb_outer(kernel_y, kernel_x)
    else:
        return np.outer(kernel_y, kernel_x)


def gaussian_highpass(
    size: KER_SIZE, sigma_y: float, sigma_x: float, parallel: bool = False
) -> Arr32F2D:
    """
    Return the 2D Gaussian high-pass filter.
        `filter(u,v) = 1 - exp(-u**2/(2*sigma_y**2) - v**2/(2*sigma_x**2))`,
    where u,v are frequencies (not indices) and sigma_y and sigma_x are
    measures of dispersion along y-direction and x-direction, respectively.

    Parameters
    ----------
    size : int
        The size of filter.
    sigma_y : float
        Measure of dispersion along y-direction.
    sigma_x : float
        Measure of dispersion along x-direction.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian low-pass filter.
    """
    kernel_y = get_freq_gaussian_kernel(size[0], sigma_y)
    kernel_x = get_freq_gaussian_kernel_pos(size[1], sigma_x)
    if parallel:
        return helpers.outer_bias_parallel(kernel_y, kernel_x, 1.0)
    else:
        return helpers.outer_bias(kernel_y, kernel_x, 1)


__SIGNATURE_GAUSSIAN_BANDPASS = [
    'float32[:,:](UniTuple(int64,2),float32,float32)',
]


@njit(__SIGNATURE_GAUSSIAN_BANDPASS, nogil=True, cache=True, fastmath=True)
def gaussian_bandpass_nonparallel(
    size: KER_SIZE, band_center: float, band_width: float
) -> Arr32F2D:
    """Return the 2D Gaussian band-pass filter.
        `filter(u,v) = exp{-[ (D(u,v)**2-(C_0)**2) / (D(u,v)W) ]**2}`,
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will `not` apply parallel computation.

    Parameters
    ----------
    size : int
        The size of filter.
    sigma_y : float
        Measure of dispersion along y-direction.
    sigma_x : float
        Measure of dispersion along x-direction.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian band-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation.
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    c_center = np.power(band_center, 2)
    c_width = np.power(band_width, 2)

    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x  # D**2
        center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
        width_term = dist * c_width  # (D*W)**2
        output[quant] = np.exp(-np.divide(center_term, width_term))
    for y in range(1, quant):
        dist = np.power(y, 2) + square_x
        center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
        width_term = dist * c_width  # (D*W)**2
        mask_val = np.exp(-np.divide(center_term, width_term))
        output[y] = mask_val
        output[-y] = mask_val
    dist = square_x
    center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
    width_term = dist * c_width  # (D*W)**2
    output[0] = np.exp(-np.divide(center_term, width_term))
    output[0, 0] = 0  # exp(-inf)
    return output


@njit(
    __SIGNATURE_GAUSSIAN_BANDPASS, nogil=True, cache=True, fastmath=True, parallel=True
)
def gaussian_bandpass_parallel(
    size: KER_SIZE, band_center: float, band_width: float
) -> Arr32F2D:
    """Return the 2D Gaussian band-pass filter.
        `filter(u,v) = exp{-[ (D(u,v)**2-(C_0)**2) / (D(u,v)W) ]**2}`,
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will apply parallel computation.

    Parameters
    ----------
    size : int
        The size of filter.
    band_center : float
        The distance between origin and center of band.
    band_width : float
        Width that allow frequencies paas.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian band-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation.
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    c_center = np.power(band_center, 2)
    c_width = np.power(band_width, 2)

    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x  # D**2
        center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
        width_term = dist * c_width  # (D*W)**2
        output[quant] = np.exp(-np.divide(center_term, width_term))
    for y in prange(1, quant):
        dist = np.power(y, 2) + square_x  # D**2
        center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
        width_term = dist * c_width  # (D*W)**2
        mask_val = np.exp(-np.divide(center_term, width_term))
        output[y] = mask_val
        output[-y] = mask_val
    dist = square_x
    center_term = np.power(dist - c_center, 2)  # (D**2-(c_0)**2)**2
    width_term = dist * c_width  # (D*W)**2
    output[0] = np.exp(-np.divide(center_term, width_term))
    output[0, 0] = 0  # exp(-inf)
    return output


def gaussian_bandpass(
    size: KER_SIZE, band_center: float, band_width: float, parallel: bool = False
) -> Arr32F2D:
    """Return the 2D Gaussian band-pass filter.
        `filter(u,v) = exp{-[ (D(u,v)**2-(C_0)**2) / (D(u,v)W) ]**2}`,
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will apply parallel computation.

    Parameters
    ----------
    size : int
        The size of filter.
    band_center : float
        The distance between origin and center of band.
    band_width : float
        Width that allow frequencies paas.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian band-pass filter.
    """
    if parallel:
        return gaussian_bandpass_parallel(size, band_center, band_width)
    else:
        return gaussian_bandpass_nonparallel(size, band_center, band_width)


@njit(
    __SIGNATURE_GAUSSIAN_BANDPASS, nogil=True, cache=True, fastmath=True, parallel=True
)
def gaussian_bandreject_parallel(
    size: KER_SIZE, band_center: float, band_width: float
) -> Arr32F2D:
    """Return the 2D Gaussian band-pass filter.
        `filter(u,v) = 1 - exp{-[ (D(u,v)**2-(C_0)**2) / (D(u,v)W) ]**2}`,
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will apply parallel computation.

    Parameters
    ----------
    size : int
        The size of filter.
    band_center : float
        The distance between origin and center of band.
    band_width : float
        Width that allow frequencies paas.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian band-pass filter.
    """
    return np.subtract(1, gaussian_bandpass_parallel(size, band_center, band_width))


def gaussian_bandreject(
    size: KER_SIZE, band_center: float, band_width: float, parallel: bool = False
) -> Arr32F2D:
    """Return the 2D Gaussian band-pass filter.
        `filter(u,v) = 1 - exp{-[ (D(u,v)**2-(C_0)**2) / (D(u,v)W) ]**2}`,
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.

    Parameters
    ----------
    size : int
        The size of filter.
    band_center : float
        The distance between origin and center of band.
    band_width : float
        Width that allow frequencies paas.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The 2D Gaussian band-pass filter.
    """
    if parallel:
        return gaussian_bandreject_parallel(size, band_center, band_width)
    else:
        return np.subtract(
            1, gaussian_bandpass_nonparallel(size, band_center, band_width)
        )


# Notch-Reject Filter
@njit(
    [
        'float32[:,:](UniTuple(int64,2),float32,UniTuple(float32,2))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def gaussian_single_notch_reject(
    size: KER_SIZE, sigma: float, center: Tuple[float, float]
) -> Arr32F2D:
    """Return a Gaussian notch-reject filter that center at `center`.
        `filter(u,v) = 1 - exp(-(D(u-p_y, v-p_x)**2 / (2*sigma**2))`
    where D is the Euclidean norm, u,v are frequencies (not indices),
    (p_y, p_x) is center of notch, and sigma is measure of dispersion.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    sigma : float
        Cutoff frequency.
    center : Tuple[float, float]
        Center of notch.

    Returns
    -------
    output : ARR_32FC1
        The Gaussian notch-reject filter.
    """
    kernel_y = get_freq_gaussian_kernel(size[0], sigma, center[0])
    kernel_x = get_freq_gaussian_kernel_pos(size[1], sigma, center[1])
    return helpers.outer_bias_parallel(kernel_y, kernel_x, 1)


@njit(
    [
        'float32[:,:](UniTuple(int64,2),float32,UniTuple(float32,2))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
def gaussian_pair_notch_reject(
    size: KER_SIZE, sigma: float, center: Tuple[float, float]
) -> Arr32F2D:
    """Return the product of a pair of Gaussian notch-reject filter with
    order `n`. The centers of two notches is at `center` and `-center`,
    respectively. The pair is required since the Fourier transform of an 2D
    real-valued function, say, F, has the property:
        `F(u,v) = conjugate(F(-u,-v))`.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    sigma : float
        Cutoff frequency.
    center : Tuple[float, float]
        Center of notch.

    Returns
    -------
    output : ARR_32FC1
        The product of a pair of Gaussian notch-reject filters.
    """
    kernel1 = gaussian_single_notch_reject(size, sigma, center)
    kernel2 = gaussian_single_notch_reject(size, sigma, (-center[0], -center[1]))
    return np.multiply(kernel1, kernel2)


def gaussian_notch_reject(
    size: KER_SIZE,
    cutoff: Union[float, Tuple[float]],
    centers: Union[float, Tuple[Tuple[float, float]]],
) -> Arr32F2D:
    """Return the product of multi-pairs of Gaussian notch-reject filters.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float | Tuple[float]
        Cutoff frequencies.
    centers : Tuple[float, float] | Tuple[Tuple[float,float]]
        Centers of notches.

    Returns
    -------
    output : ARR_32FC1
        The product of multi-pairs of Gaussian notch-reject filters.
    """
    is_iters = [is_iterable(cutoff), is_iterable(centers[0])]
    if np.any(is_iters):
        # Find minimum length of iterable arguments
        n1 = len(cutoff) if is_iters[0] else np.inf
        n2 = len(centers) if is_iters[1] else np.inf
        n = min((n1, n2))
        if not is_iters[0]:
            cutoff = [cutoff for _ in range(n)]
        if not is_iters[1]:
            centers = [centers for _ in range(n)]
        # The first pair.
        output = gaussian_pair_notch_reject(size, cutoff[0], centers[0])
        # The others.
        for i in range(n):
            output = np.multiply(
                output, gaussian_pair_notch_reject(size, cutoff[i], centers[0])
            )
    else:
        output = gaussian_pair_notch_reject(size, cutoff, centers)
    return output
