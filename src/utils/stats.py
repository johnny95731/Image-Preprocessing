__all__ = [
    'histogram_quant_8UC1',
    'histogram_quant_8UC3',
    'histogram_percentage_8UC1',
    'histogram_percentage_8UC3',
    'cumulative_percentage_8UC1',
    'histogram',
    'mean_8UC1',
    'mean_and_var',
    'moments',
    'partition',
    'quick_sort',
    'randomized_partition',
    'randomized_quick_sort',
    'max_and_min_uint8_seq',
    'median',
    'maximum',
    'minimum',
    'max_min',
]

from random import randint
from typing import Tuple, overload

import numpy as np


from numba import njit

from src.utils.img_type import (
    Arr8U2D,
    Arr8U3D,
    Arr32U1D,
    Arr32F1D,
    Arr32U2D,
    Arr32F2D,
    Arr8U1D,
    Array1D,
    Array2D,
    Array3D,
)


# Histogram, CDF
@njit(
    [
        'uint32[:](uint8[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def histogram_quant_8UC1(img: Arr8U2D) -> Arr32U1D:
    """Calculate the histogram of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    counts : ARR_32U1D
        The quantity of intensity of img.
    """
    counts = np.zeros(256, dtype=np.uint32)
    for row in img:
        for val in row:
            counts[val] += 1
    return counts


@njit(
    [
        'uint32[:,:](uint8[:,:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def histogram_quant_8UC3(img: Arr8U3D) -> Arr32U2D:
    """Calculate the histogram of an 8UC3 image.

    Parameters
    ----------
    img : ARR_8UC3
        An 8UC3 image.

    Returns
    -------
    counts : ARR_32F1D
        The quantity of intensity of img. The first dimension is channel. The
    second dimension is intensity.
    """
    counts = np.zeros((3, 256), dtype=np.uint32)
    for row in img:
        for val in row:
            counts[0, val[0]] += 1
            counts[1, val[1]] += 1
            counts[2, val[2]] += 1
    return counts


@njit(
    [
        'float32[:](uint8[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def histogram_percentage_8UC1(img: Arr8U2D) -> Arr32F1D:
    """Calculate the percentage histogram of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    counts : ARR_32F1D
        The percentage of each intensity of img.
    """
    # parallel=True is slower than parallel=False
    counts = np.zeros(256, dtype=np.float32)
    const = 1 / img.size

    for row in img:
        for val in row:
            counts[val] += 1
    counts *= const
    return counts


@njit(
    [
        'float32[:,:](uint8[:,:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def histogram_percentage_8UC3(img: Arr8U3D) -> Arr32F2D:
    """Calculate the percentage histogram of an 8UC3 image.

    Parameters
    ----------
    img : ARR_8UC3
        An 8UC3 image.

    Returns
    -------
    counts : ARR_32F2D
        The percentage of each intensity of img. The first dimension is
    channel. The second dimension is intensity.
    """
    # parallel=True is slower than parallel=False
    size = img.shape
    counts = np.zeros((3, 256), dtype=np.float32)
    const = np.divide(1, size[0] * size[1])

    for row in img:
        for val in row:
            counts[0, val[0]] += 1
            counts[1, val[1]] += 1
            counts[2, val[2]] += 1
    counts *= const
    return counts


@njit(
    [
        'float32[:](uint8[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def cumulative_percentage_8UC1(img: Arr8U2D) -> Arr32F1D:
    """Return the cumulative distribution function (cdf) of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    cdf : ARR_32F1D
        The cdf of img.
    """
    cdf = np.zeros(256, dtype=np.float32)
    recip_total = 1 / img.size
    # Counting
    for row in img:
        for val in row:
            cdf[val] += 1
    # Calc cumsum
    cdf[0] *= recip_total
    for i in range(1, 255):
        cdf[i] *= recip_total
        cdf[i] += cdf[i - 1]
    cdf[-1] = 1
    return cdf


# fmt: off
@overload
def histogram(img: Array2D, density: bool) -> Arr32U1D | Arr32F1D: pass
@overload
def histogram(img: Array3D, density: bool) -> Arr32U2D | Arr32F2D: pass
@overload
def histogram(img: Array1D, density: bool) -> None: pass
# fmt: on


def histogram(img, density: bool = False):
    if img.ndim == 2 and not density:
        return histogram_quant_8UC1(img)
    elif img.ndim == 2 and density:
        return histogram_percentage_8UC1(img)

    if img.ndim == 3 and not density:
        return histogram_quant_8UC3(img)
    elif img.ndim == 3 and density:
        return histogram_percentage_8UC3(img)
    return None


@njit(['float32(uint8[:,:])'], nogil=True, cache=True, fastmath=True)
def mean_8UC1(img: Arr8U2D) -> np.float32:
    """Calculate the mean value of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    mean_val : float
        The mean value of img.
    """
    mean_val = np.zeros(1, dtype=np.float32)
    hist = histogram_percentage_8UC1(img)
    for i, val in enumerate(hist):
        mean_val[0] += np.multiply(i, val)
    return mean_val[0]  # type: ignore


@njit('float32(uint8[:])', nogil=True, cache=True, fastmath=True)
def mean_seq(seq: Arr8U1D) -> np.float32:
    """Calculate the mean value of an uint8 sequence.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    mean_val : float
        The mean value of img.
    """
    mean_val = 0
    for val in seq:
        mean_val += val
    mean_val /= seq.size
    return np.float32(mean_val)


# fmt: off
__SIGNATURE_MOMENTS = ['float32[:](uint8[:,:])']
# fmt: on
@njit(__SIGNATURE_MOMENTS, nogil=True, cache=True, fastmath=True)
def mean_and_var(img: Arr8U2D) -> Arr32F1D:
    """Calculate the mean(average) and variation of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    output : ndarray
        Index 0 is the mean of img and index 1 is the variation of image.
    """
    output = np.zeros(2, np.float32)  # [0]: mean; [1]: var
    hist = histogram_percentage_8UC1(img)

    # Calc mean
    for i in range(256):
        output[0] += np.multiply(i, hist[i])
    # Calc var
    for i in range(256):
        output[1] += np.multiply(np.power(i - output[0], 2), hist[i])
    return output


@njit(__SIGNATURE_MOMENTS, nogil=True, cache=True, fastmath=True)
def moments(img: Arr8U2D) -> Arr32F1D:
    """Return the mean(average), variance, skewness, and kurtosis of an 8UC1
    image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    output : ndarray
        The array is ordered by moment ordinal: mean, variance, skewness, and
    excess kurtosis.
    """
    output = np.zeros(4, np.float32)
    hist = histogram_percentage_8UC1(img)

    for i, val in enumerate(hist):  # Mean
        output[0] += np.multiply(i, val)
    for i, val in enumerate(hist):  # Variance
        output[1] += (i - output[0]) ** 2 * val
    if output[1]:
        std = np.sqrt(output[1])
        for i, val in enumerate(hist):
            # Skewness
            output[2] += np.power((i - output[0]) / std, 3) * val
            # Excess kurtosis
            output[3] += np.power((i - output[0]) / std, 4) * val
        output[3] -= 3
    else:
        # var=0 implies img is constant-valued.
        output[2:] = 0
    return output


# Sorting
@njit('int64(uint8[:],int64,int64)', nogil=True, cache=True, fastmath=True)
def partition(arr: np.ndarray, low: int, high: int) -> int:
    """Rearrange arr[low:high+1] such that arr[i] <= arr[index] for i < index
    and arr[i] > arr[index] for i > index.
    The value of partition pivot is arr[high] befor partitioning.

    Parameters
    ----------
    arr : ndarray
        An uint8 array.
    low : int
        Starting point for rearranging.
    high: int
        End point (include) for rearranging.

    Returns
    -------
    index : int
        The pivot index.
    """
    pivot = arr[high]
    index = low
    for j in range(low, high):
        if arr[j] <= pivot:
            arr[index], arr[j] = arr[j], arr[index]
            index += 1
    arr[index], arr[high] = arr[high], arr[index]
    return index


@njit('void(uint8[:],int64,int64)', nogil=True, cache=True, fastmath=True)
def __quick_sort(arr: np.ndarray, low: int, high: int) -> None:
    if low < high:
        q = partition(arr, low, high)
        __quick_sort(arr, low, q - 1)
        __quick_sort(arr, q + 1, high)


@njit('uint8[:](uint8[:])', nogil=True, cache=True, fastmath=True)
def quick_sort(arr: np.ndarray) -> Arr8U1D:
    """Sorting array by using quick sort. The process will change original
    array.

    Parameters
    ----------
    arr : ndarray
        An uint8 array.

    Returns
    -------
    arr : ndarray
        Array after sorting. The output and input are same object.
    """
    __quick_sort(arr, 0, len(arr) - 1)
    return arr


@njit('int64(uint8[:],int64,int64)', nogil=True, cache=True, fastmath=True)
def randomized_partition(arr: np.ndarray, low: int, high: int) -> int:
    """Rearrange arr[low:high+1] such that arr[i] <= arr[index] for i < index
    and arr[i] > arr[index] for i > index.
    The value of partition pivot is choosed randomly befor partitioning.

    Parameters
    ----------
    arr : ndarray
        An uint8 array.
    low : int
        Starting point for rearranging.
    high: int
        End point (include) for rearranging.

    Returns
    -------
    index : int
        The pivot index.
    """
    i = randint(low, high)
    arr[high], arr[i] = arr[i], arr[high]
    return partition(arr, low, high)


@njit('void(uint8[:],int64,int64)', nogil=True, cache=True, fastmath=True)
def __randomized_quick_sort(arr: np.ndarray, low: int, high: int) -> None:
    if low < high:
        mid = randomized_partition(arr, low, high)
        __randomized_quick_sort(arr, low, mid - 1)
        __randomized_quick_sort(arr, mid + 1, high)


@njit('uint8[:](uint8[:])', nogil=True, cache=True, fastmath=True)
def randomized_quick_sort(arr: np.ndarray):
    """Sorting array by using randomized quick sort. The process will change
    original array.

    Parameters
    ----------
    arr : ndarray
        An uint8 array.

    Returns
    -------
    arr : ndarray
        Array after sorting. The output and input are same object.
    """
    __randomized_quick_sort(arr, 0, len(arr) - 1)
    return arr


# i-th smallest value searching
_SIGNATURE_MAX_AND_MIN_UINT8_SEQ = ['UniTuple(uint8,2)(uint8[:])']


@njit(_SIGNATURE_MAX_AND_MIN_UINT8_SEQ, nogil=True, cache=True, fastmath=True)
def max_and_min_uint8_seq(seq: np.ndarray):
    """Return the maximum and minimum of a sequences.

    Parameters
    ----------
    arr : ndarray
        An uint8 array.

    Returns
    -------
    output : tuple
        max and min of arr.
    """
    max_ = 0
    min_ = 255

    for val in seq:
        if val > max_:
            max_ = val
        if val < min_:
            min_ = val
    return max_, min_


__SIGNATURE_SEARCHING = [
    'uint8(uint8[:,:])',
]


@njit(__SIGNATURE_SEARCHING, nogil=True, cache=True, fastmath=True)
def median(img: Arr8U2D) -> int:
    """Find median of image by histogram.

    Parameters
    ----------
    img : ndarray
        An 8UC1 image.

    Returns
    -------
    median_ : int
        Median value of img.
    """
    # Get values
    hist = histogram_percentage_8UC1(img)
    cdf = np.zeros(256, dtype=np.float32)
    if hist[0] > 0.5:
        return 0
    cdf[0] = hist[0]
    for i in range(1, 255):
        cdf[i] += hist[i] + cdf[i - 1]
        if cdf[i] > 0.5:
            return i
    return 255


@njit(__SIGNATURE_SEARCHING, nogil=True, cache=True, fastmath=True)
def maximum(img: Arr8U2D) -> int:
    """Find the maximum of an image. Faster than np.max(img).

    Parameters
    ----------
    img : ndarray
        An 8UC1 image.

    Returns
    -------
    max_ : int
        Maximum of img.
    """
    # Get values
    hist = histogram_percentage_8UC1(img)
    for i in range(255, -1, -1):
        if hist[i]:
            return i
    return 0


@njit(__SIGNATURE_SEARCHING, nogil=True, cache=True, fastmath=True)
def minimum(img: Arr8U2D) -> int:
    """Find the minimum of an image. Faster than np.min(img).

    Parameters
    ----------
    img : ndarray
        An 8UC1 image.

    Returns
    -------
    max_ : int
        Minimum of img.
    """
    # Get values
    hist = histogram_percentage_8UC1(img)
    for i in range(256):
        if hist[i]:
            return i
    return 255


__SIGNATURE_MAX_MIN = [
    'UniTuple(uint8,2)(uint8[:,:])',
]


@njit(__SIGNATURE_MAX_MIN, nogil=True, cache=True, fastmath=True)
def max_min(img: Arr8U2D) -> Tuple[int, int]:
    """Find the maximum and minimum of an image.

    Parameters
    ----------
    img : ndarray
        An 8UC1 image.

    Returns
    -------
    output : tuple
        Maximum and minimum of img, (max, min).
    """
    # Get values
    hist = histogram_percentage_8UC1(img)
    maxi = 255
    for i in range(255, -1, -1):
        if hist[i]:
            maxi = i
            break
    for i in range(256):
        if hist[i]:
            return maxi, i
    return 0, 255
