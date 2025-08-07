__all__ = [
    'butterworth_lowpass',
    'butterworth_highpass',
    'butterworth_bandpass',
    'butterworth_bandreject',
    'butterworth_notch_reject',
]

from typing import Tuple, Union

from cython import boundscheck, wraparound
from numba import njit, prange

import numpy as np

from src.utils.helpers import is_iterable
from src.utils.img_type import Arr32F2D, KER_SIZE


# In this document, we present some Butterworth filters such as Butterworth
# Low-pass Filter, Butterworth high-pass filter, Butterworth bandpass Filter,
# Butterworth notch(reject) filter, etc.


__SIGNATURE_BUTTERWORTH_LOWPASS = [
    'float32[:,:](UniTuple(int64,2),float32,float32)',
]


@njit(__SIGNATURE_BUTTERWORTH_LOWPASS, nogil=True, cache=True, fastmath=True)
@wraparound(False)
def butterworth_lowpass_nonparallel(
    size: KER_SIZE, cutoff: float, order: float
) -> Arr32F2D:
    """Return the Butterworth low-pass filter with order `n`.
        `filter(u,v) = 1 / (1+D(u,v)/f_0)**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), and f_0 is
    cutoff frequency.
    This function will `not` apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter Must. be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth low-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = np.power(cutoff, np.multiply(2, order))
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amout for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(np.power(quant, 2) + square_x, order)
        output[quant] = np.divide(const, np.add(const, dist))

    if order != 1:
        for y in range(1, quant):
            dist = np.power(np.power(y, 2) + square_x, order)  # D**(2n)
            mask_val = np.divide(const, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    else:  # Faster since power `order` is reduced.
        for y in range(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            mask_val = np.divide(const, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    # The case of y=0.
    dist = np.power(square_x, order)
    output[0] = np.divide(const, np.add(const, dist))
    return output


@njit(
    __SIGNATURE_BUTTERWORTH_LOWPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_lowpass_parallel(
    size: KER_SIZE, cutoff: float, order: float
) -> Arr32F2D:
    """Return the Butterworth low-pass filter with order `n`.
        `filter(u,v) = 1 / (1+D(u,v)/f_0)**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), and f_0 is
    cutoff frequency.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth low-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation.
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = np.power(cutoff, np.multiply(2, order))
    if size[0] % 2:
        quant = size[0] // 2 + 1
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(np.power(quant, 2) + square_x, order)
        output[quant] = np.divide(const, np.add(const, dist))
    if order != 1:
        for y in prange(1, quant):
            dist = np.power(np.power(y, 2) + square_x, order)  # D**(2n)
            mask_val = np.divide(const, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    else:  # Faster since power `order` is reduced.
        for y in prange(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            mask_val = np.divide(const, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    # The case of y=0.
    dist = np.power(square_x, order)
    output[0] = np.divide(const, np.add(const, dist))
    return output


@njit(
    'float32[:,:](UniTuple(int64,2),float32,float32,boolean)',
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_lowpass(
    size: KER_SIZE, cutoff: float, order: float = 1, parallel: bool = False
) -> Arr32F2D:
    """Return the Butterworth low-pass filter with order `n`.
        `filter(u,v) = 1 / (1+D(u,v)/f_0)**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), and f_0 is
    cutoff frequency.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float, default=1
        Order of Butterworth filter. Must be positive.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth low-pass filter.
    """
    if cutoff <= 0 or order <= 0:
        print('Order and cutoff must be positive.')
        return np.zeros(size, dtype=np.float32)
    if parallel:
        return butterworth_lowpass_parallel(size, cutoff, order)
    else:
        return butterworth_lowpass_nonparallel(size, cutoff, order)


@njit(
    __SIGNATURE_BUTTERWORTH_LOWPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    error_model='numpy',
)
@wraparound(False)
def butterworth_highpass_nonparallel(
    size: KER_SIZE, cutoff: float, order: float
) -> Arr32F2D:
    """Return the Butterworth high-pass filter with order `n`.
        `filter(u,v) = 1 / (1+f_0/D(u,v))**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), f_0 is and
    cutoff frequency.
    This function will `not` apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth high-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = np.power(cutoff, np.multiply(2, order))
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amout for loop.
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(np.power(quant, 2) + square_x, order)
        output[quant] = np.divide(dist, np.add(const, dist))
    if order != 1:
        for y in prange(1, quant):
            dist = np.power(np.power(y, 2) + square_x, order)  # D**(2n)
            mask_val = np.divide(dist, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    else:  # Faster since power `order` is reduced.
        for y in prange(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            mask_val = np.divide(dist, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    # The case of y=0.
    dist = np.power(square_x, order)
    output[0] = np.divide(dist, np.add(const, dist))
    return output


@njit(
    __SIGNATURE_BUTTERWORTH_LOWPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    error_model='numpy',
    parallel=True,
)
@wraparound(False)
def butterworth_highpass_parallel(
    size: KER_SIZE, cutoff: float, order: float
) -> Arr32F2D:
    """Return the Butterworth high-pass filter with order `n`.
        `filter(u,v) = 1 / (1+f_0/D(u,v))**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), and f_0 is
    cutoff frequency.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth high-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = np.power(cutoff, np.multiply(2, order))
    if size[0] % 2:
        quant = size[0] // 2 + 1
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(np.power(quant, 2) + square_x, order)
        output[quant] = np.divide(dist, np.add(const, dist))
    if order != 1:
        for y in prange(1, quant):
            dist = np.power(np.power(y, 2) + square_x, order)  # D**(2n)
            mask_val = np.divide(dist, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    else:  # Faster since power `order` is reduced.
        for y in prange(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            mask_val = np.divide(dist, np.add(const, dist))
            output[y] = mask_val
            output[-y] = mask_val
    dist = np.power(square_x, order)
    output[0] = np.divide(dist, np.add(const, dist))
    return output


@boundscheck(False)
@wraparound(False)
def butterworth_highpass(
    size: KER_SIZE, cutoff: float, order: float = 1, parallel=False
) -> Arr32F2D:
    """Return the Butterworth high-pass filter with order `n`.
        `filter(u,v) = 1 / (1+f_0/D(u,v))**(2n)`, where
    D is the Euclidean norm, u,v are frequencies (not indices), and f_0 is
    cutoff frequency.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float, default=1
        Order of Butterworth filter. Must be positive.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth high-pass filter.
    """
    if cutoff <= 0 or order <= 0:
        print('Order and cutoff must be positive.')
        return
    if parallel:
        return butterworth_highpass_parallel(size, cutoff, order)
    else:
        return butterworth_highpass_nonparallel(size, cutoff, order)


__SIGNATURE_BUTTERWORTH_BANDPASS = [
    'float32[:,:](UniTuple(int64,2),float32,float32,float32)',
]


@njit(
    __SIGNATURE_BUTTERWORTH_BANDPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    error_model='numpy',
)
@wraparound(False)
def butterworth_bandpass_nonparallel(
    size: KER_SIZE, band_center: float, band_width: float, order: float = 1
) -> Arr32F2D:
    """Return the Butterworth bandpass filter with order `n`.
                                       (D(u,v)*W)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will `not` apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that pass frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    order2 = np.multiply(2, order)
    c_width = np.power(band_width, order2)
    c_center = np.power(band_center, 2)
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x
        w = np.multiply(np.power(dist, order), c_width)
        output[quant] = np.divide(w, np.add(w, np.power(dist - c_center, order2)))
    if order != 1:
        for y in range(1, quant):
            dist = np.power(np.power(y, 2) + square_x, 2)
            # w = (dist*bandWidth)**(2*order)
            w = np.multiply(np.power(dist, order), c_width)
            # w / [ w + (dist-constCenter)**(2*order) ]
            mask_value = np.add(w, np.power(dist - c_center, order2))
            mask_value = np.divide(w, mask_value)
            output[y] = mask_value
            output[-y] = mask_value
    else:  # Faster since power `order` is reduced.
        for y in range(1, quant):
            dist = np.power(y, 2) + square_x
            w = np.multiply(dist, c_width)
            mask_value = np.divide(w, np.add(w, np.power(dist - c_center, 2)))
            output[y] = mask_value
            output[-y] = mask_value
    # The case of y=0.
    dist = np.power(square_x, order)
    w = np.multiply(np.power(dist, order), c_width)
    output[0] = np.divide(w, np.add(w, np.power(dist - c_center, order2)))
    return output


@njit(
    __SIGNATURE_BUTTERWORTH_BANDPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_bandpass_parallel(
    size: KER_SIZE, band_center: float, band_width: float, order: float = 1
) -> Arr32F2D:
    """Return the Butterworth bandpass filter with order `n`.
                                       (D(u,v)*W)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that pass frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-pass filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    order2 = np.multiply(2, order)
    c_width = np.power(band_width, order2)
    c_center = np.power(band_center, 2)
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x
        w = np.multiply(np.power(dist, order), c_width)
        output[quant] = np.divide(w, np.add(w, np.power(dist - c_center, order2)))
    # Note that the filter is circular symmetry with frequency center (0,0)
    if order != 1:
        for y in range(1, quant):
            # dist = d(y,x)**2
            # d(y,x) is distence between (y,x) and frequency center.
            dist = np.power(np.power(y, 2) + square_x, 2)
            # w = (dist*bandWidth)**(2*order)
            w = np.multiply(np.power(dist, order), c_width)
            # w / [ w + (dist-constCenter)**(2*order) ]
            mask_value = np.add(w, np.power(dist - c_center, order2))
            mask_value = np.divide(w, mask_value)
            output[y] = mask_value
            output[-y] = mask_value
    else:  # Faster since power `order` is reduced.
        for y in range(1, quant):
            dist = np.power(y, 2) + square_x
            w = np.multiply(dist, c_width)
            mask_value = np.divide(w, np.add(w, np.power(dist - c_center, 2)))
            output[y] = mask_value
            output[-y] = mask_value
    # The case of y=0.
    dist = np.power(square_x, order)
    w = np.multiply(np.power(dist, order), c_width)
    output[0] = np.divide(w, np.add(w, np.power(dist - c_center, order2)))
    return output


@boundscheck(False)
@wraparound(False)
def butterworth_bandpass(
    size: KER_SIZE,
    band_center: float,
    band_width: float,
    order: float = 1,
    parallel: bool = False,
) -> Arr32F2D:
    """Return the Butterworth bandpass filter with order `n`.
                                       (D(u,v)*W)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that pass frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-pass filter.
    """
    if band_center < 0 or band_width <= 0 or order <= 0:
        print('band_center, band_width, and order must be positive.')
        return
    if parallel:
        return butterworth_bandpass_parallel(size, band_center, band_width, order)
    else:
        return butterworth_bandpass_nonparallel(size, band_center, band_width, order)


@njit(__SIGNATURE_BUTTERWORTH_BANDPASS, nogil=True, cache=True, fastmath=True)
@wraparound(False)
def butterworth_bandreject_nonparallel(
    size: KER_SIZE, band_center: float, band_width: float, order: float
):
    """Return the Butterworth band-reject filter with order `n`.
                                  (D(u,v)**2-C_0**2)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will `not` apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that reject frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-reject filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    order2 = np.multiply(2, order)
    c_width = np.power(band_width, order2)
    c_center = np.power(band_center, 2)
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x
        w = np.multiply(np.power(dist, order), c_width)
        d = np.power(dist - c_center, order2)
        output[quant] = np.divide(d, np.add(w, d))
    if order != 1:
        for y in range(1, quant):
            dist = np.power(np.power(y, 2) + square_x, 2)  # D**2
            # w = (dist*bandWidth)**(2*order)
            w = np.multiply(np.power(dist, order), c_width)
            d = np.power(dist - c_center, order2)
            # w / [ w + (dist-constCenter)**(2*order) ]
            mask_value = np.divide(d, np.add(w, d))
            output[y] = mask_value
            output[-y] = mask_value
    else:  # Faster since power `order` is reduced.
        for y in range(1, quant):
            dist = np.power(y, 2) + square_x
            w = np.multiply(dist, c_width)
            d = np.power(dist - c_center, 2)
            mask_value = np.divide(d, np.add(w, d))
            output[y] = mask_value
            output[-y] = mask_value
    # The case of y=0.
    dist = np.power(square_x, order)
    w = np.multiply(np.power(dist, order), c_width)
    d = np.power(dist - c_center, order2)
    output[0] = np.divide(d, np.add(w, d))
    return output


@njit(
    __SIGNATURE_BUTTERWORTH_BANDPASS,
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_bandreject_parallel(
    size: KER_SIZE, band_center: float, band_width: float, order: float = 1
) -> Arr32F2D:
    """Return the Butterworth band-reject filter with order `n`.
                                  (D(u,v)**2-C_0**2)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that reject frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-reject filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    order2 = np.multiply(2, order)
    c_width = np.power(band_width, order2)  # W**(2n)
    c_center = np.power(band_center, 2)  # (C_0)**2
    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(quant, 2) + square_x  # D**2
        center_term = np.power(dist - c_center, order2)  # (D**2-C_0**2)**(2n)
        width_term = np.power(dist, order) * c_width  # (D*W)**(2n)
        output[quant] = np.divide(center_term, width_term + center_term)
    if order != 1:
        for y in prange(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            center_term = np.power(dist - c_center, order2)  # (D**2-C_0**2)**(2n)
            width_term = np.power(dist, order) * c_width  # (D*W)**(2n)
            mask_value = np.divide(center_term, width_term + center_term)
            output[y] = mask_value
            output[-y] = mask_value
    else:  # Faster since power `order` is reduced.
        for y in prange(1, quant):
            dist = np.power(y, 2) + square_x  # D**2
            center_term = np.power(dist - c_center, 2)  # (D**2-C_0**2)**2
            width_term = np.multiply(dist, c_width)  # (D*W)**(2n)
            mask_value = np.divide(center_term, width_term + center_term)
            output[y] = mask_value
            output[-y] = mask_value
    dist = np.power(square_x, order)  # D**2
    center_term = np.power(dist - c_center, order2)  # (D**2-(C_0)**2)**(2n)
    width_term = np.power(dist, order) * c_width  # (D*W)**(2n)
    output[0] = np.divide(center_term, width_term + center_term)
    return output


@boundscheck(False)
@wraparound(False)
def butterworth_bandreject(
    size: KER_SIZE,
    band_center: float,
    band_width: float,
    order: float = 1,
    parallel: bool = False,
) -> Arr32F2D:
    """Return the Butterworth band-reject filter with order `n`.
                                  (D(u,v)**2-C_0**2)**(2n)
        filter(u,v) = -----------------------------------------------,
                        (D(u,v)**2-C_0**2)**(2n) + (D(u,v)*W)**(2n)
    where D is the Euclidean norm, u,v are frequencies (not indices), C_0 is
    band center, and W is band width.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    band_center : float
        The distance between origin and center of band. Must be positive.
    band_width : float
        Width that reject frequencies. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth band-reject filter.
    """
    if band_center < 0 or band_width <= 0 or order <= 0:
        print('band_center, band_width, and order must be positive.')
        return
    if parallel:
        return butterworth_bandreject_parallel(size, band_center, band_width, order)
    else:
        return butterworth_bandreject_nonparallel(size, band_center, band_width, order)


# Notch-Reject Filter
@njit(
    [
        'float32[:,:](UniTuple(int64,2),float32,float32,UniTuple(float32,2))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_single_notch_reject(
    size: KER_SIZE, cutoff: float, order: float, center: Tuple[float, float]
) -> Arr32F2D:
    """Return a Butterworth notch-reject filter with order `n`. The center of
    notch is at `center`.
                            D(u-p_y, v-p_x)**n
        filter(u,v) = -------------------------------,
                        D(u-p_y, v-p_x)**n + f_0**n
    where D is the Euclidean norm, u,v are frequencies (not indices),
    (p_y, p_x) is the center of notch, and f_0 is cutoff frequency.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.
    center : Tuple[float, float]
        Center of notch.

    Returns
    -------
    output : ARR_32FC1
        The Butterworth notch-reject filter.
    """
    output = np.empty(size, dtype=np.float32)
    # Reduce computation
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x - center[1], 2)
    const = np.power(cutoff, order)

    if size[0] % 2:
        quant = size[0] // 2 + 1  # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        dist = np.power(np.sqrt(np.power(quant - center[0], 2) + square_x), order)
        output[quant] = np.divide(dist, np.add(const, dist))
    if order != 1:
        for y in prange(-quant + 1, quant):
            # D(u-p_y, v-p_x)**n
            dist = np.power(np.sqrt(np.power(y - center[0], 2) + square_x), order)
            output[y] = np.divide(dist, np.add(const, dist))
    else:  # Faster since power `order` is reduced.
        for y in prange(-quant + 1, quant):
            # D(u-p_y, v-p_x)**n
            dist = np.sqrt(np.power(y - center[0], 2) + square_x)
            output[y] = np.divide(dist, np.add(const, dist))
    return output


@njit(
    [
        'float32[:,:](UniTuple(int64,2),float32,float32,UniTuple(float32,2))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
    parallel=True,
)
@wraparound(False)
def butterworth_pair_notch_reject(
    size: KER_SIZE, cutoff: float, order: float, center: Tuple[float, float]
) -> Arr32F2D:
    """Return the product of a pair of Butterworth notch-reject filter with
    order `n`. The centers of two notches is at `center` and `-center`,
    respectively. The pair is required since the Fourier transform of an 2D
    real-valued function, say, F, has the property:
        `F(u,v) = conjugate(F(-u,-v))`.
    This function will apply parallel computation.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float
        Order of Butterworth filter. Must be positive.
    center : Tuple[float, float]
        Center of notch.

    Returns
    -------
    output : ARR_32FC1
        The product of a pair of Butterworth notch-reject filters.
    """
    kernel1 = butterworth_single_notch_reject(size, cutoff, order, center)
    kernel2 = butterworth_single_notch_reject(
        size, cutoff, order, (-center[0], -center[1])
    )
    return np.multiply(kernel1, kernel2)


@boundscheck(False)
@wraparound(False)
def butterworth_notch_reject(
    size: KER_SIZE,
    cutoff: Union[float, Tuple[float]],
    orders: Union[float, Tuple[float]],
    centers: Union[Tuple[float, float], Tuple[Tuple[float, float]]],
) -> Arr32F2D:
    """Return the product of multi-pairs Butterworth notch-reject filters.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    cutoff : float | Tuple[float]
        Cutoff frequencies.
    orders : float | Tuple[float]
        Orders of Butterworth filters.
    centers : Tuple[float, float] | Tuple[Tuple[float,float]]
        Centers of notches.

    Returns
    -------
    output : ARR_32FC1
        The product of multi-pairs Butterworth notch-reject filters.
    """
    is_iters = [is_iterable(cutoff), is_iterable(orders), is_iterable(centers[0])]
    if np.any(is_iters):
        # Find minimum length of iterable arguments
        n1 = len(cutoff) if is_iters[0] else np.inf
        n2 = len(orders) if is_iters[1] else np.inf
        n3 = len(centers) if is_iters[2] else np.inf
        n = min((n1, n2, n3))
        if not is_iters[0]:
            cutoff = [cutoff for _ in range(n)]
        if not is_iters[1]:
            orders = [orders for _ in range(n)]
        if not is_iters[2]:
            centers = [centers for _ in range(n)]
        # The first pair.
        output = butterworth_pair_notch_reject(size, cutoff[0], orders[0], centers[0])
        # The others.
        for i in range(1, n):
            output = np.multiply(
                output,
                butterworth_pair_notch_reject(size, cutoff[i], orders[i], centers[i]),
            )
    else:
        output = butterworth_pair_notch_reject(size, cutoff, orders, centers)
    return output
