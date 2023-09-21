from typing import Union, Optional, Literal

from cython import boundscheck, wraparound
from numba import njit

import numpy as np
from numpy import pi

from commons import outer_bias_parallel
from rfft2_butterworth import *
from rfft2_gaussian import *
# from rfft2_approx_butterworth import *
from img_type import(
    KER_SIZE, ARR_32F2D, IMG_FREQ, ARR_64C2D, ARR_64F2D, ARR_64C2D
)

# rfft = fft for real number.
# rfft2 function will apply rfft in x-direction firstly and then apply fft
# in y-direction.
# For a sequence of length N, the frequencies of rfft of sequence is:
#     [0, 1, ...,     N/2-1,     N/2] if N is even; 
#     [0, 1, ..., (N-1)/2-1, (N-1)/2] if N is odd.
# For a sequence of length N, the frequencies of fft of sequence is:
#     [0, 1, ...,   N/2-1,     -N/2, ..., -1] if N is even; 
#     [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1] if N is odd.

# The filters in this document can directly apply on rfft_img. Don't have to
# apply rfft_shift(rfft_img).


@wraparound(False)
def rfft_shift(rfft_img: IMG_FREQ) -> IMG_FREQ:
    """Shift y-direction such that frequence 0 term is in the middle:
        [-N/2,     ..., -1, 0, 1, ..., N/2-1, ] if N is even; 
        [-(N-1)/2, ..., -1, 0, 1, ..., (N-1)/2] if N is odd.
    
    Parameters
    ----------
    rfft_img : IMG_FREQ
        The frequencies of image.

    Returns
    -------
    output : IMG_FREQ
        The frequencies after shift.
    """
    size_y = rfft_img.shape[0]
    output = np.empty_like(rfft_img)
    half = size_y // 2
    
    if (size_y % 2):
        start = half + 1
        output[-1] = rfft_img[half]
    else:
        start = half
    output[:half] = rfft_img[start:start+half]
    # output[half:] occurs error when size_y is odd.
    output[half:half+half] = rfft_img[:half]
    return output


@wraparound(False)
def rfft_ishift(rfft_img: IMG_FREQ) -> IMG_FREQ:
    """Inverse shift y-direction such that frequence 0 term is index-0:
        [0, 1, ...,   N/2-1,     -N/2, ..., -1] if N is even; 
        [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1] if N is odd.

    Parameters
    ----------
    rfft_img : IMG_FREQ
        The frequencies of image.

    Returns
    -------
    output : IMG_FREQ
        The frequencies after shift.
    """
    size_y = rfft_img.shape[0]
    output = np.empty_like(rfft_img)
    half = size_y//2 + 1 if (size_y % 2) else size_y // 2
    output[:half] = rfft_img[half:]
    output[half:] = rfft_img[:half]
    return output


__SIGNATURE_FREQUENCIES_MATRIX = [
    "int64[:,:,:](UniTuple(int64,2))",
]
@njit(
    __SIGNATURE_FREQUENCIES_MATRIX,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def frequencies_matrix(size: KER_SIZE) -> np.ndarray[np.int64]:
    """To display the relation between indices and frequencies.
        freq[y,x] = (u,v), where u, v are frequcies along y-direction and
    x-direction, respectively.
    

    Parameters
    ----------
    size : Tuple[int, int]
        The size of image in frequency domain.

    Returns
    -------
    freq : ndarray[]
        The frequencies of discrete rfft.
    """
    freq = np.empty((size[0], size[1], 2), np.int64) # (freq_y, freq_x)
    if (size[0] % 2):
        quant = size[0]//2 + 1 # amount for loop
    else:
        quant = size[0] // 2
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        for x in range(size[1]):
            freq[quant,x,0] = -quant
            freq[quant,x,1] = x
    for y in range(1, quant):
        for x in range(size[1]):
            freq[y,x,0] = y
            freq[-y,x,0] = -y
            freq[y,x,1] = x
            freq[-y,x,1] = x
            
    for x in range(size[1]):
        freq[0,x,0] = 0
        freq[0,x,1] = x
    return freq


# Filtering
@wraparound(False)
def filtering(
        rfft_img: IMG_FREQ, mask: Union[ARR_32F2D, ARR_64C2D]
    ) -> IMG_FREQ:
    """Image filtering.

    Parameters
    ----------
    rfft_img : IMG_FREQ
        Image in frequency domain.
    mask : ARR_32F2D | ARR_64C2D
        The filter.

    Returns
    -------
    output : IMG_FREQ
    """
    return np.multiply(rfft_img, mask, dtpe=rfft_img.dtype)



# Laplacian filter in frequency domain
__SIGNATURE_LAPLACIAN = [
    "float32[:,:](UniTuple(int64,2))",
]
@njit(
    __SIGNATURE_LAPLACIAN,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def laplacian_filter(size: KER_SIZE) -> ARR_32F2D:
    """Return the Laplacian filter that be compressed to [0,1].
        `filter(u,v) = -c*Laplacian(u,v) = 4*c*(π*D(u,v))**2`, where
    D is distance, u,v are frequencies of y-component and x-component,
    respectively, and c is the maximum of laplacian in the given size.
    
    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.

    Returns
    -------
    filter : ARR_32F2D
        The Laplacian filter add 1.
    """
    output = np.empty(size, dtype=np.float32)
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = 4 * np.power(pi, 2)
        
    if (size[0] % 2):
        quant = size[0]//2 + 1
        max_ = const * (np.power(quant,2)+square_x[-1]) # Maximum of Laplacian
    else:
        quant = size[0] // 2
        max_ = const * (np.power(quant,2)+square_x[-1]) # Maximum of Laplacian
        # If size[0] is even, then frequency `-quant` exists but frequency
        # `quant` do not.
        output[quant] = (const*(np.power(quant, 2)+square_x)) / max_
    for y in range(1,quant):
        mask_val = (const*(np.power(y,2)+square_x)) / max_
        output[y] = mask_val
        output[-y] = mask_val
    output[0] = const*square_x / max_
    return output


@njit(
    __SIGNATURE_LAPLACIAN,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def laplacian_sharpening_filter(size: KER_SIZE) -> ARR_32F2D:
    """Return the Laplacian sharpening filter.
        `filter(u,v) = (1 - c*Laplacian(u,v)) = 1 + 4c*(π*D(u,v))**2`, where
    D is distance, u,v are frequencies of y-component and x-component,
    respectively, and c is the maximum of laplacian in the given size.
    
    
    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.

    Returns
    -------
    filter : ARR_32F2D
        The Laplacian filter add 1.
    """
    output = np.empty(size, dtype=np.float32)
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x,2)
    const = 4 * np.power(pi, 2)
        
    if (size[0] % 2):
        quant = size[0]//2 + 1
        max_ = const * (np.power(quant,2)+square_x[-1]) # Maximum of Laplacian
    else:
        quant = size[0] // 2
        max_ = const * (np.power(quant,2)+square_x[-1]) # Maximum of Laplacian
        output[quant] = 1 + (const*(np.power(quant,2)+square_x))/max_
    for y in range(1,quant):
        mask_val = 1 + (np.multiply(const, np.power(y,2)+square_x))/max_
        output[y] = mask_val
        output[-y] = mask_val
    output[0] = 1 + (np.multiply(const, square_x))/max_
    return output


@njit(
    __SIGNATURE_LAPLACIAN,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def laplacian_squared(size: KER_SIZE) -> ARR_32F2D:
    """Return the squared Laplacian filter.
        `filter(u,v) = Laplacian(u,v)**2 = 16*(π*D(u,v))**4`, where
    D is distance and u,v are frequencies of y-component and x-component,
    respectively.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.

    Returns
    -------
    filter : ARR_32F2D
        The squared Laplacian filter.
    """
    output = np.empty(size, dtype=np.float32)
    square_x = np.empty(size[1], dtype=np.float32)
    for x in range(size[1]):
        square_x[x] = np.power(x, 2)
    const = 16 * np.power(pi, 4)
        
    if size[0]%2:
        quant = size[0]//2 + 1
    else:
        quant = size[0]//2
        output[quant] = const * np.power(np.power(quant, 2)+square_x, 2)
    for y in range(1,quant):
        mask_val = const * np.power(np.power(y, 2)+square_x, 2)
        output[y] = mask_val
        output[-y] = mask_val
    output[0] = np.multiply(const, np.power(square_x, 2))
    return output



# Motion blur
__SIGNATURE_MOTION_BLUR = [
    "complex64[:,:](UniTuple(int64,2),float32,float32,float32)",
]
@njit(
    __SIGNATURE_MOTION_BLUR,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def linear_motion_blur(
        size: KER_SIZE, T: float, coeff_y: float, coeff_x: float
    ) -> ARR_64C2D:
    """Return the linear motion blur filter, where the speed of motion:
    `y(t) = at/T` and `x(t) = bt/T`, t∈[0,T], T is exposure time, and a, b are
    speed coefficient.
        `filter(u,v) = (T/(π(u*a+v*b)))*sin(π*u*v) * exp(-jπ*(u*a+v*b))`, where
    u,v are frequencies of y-component and x-component, respectively.

    Parameters
    ----------
    size : Tuple[int, int]
        The size of filter.
    T : float
        Exposure time.
    coeff_y : float
        Coefficient of speed in y-direction.
    coeff_x : float
        Coefficient of speed in x-direction.

    Returns
    -------
    filter : ARR_64C2D
        The linear motion blur filter.
    """
    output = np.empty(size, dtype=np.complex64)
    range_x = np.empty(size[1], dtype=np.float32)
    const_y = pi * coeff_y
    const_x = pi * coeff_x
    for x in range(size[1]):
        range_x[x] = x
        
    if (size[0] % 2):
        quant = size[0]//2 + 1
    else:
        quant = size[0]//2
        dist = const_y*quant + const_x*range_x
        output[quant] = np.multiply(
            np.divide(T, dist),
            np.multiply(np.sin(dist), np.exp(-1j * dist)))

    for y in range(1,quant):
        dist = np.multiply(const_y, y) + np.multiply(const_x, range_x)
        mask_val = np.multiply(
            np.divide(T, dist),
            np.multiply(np.sin(dist), np.exp(-1j * dist))
        )
        output[y] = mask_val
        output[-y] = mask_val
    dist = np.multiply(const_x, range_x)
    output[0] = np.multiply(
        np.divide(T, dist),
        np.multiply(np.sin(dist), np.exp(-1j*dist)))
    output[0,0] = np.exp(-1j * dist[0])
    return output


def get_availible_filter_names() -> Tuple[str]:
    return (
        "gaussian", "butterworth"
    )

@boundscheck(False)
@wraparound(False)
def lowpass_filter(
        name: Literal["laplacian", "gaussian", "butterworth"],
        size: KER_SIZE,
        parallel: bool = False,
        **kwargs
    ) -> ARR_32F2D:
    """Return the low-pass filter of specified name.

    Parameters
    ----------
    name : {"laplacian", "gaussian", "butterworth"}
        Filter name.
    size : Tuple[int, int] | None, default=None
        The size of filter. Must given a size if filter_ is str.
    parallel : bool, default=False
        Using parallel computation.
    **kwargs : dict[float]
        sigma_y, sigma_x : float, default=1
            Measure of dispersion along y-direction and x-direction,
            respectively. Works only if name == "gaussian".
        cutoff : float, default=1
            Cutoff frequency. Works only if name == "butterworth".
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".
            
    Returns
    -------
    lowpass : ARR_32FC1
        The low-pass filter.
    """
    if name == "laplacian":
        lowpass = laplacian_filter(size)
    elif name == "gaussian":
        sigma_y = kwargs.get("sigma_y", 1)
        sigma_x = kwargs.get("sigma_x", 1)
        lowpass = gaussian_lowpass(size, sigma_y, sigma_x, parallel)
    elif name == "butterworth":
        cutoff = kwargs.get("cutoff", 1)
        order = kwargs.get("order", 1)
        lowpass = butterworth_lowpass(size, cutoff, order, parallel)
    return lowpass


@boundscheck(False)
@wraparound(False)
def highpass_filter(
        name: Literal["laplacian", "gaussian", "butterworth"],
        size: KER_SIZE,
        parallel: bool = False,
        **kwargs
    ) -> ARR_32F2D:
    """Return the high-pass filter of specified name.

    Parameters
    ----------
    name : {"laplacian", "gaussian", "butterworth"}
        Filter name.
    size : Tuple[int, int]
        The size of filter.
    parallel : bool, default=False
        Using parallel computation.
    **kwargs : dict[float]
        sigma_y, sigma_x : float, default=1
            Measure of dispersion along y-direction and x-direction,
            respectively. Works only if name == "gaussian".
        cutoff : float, default=1
            Cutoff frequency. Works only if name == "butterworth".
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".
            
    Returns
    -------
    highpass : ARR_32FC1
        The high-pass filter.
    """
    if name == "laplacian":
        highpass = laplacian_filter(size)
    elif name == "gaussian":
        sigma_y = kwargs.get("sigma_y", 1)
        sigma_x = kwargs.get("sigma_x", 1)
        highpass = gaussian_highpass(size, sigma_y, sigma_x, parallel)
    elif name == "butterworth":
        cutoff = kwargs.get("cutoff", 1)
        order = kwargs.get("order", 1)
        highpass = butterworth_highpass(size, cutoff, order, parallel)
    return highpass


@boundscheck(False)
@wraparound(False)
def bandpass_filter(
        name: Literal["gaussian", "butterworth"],
        size: KER_SIZE,
        band_center: float,
        band_width: float,
        parallel: bool = False,
        **kwargs
    ) -> ARR_32F2D:
    """Return the band-pass filter of specified name.

    Parameters
    ----------
    name : {"gaussian", "butterworth"}
        Filter name.
    size : Tuple[int, int]
        The size of filter.
    parallel : bool, default=False
        Using parallel computation.
    **kwargs : dict[float]
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".
            
    Returns
    -------
    bandpass : ARR_32FC1
        The band-pass filter.
    """
    if name == "gaussian":
        bandpass = gaussian_bandpass(size, band_center, band_width, parallel)
    elif name == "butterworth":
        order = kwargs.get("order", 1)
        bandpass = butterworth_bandpass(size, band_center, band_width, order,
                                        parallel)
    return bandpass


@boundscheck(False)
@wraparound(False)
def bandreject_filter(
        name: Literal["gaussian", "butterworth"],
        size: KER_SIZE,
        band_center: float,
        band_width: float,
        parallel: bool = False,
        **kwargs
    ) -> ARR_32F2D:
    """Return the band-reject filter of specified name.

    Parameters
    ----------
    name : {"gaussian", "butterworth"}
        Filter name.
    size : Tuple[int, int]
        The size of filter.
    parallel : bool, default=False
        Using parallel computation.
    **kwargs : dict[float]
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".
            
    Returns
    -------
    bandpass : ARR_32FC1
        The band-reject filter.
    """
    if name == "gaussian":
        bandpass = gaussian_bandpass(size, band_center, band_width, parallel)
    elif name == "butterworth":
        order = kwargs.get("order", 1)
        bandpass = butterworth_bandpass(size, band_center, band_width, order,
                                        parallel)
    return bandpass


@boundscheck(False)
@wraparound(False)
def notch_reject_filter(
        name: Literal["gaussian", "butterworth"],
        size: KER_SIZE,
        cutoff: Union[float,Tuple[float]],
        centers: Union[float,Tuple[Tuple[float,float]]],
        **kwargs
    ) -> ARR_32F2D:
    """Return the band-reject filter of specified name.

    Parameters
    ----------
    name : {"gaussian", "butterworth"}
        Filter name.
    size : Tuple[int, int]
        The size of filter.
    cutoff : float | Tuple[float]
        Cutoff frequencies.
    centers : Tuple[float, float] | Tuple[Tuple[float,float]]
        Centers of notches.
    **kwargs : dict[float]
        order : float | Tuple[float], default=1
            Order of Butterworth filter. Works only if name == "butterworth".
            
    Returns
    -------
    bandpass : ARR_32FC1
        The band-reject filter.
    """
    if name == "gaussian":
        bandpass = gaussian_notch_reject(size, cutoff, centers)
    elif name == "butterworth":
        order = kwargs.get("order", 1)
        bandpass = butterworth_notch_reject(size, cutoff, centers, order)
    return bandpass



## Image Sharpening
@boundscheck(False)
@wraparound(False)
def high_frequency_emphasis_filter(
        filter_: Union[ARR_32F2D, ARR_64F2D,
                       Literal["gaussian", "butterworth"]],
        size: Optional[KER_SIZE] = None,
        amount: float = 1.,
        bias: float = 1.,
        **kwargs
    ) -> Union[ARR_32F2D, ARR_64F2D]:
    """Return the high-frequency-emphasis filter (HFEF).
        `HFEF = (bias + amount*HP) = [(bias+amount) - amount*LP]`,
    where HP is a high-pass filter and LP = 1 - HP.

    Parameters
    ----------
    filter_ : ARR_32FC1 | ARR_64FC1 | {"gaussian", "butterworth"}
        Low-pass filter. Given a filter or an availible name.
    size : Tuple[int, int] | None, default=None
        The size of filter. Must given a size if filter_ is str.
    amount : float, default=1
        High-frequency increasing amount.
    bias : float, default=1
        Basic increasing amount.
    **kwargs : dict[float | bool]
        parallel : bool, default=False
            Using parallel computation.
        sigma_y, sigma_x : float, default=1
            Measure of dispersion along y-direction and x-direction,
            respectively. Works only if name == "gaussian".
        cutoff : float, default=1
            Cutoff frequency. Works only if name == "butterworth".
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".

    Returns
    -------
    output : ARR_32FC1 | ARR_64FC1
        The high-frequency-emphasis filter. The dtype is the same as filter_ if
        filter_ is ndarray. Otherwise, dtype is float32.
    """
    # We use low-pass filter because create Gaussian low-pass filter is faster
    # than Gaussian high-pass filter. And the speed of Butterworth low-pass
    # filter is close to Butterworth high-pass filter.
    if isinstance(filter_, np.ndarray):
        lowpass = filter_
    elif filter_ == "gaussian":
        sigma_y = kwargs.get("sigma_y", 1)
        sigma_x = kwargs.get("sigma_x", 1)
        kernel_y = get_freq_gaussian_kernel(size[0], sigma_y)
        kernel_x = get_freq_gaussian_kernel_pos(size[1], sigma_x, c=amount)
        # (bias+amount) - outer_product(kernel_y, kernel_x)
        lowpass =  outer_bias_parallel(kernel_y, kernel_x, bias+amount)
    elif isinstance(filter_, str):
        lowpass = lowpass_filter(filter_, size, **kwargs)
    return np.subtract(bias+amount,
                       np.multiply(amount, lowpass, dtype=lowpass.dtype),
                       dtype=lowpass.dtype)


@boundscheck(False)
@wraparound(False)
def unsharp_mask(
        filter_: Union[ARR_32F2D, ARR_64F2D,
                       Literal["gaussian", "butterworth"]],
        size: Optional[KER_SIZE] = None,
        amount: float = 1.,
        **kwargs
    ) -> Union[ARR_32F2D,ARR_64F2D]:
    """Return the unsharp mask (USM).
        `output = 1 + amount*HP = (1+amount) - amount*LP`,
    where HP is an high-pass filter and LP = 1 - HP.
    Unsharp Masking is a Special Case of HFEF.

    Parameters
    ----------
    filter_ : ARR_32FC1 | ARR_64FC1 | {"gaussian", "butterworth"}
        Low-pass filter. Given a filter or an availible name.
    size : Tuple[int, int] | None, default=None
        The size of filter. Must given a size if filter_ is str.
    amount : float, default=1
        Increasing amount of sharpenning. Usually less than 4.5.
    **kwargs : dict[float | bool]
        parallel : bool, default=False
            Using parallel computation.
        sigma_y, sigma_x : float, default=1
            Measure of dispersion along y-direction and x-direction,
            respectively. Works only if name == "gaussian".
        cutoff : float, default=1
            Cutoff frequency. Works only if name == "butterworth".
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".

    Returns
    -------
    output : ARR_32FC1 | ARR_64FC1
        The high-frequency-emphasis filter. The dtype is the same as filter_ if
        filter_ is ndarray. Otherwise, dtype is float32.
    """
    return high_frequency_emphasis_filter(filter_, size, amount, **kwargs)


@boundscheck(False)
@wraparound(False)
def homomorphic_filter(
        filter_: Union[ARR_32F2D, ARR_64F2D,
                       Literal["gaussian", "butterworth"]],
        size: Optional[KER_SIZE] = None,
        const_low: float = 0.5,
        const_high: float = 2,
        **kwargs
    ) -> Union[ARR_32F2D, ARR_64F2D]:
    """Return the homomorphic filter.
        `output = const_low*LP + const_high*HP`
        `       = const_high - (const_high-const_low)*LP`,
    where HP is an high-pass filter and LP = 1 - HP.
    Usually, const_low <= 1 <= const_high.
    
    The image should take logarithm before applying fft.

    Parameters
    ----------
    filter_ : ARR_32FC1 | ARR_64FC1 | {"gaussian", "butterworth"}
        Low-pass filter. Given a filter or an availible name.
    size : Tuple[int, int] | None, default=None
        The size of filter. Must given a size if filter_ is str.
    const_low : float, default=0.5
        Illuminance increasing (greater than 1) or decreasing (less than 1).
    const_high : float, default=2
        Reflection increasing (greater than 1) or decreasing (less than 1).
    **kwargs : dict[float | bool]
        parallel : bool, default=False
            Using parallel computation.
        sigma_y, sigma_x : float, default=1
            Measure of dispersion along y-direction and x-direction,
            respectively. Works only if name == "gaussian".
        cutoff : float, default=1
            Cutoff frequency. Works only if name == "butterworth".
        order : float, default=1
            Order of Butterworth filter. Works only if name == "butterworth".

    Returns
    -------
    output : ARR_32FC1 | ARR_64FC1
        The homomorphic filter. The dtype is the same as filter_ if
        filter_ is ndarray. Otherwise, dtype is float32.
    """
    return high_frequency_emphasis_filter(
        filter_, size, amount=const_high-const_low, bias=const_low, **kwargs)


## Image Restortion
@wraparound(False)
def inverse_filter(
        mask: Union[ARR_32F2D, ARR_64C2D]
    ) -> Union[ARR_32F2D, ARR_64C2D]:
    """Inverse filter.
        `inverse(mask) = 1 / (mask)`,
    where mask is the degradation function.

    Parameters
    ----------
    mask : ARR_32F2D | ARR_64C2D
        An filter. Degradation function.

    Returns
    -------
    output : ARR_32F2D | ARR_64C2D
    """
    return np.divide(1, mask, dtpe=mask.dtype)


@boundscheck(False)
@wraparound(False)
def modified_inverse_filter(
        mask: Union[ARR_32F2D, ARR_64C2D], cutoff: float, order: float = 1,
        parallel: bool = False
    ) -> Union[ARR_32F2D, ARR_64C2D]:
    """Inverse filter. Restrict the mask by multiplying an Butterworth
    low-pass filter.
        `mod_inv(mask) = 1 / (blpf*mask)`,
    where mask is the degradation function and blpf is an Butterworth filter.

    Parameters
    ----------
    mask : ARR_32F2D | ARR_64C2D
        An filter. Degradation function.
    cutoff : float
        Cutoff frequency. Must be positive.
    order : float, default=1
        Order of Butterworth filter Must. be positive.
    parallel : bool, default=False
        Using parallel computation.

    Returns
    -------
    output : ARR_32F2D | ARR_64C2D
    """
    blpf = butterworth_lowpass(mask.shape, cutoff, order, parallel)
    return np.divide(1, np.multiply(blpf, mask, dtype=mask.dtype),
                     dtype=mask.dtype)


@boundscheck(False)
@wraparound(False)
def winner_filter(
        mask: Union[ARR_32F2D, ARR_64C2D], k: float
    ) -> Union[ARR_32F2D, ARR_64C2D]:
    """Winner filter. Restrict the mask by multiplying an Butterworth
    low-pass filter.
        `winer(mask, k) = conjugate(mask) / (|mask|**2+k)`,
    where mask is the degradation function and k is an estimated parameter.

    Parameters
    ----------
    mask : ARR_32F2D | ARR_64C2D
        An filter. Degradation function.
    k : float
        Estimated signal-to-noise ratio.

    Returns
    -------
    output : ARR_32F2D | ARR_64C2D
    """
    return np.divide(mask, np.add(mask**2, k, dtype=mask.dtype),
                     dtype=mask.dtype)


@boundscheck(False)
@wraparound(False)
def constrained_least_squares_filter(
        mask: Union[ARR_32F2D, ARR_64C2D], gamma: float
    ) -> Union[ARR_32F2D, ARR_64C2D]:
    """Constrained least squares filter (CLSF).
        `clsf(mask, k) = conjugate(mask) / (|mask|**2+gamma*|laplacian|**2)`,
    where mask is the degradation function, gamma is an estimated parameter,
    and laplacian is the Laplacian filter in frequency domain.

    Parameters
    ----------
    mask : ARR_32F2D | ARR_64C2D
        An filter. Degradation function.
    gamma : float
        Restriction parameter.

    Returns
    -------
    output : ARR_32F2D | ARR_64C2D"""
    lap_square = laplacian_squared(mask.shape)
    return np.divide(mask, np.power(mask,2) + np.multiply(gamma, lap_square))


@wraparound(False)
def geometric_mean_filter(
        mask: Union[ARR_32F2D, ARR_64C2D], alpha: float, k: float
    ) -> Union[ARR_32F2D, ARR_64C2D]:
    """Geometric mean filter. The weighted geometric mean of inverse filter and
    winner filter.
        `filter = (1/mask)**alpha * (winner(mask, k))**(1-alpha)`,
    where alpha is the weight, , and winner(mask, k) is the winner filter with
    degradation function `mask` and parameter `k`.

    Parameters
    ----------
    mask : ARR_32F2D | ARR_64C2D
        An filter. Degradation function.
    alpha : float
        The weight.
    k : float
        Estimated signal-to-noise ratio.

    Returns
    -------
    output : ARR_32F2D | ARR_64C2D"""
    inverse = np.power(np.divide(1,mask),alpha)
    winner  = np.power(np.divide(mask, np.power(mask,2) + k),
                       np.subtract(1,alpha))
    return np.multiply(inverse,winner)

