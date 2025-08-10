__all__ = [
    'linear_transformation',
    'linear_transformation_8UC1',
    'piecewise_linear_function',
    'piecewise_linear_transformation',
    'intensity_level_slicing_type1',
    'intensity_level_slicing_type2',
    'intensity_level_slicing_type2_inv',
    'gamma_correction',
    'gamma_correction_8UC1',
    'gamma_correction_32FC1',
    'log_transformation',
    'log_transformation_8UC1',
    'arctan_transformation_8UC1',
    'logistic_correction_8UC1',
    'beta_correction',
    'beta_correction_8UC1',
    'auto_gamma_correction_PB_PZ',
    'histogram_equalization',
    'histogram_matching',
]

from math import atan, exp, ceil

from numba import njit

import numpy as np
from numpy.typing import NDArray, DTypeLike
from cv2 import equalizeHist

from src.utils.helpers import normalize_uint8, transform
import src.utils.stats as stats
from src.utils.specials import betainc
from src.utils.img_type import Arr8U2D, Arr32F2D, Arr8U1D, ARR_1D


def __clip_and_transform_dtype(
    img: NDArray, clip_: bool, dtype: DTypeLike, maximum: float = 255
) -> NDArray:
    """Clips the range of values to [0, maximum] and transforms dtype.

    Parameters
    ----------
    img : NDArray
        A numpy array.
    clip_ : bool
        Clip img or not.
    dtype : DTypeLike
        The dtype of output array.
    maximum : float, default=255
        The maximum of output array. Works only if clip_ == True.

    Returns
    -------
    output : NDArray
        output array.
    """
    if clip_:
        output = np.clip(img, 0, maximum)
    else:
        output = img
    if output.dtype != dtype:
        return output.astype(dtype)
    else:
        return output


# Image enhance 影像增強


def linear_transformation(
    img: NDArray,
    ratio: float = 1,
    bias: float = 0,
    maximum: float | None = None,
    clip_: bool = True,
    dtype: DTypeLike = np.uint8,
) -> NDArray:
    """Intensity transformation by linear transformation:
        T(img) = ratio*img + bias.

    The default dtype is uint8. If clip_=True, then the output will be clip to
    [0,255].

    Parameters
    ----------
    img : ndarray
        A numpy array.
    ratio : float, default=1
        Scaling coefficient.
    bias : float, default=0
        Translation coefficient.
    maximum : float | None, default=None
        The maximum of output. If maximum is None, then put maximum = 255 when
        dtype == np.uint8. Otherwise put maximum = 1. The parameter works only
        if clip_ == True.
    clip : bool, default=True
        Whether clips range of img to [0, maximum].
    dtype : DTypeLike, default=np.uint8
        The dtype of output array.

    Returns
    -------
    output : ndarray
        Transformed img.
    """
    if maximum is None:
        maximum = 255 if img.dtype == np.uint8 else 1
    output = np.multiply(img, ratio, dtype=np.float32)
    np.add(output, bias, out=output, dtype=np.float32)
    return __clip_and_transform_dtype(output, clip_, dtype, maximum)


__SIGNATURE_LINEAR_TRANSFORMATION = [
    'uint8[:,:](uint8[:,:],float32,float32)',
    'uint8[:,:](uint8[:,:],float32,Omitted(0))',
    'uint8[:,:](uint8[:,:],Omitted(1),float32)',
    'uint8[:,:](uint8[:,:],Omitted(1),Omitted(0))',
]


@njit(__SIGNATURE_LINEAR_TRANSFORMATION, nogil=True, cache=True, fastmath=True)
def linear_transformation_8UC1(
    img: Arr8U2D, ratio: float = 1, bias: float = 0
) -> Arr8U2D:
    """Intensity transformation by linear transformation:
        T(img) = ratio*img + bias.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    ratio : float, default=1
        Scaling coefficient.
    bias : float, default=0
        Translation coefficient.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = np.empty(256, dtype=np.uint8)
    # Calculate table
    for i in range(256):
        val = ratio * i + bias
        if val < 0:
            table[i] = 0
        elif val > 255:
            table[i] = 255
        else:
            table[i] = round(val)
    return transform(img, table)


@njit('uint8[:](float32[:],float32[:])', nogil=True, cache=True, fastmath=True)
def piecewise_linear_function(x_points: ARR_1D, y_heights: ARR_1D) -> Arr8U1D:
    """Create a piecewise linear function from breakpoints x_points and
    corresponding values y_heights.
    This function only return the values at {0, 1, 2, ..., 255}.

    Parameters
    ----------
    x_points : ARR_1D
        The breakpoints of function. The value should be float in [0,255].
    y_heights : ARR_1D
        The value of function on breakpoints.

    Returns
    -------
    output : ARR_8U1D
        Piecewise linear function on {0, 1, 2, ..., 255}.
    """
    output = np.empty(256, dtype=np.uint8)
    for i, val in enumerate(x_points[:-1]):
        start, end = ceil(val), ceil(x_points[i + 1])
        slope = (y_heights[i + 1] - y_heights[i]) / (x_points[i + 1] - val)
        for j in range(start, end):
            output[j] = round(slope * (j - val) + y_heights[i])
    if x_points[0] > 0:
        slope = (y_heights[1] - y_heights[0]) / (x_points[1] - x_points[0])
        output[0] = round(slope * (0 - x_points[0]) + y_heights[0])
    if x_points[-1] == 255:
        output[-1] = round(y_heights[-1])
    elif x_points[-1] < 255:
        slope = (y_heights[-1] - y_heights[-2]) / (x_points[-1] - x_points[-2])
        output[-1] = round(slope * (255 - x_points[-1]) + y_heights[-1])
    return output


@njit(
    'uint8[:,:](uint8[:,:],float32[:],float32[:])',
    nogil=True,
    cache=True,
    fastmath=True,
)
def piecewise_linear_transformation(
    img: Arr8U2D, x_points: ARR_1D, y_heights: ARR_1D
) -> Arr8U2D:
    """Image transforms by using piecewise linear function which breakpoints
    and corresponding values are x_points and y_heights, respectively.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    x_points : ARR_1D
        The breakpoints of function. The value should be float in [0,255].
    y_heights : ARR_1D
        The value of function on breakpoints.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = piecewise_linear_function(x_points, y_heights)
    return transform(img, table)


@njit(
    'uint8[:,:](uint8[:,:],UniTuple(uint8,2),uint8,uint8)',
    nogil=True,
    cache=True,
    fastmath=True,
)
def intensity_level_slicing_type1(
    img: Arr8U2D, region: tuple[int, int], fg: int, bg: int
) -> Arr8U2D:
    """Slice specified intensity region. The others region is assigned to be
    background color.
    output[y,x] = fg, region[0] <= img[y,x] <= region[1],
                = bg, otherwise.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    region : tuple[int, int]
        Lower bound and upper bound (both include).
    fg : int
        Foreground color. Must be uint8.
    bg : int
        Background color. Must be uint8.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = np.empty(256, dtype=np.uint8)
    for i in range(256):
        if region[0] <= i <= region[1]:
            table[i] = fg
        else:
            table[i] = bg
    return transform(img, table)


@njit(
    'uint8[:,:](uint8[:,:],UniTuple(uint8,2),uint8)',
    nogil=True,
    cache=True,
    fastmath=True,
)
def intensity_level_slicing_type2(
    img: Arr8U2D, region: tuple[int, int], level: int
) -> Arr8U2D:
    """Slice specified intensity region. The others region is preserved.
    output[y,x] = level,    region[0] <= img[y,x] <= region[1],
                = img[y,x], otherwise.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    region : tuple[int, int]
        Lower bound and upper bound (both include).
    level : int
        Foreground color. Must be uint8.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = np.empty(256, dtype=np.uint8)
    for i in range(256):
        if region[0] <= i <= region[1]:
            table[i] = level
        else:
            table[i] = i
    return transform(img, table)


@njit(
    'uint8[:,:](uint8[:,:],UniTuple(uint8,2),uint8)',
    nogil=True,
    cache=True,
    fastmath=True,
)
def intensity_level_slicing_type2_inv(
    img: Arr8U2D, region: tuple[int, int], level: int
) -> Arr8U2D:
    """Preserve specified intensity region. The others region is sliced.
    output[y,x] = img[y,x], region[0] <= img[y,x] <= region[1],
                = level,    otherwise.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    region : tuple[int, int]
        Lower bound and upper bound (both include).
    level : int
        Background color. Must be uint8.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = np.empty(256, dtype=np.uint8)
    for i in range(256):
        if region[0] <= i <= region[1]:
            table[i] = i
        else:
            table[i] = level
    return transform(img, table)


def gamma_correction(
    img: NDArray,
    gamma: float | NDArray = 1,
    ratio: float | None = None,
    bias: float = 0,
    maximum: float | None = None,
    clip_: bool | None = None,
    dtype=np.uint8,
) -> NDArray:
    """Intensity transformation by gamma correction:
        T(img) = ratio * img**gamma.

    Parameters
    ----------
    img : ndarray
        A numpy array.
    gamma : float, default=1
        Power coefficient. If gamma == 1, return img.
    ratio : float | None, default=None
        Scaling coefficient. If ratio is None, then choose ratio such that
        ratio * maximum**gamma = maximum.
    bias : float, default=0
        Translation coefficient.
    maximum : float | None, default=None
        The maximum of output. If maximum is None, then put maximum = 255 when
        dtype == np.uint8. Otherwise put maximum = 1. The parameter works only
        if clip_ == True.
    clip_ : bool | None, default=None
        Whether clips range of img to [0, maximum].
    dtype : DTypeLike, default=np.uint8
        The dtype of output array.

    Returns
    -------
    output : ndarray
        Transformed img.
    """
    if gamma == 1:
        return img
    if maximum is None:
        maximum = 255 if img.dtype == np.uint8 else 1
    if ratio is None and clip_ is None:
        clip_ = False
        # Choose ratio such that T(maximum) = maximum
        ratio = np.power(maximum, 1 - gamma, dtype=np.float32)
    elif clip_ is None:
        clip_ = True

    output = np.power(img, gamma, dtype=np.float32)
    if ratio != 1:
        np.multiply(output, ratio, out=output, dtype=np.float32)
    if bias != 0:
        np.add(output, bias, out=output, dtype=np.float32)
    return __clip_and_transform_dtype(output, clip_, dtype, maximum)


__SIGNATURE_GAMMA_CORRECTION = [
    'uint8[:,:](uint8[:,:],float32,Optional(float32),float32)',
    'uint8[:,:](uint8[:,:],float32,Optional(float32),Omitted(0))',
    'uint8[:,:](uint8[:,:],Omitted(1.),Optional(float32),float32)',
    'uint8[:,:](uint8[:,:],float32,Omitted(None),float32)',
    'uint8[:,:](uint8[:,:],Omitted(1.),Omitted(None),float32)',
    'uint8[:,:](uint8[:,:],Omitted(1.),Optional(float32),Omitted(0))',
    'uint8[:,:](uint8[:,:],float32,Omitted(None),Omitted(0))',
    'uint8[:,:](uint8[:,:],Omitted(1.),Omitted(None),Omitted(0))',
]


@njit(__SIGNATURE_GAMMA_CORRECTION, nogil=True, cache=True, fastmath=True)
def gamma_correction_8UC1(
    img: Arr8U2D, gamma: float = 1.0, ratio: float = None, bias: float = 0
) -> Arr8U2D:
    """Intensity transformation by gamma correction:
        T(img) = ratio * img**gamma + bias.
    Input and output images are 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        A numpy array.
    gamma : float, default=1
        Power coefficient. If gamma == 1, then return img.
    ratio : float | None, default=None
        Scaling coefficient. If ratio is None, then choose ratio such that
        ratio * 255**gamma = 255.
    bias : float, default=0
        Translation coefficient.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    if gamma == 1:
        return img
    table = np.empty(256, dtype=np.uint8)
    if ratio is None:
        # constant that makes 255^gamma * ratio = 255.
        ratio = np.float32(np.power(255, 1 - gamma))
    # Calculate table
    for i in range(256):
        val = np.power(i, gamma) * ratio + bias
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        table[i] = round(val)
    return transform(img, table)


__SIGNATURE_GAMMA_CORRECTION2 = [
    'float32[:,:](uint8[:,:],float32,Optional(float32),float32)',
    'float32[:,:](uint8[:,:],float32,Optional(float32),Omitted(0))',
    'float32[:,:](uint8[:,:],Omitted(1.),Optional(float32),float32)',
    'float32[:,:](uint8[:,:],float32,Omitted(None),float32)',
    'float32[:,:](uint8[:,:],Omitted(1.),Omitted(None),float32)',
    'float32[:,:](uint8[:,:],Omitted(1.),Optional(float32),Omitted(0))',
    'float32[:,:](uint8[:,:],float32,Omitted(None),Omitted(0))',
    'float32[:,:](uint8[:,:],Omitted(1.),Omitted(None),Omitted(0))',
]


@njit(__SIGNATURE_GAMMA_CORRECTION2, nogil=True, cache=True, fastmath=True)
def gamma_correction_32FC1(
    img: Arr8U2D, gamma: float = 1.0, ratio: float = None, bias: float = 0
) -> Arr32F2D:
    """Intensity transformation by gamma correction:
        T(img) = ratio * img**gamma.
    Input image is 8UC1 and output image is 32FC1. The range will be compressed
    from [0,255] to [0,1].

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    gamma : float, default=1
        Power coefficient. If gamma == 1, then just compress the range.
    ratio : float | None, default=None
        Scaling coefficient. If ratio is None, then choose ratio such that
        ratio * 255**gamma = 1.
    bias : float, default=0
        Translation coefficient.

    Returns
    -------
    output : ARR_32F2D
        Transformed img.
    """
    const = (1 / 255) ** gamma
    if gamma == 1:
        table = np.empty(256, dtype=np.float32)
        for i in range(256):
            table[i] = i * const
        return transform(img, table)
    table = np.empty(256, dtype=np.float32)
    if ratio is None:
        # constant that makes ratio * 255^gamma = 1.
        ratio = 255 ** (-gamma)
    # Calculate table
    const *= ratio
    for i in range(256):
        val = np.power(i, gamma) * const + bias
        if val < 0:
            val = 0
        elif val > 1:
            val = 1
        table[i] = val
    return transform(img, table)


def log_transformation(
    img: NDArray,
    ratio: bool | None = None,
    bias: float = 0,
    clip_: bool | None = None,
    maximum: float | None = None,
    dtype: DTypeLike = np.uint8,
) -> NDArray:
    """Intensity transformation by log transformation:
        T(img) = ratio*ln(1+img) + bias.
    If ratio < 0, then it will be choosed by maximum such that
    T(maximum)=maximum.

    Parameters
    ----------
    img : ndarray
        A numpy array.
    gamma : float, default=1
        Power coefficient. If gamma == 1, return img.
    ratio : float | None, default=None
        Scaling coefficient. If ratio is None, then choose ratio such that
        ratio * ln(1+maximum) = maximum.
    bias : float, default=0
        Translation coefficient.
    maximum : float | None, default=None
        The maximum of output. If maximum is None, then put maximum = 255 when
        dtype == np.uint8. Otherwise put maximum = 1. The parameter works only
        if clip_ == True.
    clip_ : bool | None, default=None
        Whether clips range of img to [0, maximum].
    dtype : DTypeLike, default=np.uint8
        The dtype of output array.

    Returns
    -------
    output : ndarray
        Transformed img.
    """
    if maximum is None:
        maximum = 255 if img.dtype == np.uint8 else 1
    if ratio is None and clip_ is None:
        # Choose ratio such that T(maximum) = maximum
        clip_ = False
        ratio = maximum / np.log1p(maximum)
    elif clip_ is None:
        clip_ = True

    output = np.log1p(img, dtype=np.float32)
    if ratio != 1:
        np.multiply(output, ratio, out=output, dtype=np.float32)
    if bias != 0:
        np.add(output, bias, out=output, dtype=np.float32)
    return __clip_and_transform_dtype(output, clip_, dtype, maximum)


__SIGNATURE_LOG_TRANSFORMATION = [
    'uint8[:,:](uint8[:,:],Optional(float32),float32)',
    'uint8[:,:](uint8[:,:],Omitted(None),float32)',
    'uint8[:,:](uint8[:,:],Optional(float32),Omitted(0))',
    'uint8[:,:](uint8[:,:],Omitted(None),Omitted(0))',
]


@njit(__SIGNATURE_LOG_TRANSFORMATION, nogil=True, cache=True, fastmath=True)
def log_transformation_8UC1(
    img: Arr8U2D, ratio: float = None, bias: float = 0
) -> Arr8U2D:
    """
    Image intensity transformation by gamma correction:
        T(img) = ratio * ln(1+img).
    Input and output images are 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    ratio : float | None, default=None
        Scaling coefficient. If ratio is None, then choose ratio such that
        T(255) = 255.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    table = np.empty(256, dtype=np.uint8)
    if ratio is None:
        # Choose ratio such that T(maximum) = maximum
        ratio = 255 / np.log1p(255)
    # Calculate table
    for i in range(256):
        val = np.multiply(np.log1p(i), ratio) + bias
        if val < 0:
            val = 0
        elif val > 255:
            val = 255
        table[i] = round(val)
    return transform(img, table)


# Image Correction with S-shape functions.
__SIGNATURE_ARCTAN_TRANSFORMATION = [
    'uint8[:,:](uint8[:,:],float32,float32)',
    'uint8[:,:](uint8[:,:],Omitted(0.5),float32)',
    'uint8[:,:](uint8[:,:],float32,Omitted(-1))',
    'uint8[:,:](uint8[:,:],Omitted(0.5),Omitted(-1))',
]


@njit(__SIGNATURE_ARCTAN_TRANSFORMATION, nogil=True, cache=True, fastmath=True)
def arctan_transformation_8UC1(
    img: Arr8U2D, gamma: float = 0.5, center: float = -1
) -> Arr8U2D:
    """Intensity transformation by an S-shaped function:
        T(img) = c*arctan(g*img-b) + d,
    where g = gamma/100, b = center*g, c = 255/(arctan(255*g-b)+arctan(b)),
          d = c*arctan(b).
    Input and output images are 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    gamma : float, default=0.5
        Winding rate of S. Must be positive.
    center : float, default=-1
        Approximate fixed point of function. If center < 0 or center > 255,
        put ceenter = 127.5.


    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    if gamma == 0:
        return img
    if center < 0 or center > 255:
        center = 127.5

    table = np.empty(256, dtype=np.uint8)
    # constants
    gamma /= 100
    b = center * gamma
    c = 255 / (atan(255 * gamma - b) + atan(b))
    d = c * atan(b)
    # Calculate table
    for i in range(256):
        val = gamma * i - b
        table[i] = round(c * atan(val) + d)
    return transform(img, table)


__SIGNATURE_SIGMOID_CORRECTION = [
    'uint8[:,:](uint8[:,:],float32,float32)',
    'uint8[:,:](uint8[:,:],Omitted(7),float32)',
    'uint8[:,:](uint8[:,:],float32,Omitted(-1))',
    'uint8[:,:](uint8[:,:],Omitted(7),Omitted(-1))',
]


@njit(__SIGNATURE_SIGMOID_CORRECTION, nogil=True, cache=True, fastmath=True)
def logistic_correction_8UC1(
    img: Arr8U2D, sigma: float = 7, center: float = -1
) -> Arr8U2D:
    """
    Image intensity transformation by an S-shaped function:
        T(img) = 255 * (P(img) - P(0)) / (P(255)-P(0))
    where P(x) = 1 / (1+e**(-(x-center)/sigma**2)) is a logistic function.
    Input and output images are 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    sigma : float, default=0.5
        Winding rate of S. Must be positive.
    center : float, default=-1
        Fixed point of function. If center < 0 or center > 255,
        put ceenter = 127.5.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    if sigma == 0:
        return img
    if center < 0 or center > 255:
        center = 127.5

    table = np.empty(256, dtype=np.uint8)
    # Constants
    sigma2 = sigma**2
    bias = 1 / (1 + exp(center / sigma2))
    c0 = 1 / (1 + exp(-(255 - center) / sigma2)) - bias
    c = 255 / (c0)  # 255 / (P(255)-P(0)) Scaling coefficient
    bias *= c  # 255 * P(0)/(P(255)-P(0)) Translation coefficient
    # Calculate table
    for i in range(256):
        val = (center - i) / sigma2
        table[i] = round(c / (1 + exp(val)) - bias)
    return transform(img, table)


def beta_correction(img: NDArray, a: float = -1, b: float = -1) -> NDArray:
    """Intensity transformation by imcomplete beta function:
        T(img) = incbeta(img; a, b), img in [0,1].
    output image is 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    a : float
        Shape of incomplete beta function. Must be positive.
    b : float
        Shape of incomplete beta function. Must be positive.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    if a <= 0 or b <= 0:
        return img
    if img.dtype == np.uint8:
        output = betainc(a, b, normalize_uint8(img))
    else:
        output = betainc(a, b, img)
    output = np.multiply(output, 255, dtype=np.float32)
    return __clip_and_transform_dtype(output, True, np.uint8)


def beta_correction_8UC1(img: Arr8U2D, a: float = 2, b: float = 2) -> Arr8U2D:
    """
    Image intensity transformation by implete beta function:
        T(img) = incbeta(img, a, b),
    where incbeta is the incomplete beta function.
    Input and output images are 8UC1.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    a : float
        Shape of incomplete beta function. Must be positive.
    b : float
        Shape of incomplete beta function. Must be positive.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    if a <= 0 or b <= 0:
        return img
    table = np.linspace(0, 1, 256)
    table = np.around(betainc(a, b, table) * 255).astype(np.uint8)
    return transform(img, table)


# Automatic image enhancement
def auto_gamma_correction_PB_PZ(img: NDArray):
    """
    Apply the gamma correction to an image img with
        γ = log(0.5) / log(mean(img)), where img takes values in [0,1].

    This method is proposed by `Pedram Babakhani1 and Parham Zarei in
    Automatic gamma correction based on average of brightness. (2015).`

    Parameters
    ----------
    img : ndarray
        A numpy array with ndim is 2 or 3.

    Returns
    -------
    output : ndarray
        Transformed img. The dtype is the same as img.
    """
    if img.dtype == np.uint8:
        gamma = np.divide(
            np.log10(0.5), np.log10(np.divide(np.mean(img, axis=(0, 1)), 255))
        )
        if img.ndim == 2:
            return gamma_correction_8UC1(img, gamma)
        else:
            output = np.empty_like(img)
            for i in range(img.shape[2]):
                output[:, :, i] = gamma_correction_8UC1(img[:, :, i], gamma[i])
            return output
    else:
        gamma = np.divide(np.log10(0.5), np.log10(np.mean(img, axis=(0, 1))))
        if img.ndim == 2:
            return gamma_correction(img, gamma)
        else:
            output = np.empty_like(img)
            for i in range(img.shape[2]):
                output[:, :, i] = gamma_correction(
                    img[:, :, i], gamma[i], dtype=img.dtype
                )
            return output


# Methods with Histogram
__SIGNATURE_HISTOGRAM_EQUALIZATION = [
    'uint8[:,:](uint8[:,:])',
]


@njit(
    __SIGNATURE_HISTOGRAM_EQUALIZATION,
    nogil=True,
    cache=True,
    fastmath=True,
)
def histogram_equalization(img: Arr8U2D) -> Arr8U2D:
    """Return the histogram equalization of an 8UC1 image.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    # Grayscale histogram equalization.
    cdf = stats.cumulative_percentage_8UC1(img)
    # Intensity table, table[input] = output.
    table = np.empty(256, dtype=np.uint8)
    bias = cdf[0]
    c = 1 / (1 - bias)  # Scaling and translation to [0,255]
    # Calculate table
    for i in range(255):
        cdf[i] = c * (cdf[i] - bias)
        table[i] = round(np.multiply(cdf[i], 255))
    table[255] = 255
    return transform(img, table)


# -Histogram matching
# -Three methods, histogram_matching, histogram_matching2, and
#  histogram_matching3, are very similar. The only difference is how these
#  approximate G_inverse.
__SIGNATURE_HISTOGRAM_MATCHING = [
    'uint8[:,:](uint8[:,:],float32[:])',
    'uint8[:,:](uint8[:,:],float64[:])',
]


@njit(__SIGNATURE_HISTOGRAM_MATCHING, nogil=True, cache=True, fastmath=True)
def __histogram_matching(he_img, pdf):
    # see Gonzalez, Woods, 數位影像處理(Digital Image Processing), 4e, page 108
    G = np.empty(256, dtype=np.uint8)  # Specified intensity transformation.
    G_inverse = np.empty(256, dtype=np.uint8)
    # Min index of distinct values in G
    # eg, G = [0, 1, 1, 5, 5, 5, 5, 6] => distinct_index = [0, 1, 3, 7]
    distinct_index = np.empty(256, dtype=np.uint8)

    s = np.sum(pdf)
    if s > 1:
        pdf[:] /= s
    # Calculate G.
    cdf = 0  # cdf
    for j in range(255):
        cdf += pdf[j]
        G[j] = round(np.multiply(cdf, 255))
    G[-1] = 255
    # Calculate distinct_index.
    index = 0
    val = -1
    for i in range(256):
        if G[i] != val:
            distinct_index[index] = i
            val = G[i]
            index += 1
    # Calculate G_inverse,
    # G_inverse[i] = min{j | d(G[j],i) <= d(G[k],i) for k=0, 1, ..., 255}.
    G_inverse[-1] = 255
    for i in range(index - 1):
        val1 = G[distinct_index[i]]
        val2 = G[distinct_index[i + 1]]
        diff = val2 - val1
        half = ceil(diff / 2)
        G_inverse[val1 : val1 + half] = distinct_index[i]
        G_inverse[val1 + half : val2] = distinct_index[i + 1]
    return transform(he_img, G_inverse)


def histogram_matching(img: Arr8U2D, pdf: np.ndarray):
    """Return the histogram matching of an 8UC1 image with specified pdf.

    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    pdf : ndarray
        The pdf that will be img mathched. A numpy 1-D float32/float64 array.

    Returns
    -------
    output : ARR_8U2D
        Transformed img.
    """
    return __histogram_matching(equalizeHist(img), pdf)


# @njit(
#     __SIGNATURE_HISTOGRAM_MATCHING,
#     nogil=True, cache=True, fastmath=True)
#
# def __histogram_matching2(img, pdf):
#     # see Gonzalez, Woods, 數位影像處理(Digital Image Processing), 4e, page 108
#     G = np.empty(256, dtype=np.uint8) # Specified intensity transformation.
#     G_inverse = np.zeros(256, dtype=np.uint8)

#     s = round(np.sum(pdf), 1)
#     if  s> 1:
#         pdf[:] /= np.sum(pdf)

#     # Calculate G.
#     cdf = 0 # cdf
#     for i in range(255):
#         cdf += pdf[i]
#         G[i] = round(np.multiply(cdf, 255))
#     G[-1] = 255
#     # Calculate Ginv
#     # -Since G[255] = G[-1] = 255 > 254 ≥ i, the range start from 254.
#     G_inverse[-1] = 255
#     last_j = 0
#     for i in range(255):
#         # G_inverse[i] = min{ j | G[j] >= i }
#         for j in range(last_j, 255):
#             if (i <= G[j]):
#                 break
#         G_inverse[i] = j
#         last_j = j
#     return commons.transform(img, G_inverse)
#
#
# def histogram_matching2(img, pdf):
#     return __histogram_matching2(equalizeHist(img), pdf)


# @njit(
#     __SIGNATURE_HISTOGRAM_MATCHING,
#     nogil=True, cache=True, fastmath=True)
#
# def __histogram_matching3(img, pdf):
#     # see Gonzalez, Woods, 數位影像處理(Digital Image Processing), 4e, page 108
#     G = np.empty(256, dtype=np.uint8) # Specified intensity transformation.
#     G_inverse = np.empty(256, dtype=np.uint8)

#     s = round(np.sum(pdf), 1)
#     if s > 1:
#         pdf[:] /= np.sum(pdf)

#     # Calculate G.
#     cdf = 0 # cdf
#     for j in range(255):
#         cdf += pdf[j]
#         G[j] = round(np.multiply(cdf, 255))
#     G[-1] = 255
#     # Calculate Ginv,
#     G_inverse[:G[0]] = 0
#     j = 0
#     val = G[0]
#     # G_inverse[i] = min{ j | G[j] > i }-1, min∅ = 0, (0-1) = 255 (mod 256)
#     for k in range(256):
#         if G[k] != val:
#             G_inverse[val:G[k]] = j
#             j = k
#             val = G[k]
#             if val == 255:
#                 break
#     G_inverse[val:] = j
#     return commons.transform(img, G_inverse)
#
#
# def histogram_matching3(img, pdf):
#     return __histogram_matching3(equalizeHist(img), pdf)

# # Show difference of three methods
# def display_histogram_matchings(img, pdf):
#     import matplotlib.pyplot  as plt
#     from stats import histogram_percentage_8UC1
#     n_plot = (3, 3) # row, col
#     n = np.arange(256)
#     color = "viridis" if img.ndim == 3 else "gray"

#     plt.subplot(n_plot[0], n_plot[1], 1)
#     plt.imshow(img, cmap=color)
#     plt.axis("off")
#     plt.subplot(n_plot[0], n_plot[1], 2)
#     hist = histogram_percentage_8UC1(img)
#     plt.bar(n, hist)

#     hm = [
#         histogram_matching(img, pdf), histogram_matching2(img, pdf),
#         histogram_matching3(img, pdf)
#     ]
#     hm_hists = [
#         histogram_percentage_8UC1(val) for val in hm
#     ]
#     for i in range(3):
#         plt.subplot(n_plot[0], n_plot[1], 4+i)
#         plt.imshow(hm[0], cmap=color)
#         plt.axis("off")
#         plt.title(f"Histogram matching {i+1}")
#         plt.subplot(n_plot[0], n_plot[1], 7+i)
#         plt.bar(n, pdf, alpha=0.5, color="r")
#         plt.bar(n, hm_hists[i], color="g")
#     plt.show()
