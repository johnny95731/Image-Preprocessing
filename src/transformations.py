__all__ = [
    'DCT1D',
    'IDCT1D',
    'DCT2D',
    'haar',
    'haar_wavelet',
    'haar_same',
    'haar_wavelet_same',
    'steganography_decomposition',
    'steganography_reconstruction',
    'randon_transform',
    'inv_randon_transform_standard',
    'window_kernel',
    'window_kernel_half',
    'filtered_inv_randon_transform',
]

from math import ceil, floor, sin, cos, tan


from numba import njit

import numpy as np
from numpy import pi

from numpy.fft import irfft2, rfft2

from src.utils.img_type import IMG_GRAY, Arr8U2D, Arr32F2D, Arr64F2D


# 修改自
# Copyright (c) 2020 Project Nayuki. (MIT License)
# https://www.nayuki.io/page/fast-discrete-cosine-transform-algorithms


# DCT type II, unscaled. Algorithm by Byeong Gi Lee, 1984.
# See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.118.3056&rep=rep1&type=pdf#page=34


def DCT1D(vector):  # 1-D Discrete Cosine Transform
    n = len(vector)
    if n == 1:
        return list(vector)
    elif n == 0 or n % 2 != 0:
        raise ValueError()
    else:
        half = n // 2
        alpha = [(vector[i] + vector[-(i + 1)]) for i in range(half)]
        beta = [
            (vector[i] - vector[-(i + 1)]) / (cos((i + 0.5) * pi / n) * 2.0)
            for i in range(half)
        ]
        alpha = DCT1D(alpha)
        beta = DCT1D(beta)
        result = []
        for i in range(half - 1):
            result.append(alpha[i])
            result.append(beta[i] + beta[i + 1])
        result.append(alpha[-1])
        result.append(beta[-1])
        return result


# DCT type III, unscaled. Algorithm by Byeong Gi Lee, 1984.
# See: https://www.nayuki.io/res/fast-discrete-cosine-transform-algorithms/lee-new-algo-discrete-cosine-transform.pdf


def IDCT1D(vector, root=True):  # 1-D Inverse Discrete Cosine Transform
    if root:
        vector = list(vector)
        vector[0] /= 2
    n = len(vector)
    if n == 1:
        return vector
    elif n == 0 or n % 2 != 0:
        raise ValueError()
    else:
        half = n // 2
        alpha = [vector[0]]
        beta = [vector[1]]
        for i in range(2, n, 2):
            alpha.append(vector[i])
            beta.append(vector[i - 1] + vector[i + 1])
        IDCT1D(alpha, False)
        IDCT1D(beta, False)
        for i in range(half):
            x = alpha[i]
            y = beta[i] / (cos((i + 0.5) * pi / n) * 2)
            vector[i] = x + y
            vector[-(i + 1)] = x - y
        return vector


@njit('float32[:,:](uint8[:,:])')
def DCT2D(img):  # 2D DCT
    size = img.shape
    step1 = np.empty_like(img, dtype=np.float32)
    output = np.empty_like(img, dtype=np.float32)

    const0 = np.sqrt(np.divide(1, size[1]))
    const1 = np.sqrt(np.divide(2, size[1]))
    # x-Direction
    for y in range(size[0]):
        step1[y] = np.multiply(img[y, 0], const0)
        for x in range(size[1]):
            c = np.divide(np.multiply(2 * x + 1, pi), np.multiply(2, size[1]))
            val = 0
            for u in range(1, size[1]):
                val += np.multiply(np.multiply(img[y, u], const1), cos(c * u))
            step1[y, x] += val
    # y-Direction
    const0 = np.sqrt(np.divide(1, size[0]))
    const1 = np.sqrt(np.divide(2, size[0]))
    for y in range(size[0]):
        output[y] = np.multiply(step1[y], const0)
        for x in range(size[1]):
            c = np.divide(np.multiply(2 * x + 1, pi), np.multiply(2, size[1]))
            val = 0
            for v in range(1, size[0]):
                val += np.multiply(np.multiply(step1[v, x], const1), cos(c * u))
            output[y, x] += val
    return output


# Discrete Wavelet Transformation
__SIGNATURE_HAAR = [
    'UniTuple(float32[:,:],4)(uint8[:,:])',
    'UniTuple(float32[:,:],4)(float32[:,:])',
]


@njit(__SIGNATURE_HAAR, nogil=True, cache=True, fastmath=True)
def haar(img):
    """Return the Haar wavelet transform of an image.

    Parameters
    ----------
    img : IMG_32FC1
        An 1-channel image with dtype uint8 or float32.

    Returns
    -------
    output : list[IMG_32FC1]
        4 images that applied Haar decomposition.
    """
    size = img.shape
    half_y, half_x = size[0] // 2, size[1] // 2
    low = np.empty((size[0], half_x), dtype=np.float32)
    high = np.empty((size[0], half_x), dtype=np.float32)
    LL = np.empty((half_y, half_x), dtype=np.float32)
    LH = np.empty((half_y, half_x), dtype=np.float32)
    HL = np.empty((half_y, half_x), dtype=np.float32)
    HH = np.empty((half_y, half_x), dtype=np.float32)
    # x-Direction
    for y in range(size[0]):
        for x in range(half_x):
            low[y, x] = np.multiply(0.5, img[y, 2 * x] + img[y, 2 * x + 1])
            high[y, x] = low[y, x] - img[y, 2 * x + 1]
    # y-Direction
    for y in range(half_y):
        for x in range(half_x):
            LL[y, x] = np.multiply(0.5, low[2 * y, x] + low[2 * y + 1, x])  # LL
            LH[y, x] = LL[y, x] - low[2 * y + 1, x]  # LH
            HL[y, x] = np.multiply(0.5, high[2 * y, x] + high[2 * y + 1, x])  # HL
            HH[y, x] = HL[y, x] - high[2 * y + 1, x]  # HH
    return LL, LH, HL, HH


def haar_wavelet(img: IMG_GRAY, level: int = 1):
    """Return the Haar wavelet transform of an image.

    Parameters
    ----------
    img : IMG_8UC1 | IMG_32FC1
        An 1-channel image with dtype uint8 or float32.
    level: positive int
        Number of wavelet decomposition applied.

    Returns
    -------
    output : list[IMG_32FC1]
        Imagess that applied Haar decomposition.
    """
    output = [*haar(img)]
    for i in range(level - 1):
        output.extend(haar(output[4 * i]))
    return output


@njit(
    [
        'UniTuple(float32[:,:],4)(uint8[:,:])',
        'UniTuple(float32[:,:],4)(float32[:,:])',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def haar_same(img: Arr32F2D):
    """Return the Haar wavelet transform of an image. The output is same size
    as input.

    Parameters
    ----------
    img : IMG_8UC1 | IMG_32FC1
        An 1-channel image with dtype uint8 or float32.

    Returns
    -------
    output : list[IMG_32FC1]
        4 images that applied Haar decomposition.
    """
    size = img.shape
    low = np.empty(size, dtype=np.float32)
    high = np.empty(size, dtype=np.float32)
    LL = np.empty(size, dtype=np.float32)
    LH = np.empty(size, dtype=np.float32)
    HL = np.empty(size, dtype=np.float32)
    HH = np.empty(size, dtype=np.float32)
    # x-Direction
    for y in range(size[0]):
        for x in range(size[1] - 1):
            low[y, x] = np.multiply(0.5, img[y, x] + img[y, x + 1])
            high[y, x] = low[y, x] - img[y, x + 1]
        low[y, -1] = low[y, -2]
        high[y, -1] = -high[y, -2]
    # y-Direction
    for y in range(size[0] - 1):
        for x in range(size[1]):
            LL[y, x] = np.multiply(0.5, low[y, x] + low[y + 1, x])  # LL
            LH[y, x] = LL[y, x] - low[y + 1, x]  # LH
            HL[y, x] = np.multiply(0.5, high[y, x] + high[y + 1, x])  # HL
            HH[y, x] = HL[y, x] - high[y + 1, x]  # HH
    for x in range(size[1]):
        LL[-1, x] = LL[-2, x]
        LH[-1, x] = -LH[-2, x]
        HL[-1, x] = HL[-2, x]
        HH[-1, x] = -HH[-2, x]
    return LL, LH, HL, HH


def haar_wavelet_same(img: IMG_GRAY, level: int = 1) -> list[Arr32F2D]:
    """Return the Haar wavelet transform of an image.

    Parameters
    ----------
    img : IMG_8UC1 | IMG_32FC1
        An 1-channel image with dtype uint8 or float32.
    level: positive int
        Number of wavelet decomposition applied.

    Returns
    -------
    output : list[IMG_32FC1]
        Imagess that applied Haar decomposition.
    """
    output = [*haar_same(img)]
    for i in range(level - 1):
        output.extend(haar_same(output[4 * i]))
    return output


# 隱像術 Steganography
def steganography_decomposition(
    img: Arr8U2D, return_bool: bool = False
) -> list[Arr8U2D]:
    """Basic steganography by decomposition an 8-bit image into 8 binary images.
    Each image contains the n-th bit of input image.

    Parameters
    ----------
    img : IMG_8UC1
        An 1-channel image with dtype uint8 or float32.
    return_bool : bool
        The return dtype is bool or not.

    Returns
    -------
    output : list[IMG_8UC1]
        8 images, the n-th image contains the n-th bit of input image.
    """
    if return_bool:
        return np.array([np.bitwise_and(img, 2**n) for n in range(8)], dtype=np.bool)
    else:
        return np.array([np.bitwise_and(img, 2**n) for n in range(8)])


def steganography_reconstruction(
    imgs: list[Arr8U2D], return_bool: bool = False
) -> Arr8U2D:
    """Reconstruction of steganography decomposition.

    Parameters
    ----------
    imgs : list[IMG_8UC1]
        8 binary images.
    return_bool : bool
        The return dtype is bool or not.

    Returns
    -------
    output : IMG_8UC1
        The original image before steganography decomposition.
    """
    if return_bool:
        return np.sum(
            [np.multiply(img, 2**n, dtype=np.uint8) for n, img in enumerate(imgs)],
            axis=0,
        )
    else:
        return np.sum(imgs, axis=0)


# 投影重建影像，用於電腦斷層掃描(CT)等
# Gonzalez, Woods, 數位影像處理,4e, Sec. 5.9
# Randon Transformation
@njit(
    [
        'float64[:,:](uint8[:,:],float32)',
        'float64[:,:](uint8[:,:],Omitted(.1))',
        'float64[:,:](float32[:,:],float32)',
        'float64[:,:](float32[:,:],Omitted(.1))',
        'float64[:,:](float64[:,:],float32)',
        'float64[:,:](float64[:,:],Omitted(.1))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def randon_transform(img: IMG_GRAY, deg: float = 0.1) -> Arr64F2D:
    """Simulate tomography by Randon transform.

    Parameters
    ----------
    img : IMG_8UC1
        An 1-channel image with dtype uint8 or float32.
    deg : float
        The increment of deg that ray rotate.

    Returns
    -------
    output : IMG_64FC1
        An image after applying Randon transform.
    """
    size = img.shape

    # Consts
    quant_deg = ceil(np.divide(180, deg))
    quant_rho = ceil(np.sqrt(np.power(size[0], 2) + np.power(size[1], 2)) / 2)
    const_arc = np.multiply(np.divide(pi, 180), deg)  # Radian per deg increment
    half_y = size[0] // 2
    half_x = size[1] // 2
    half_deg = floor(90 / deg)

    output = np.empty((quant_deg, 2 * quant_rho), dtype=np.float64)
    maxi = 0

    for i in range(1, half_deg):  # from deg to 90-deg
        arc = const_arc * i  # Radinan be
        cosArc = cos(arc)
        tanArc = tan(arc)
        for ind, j in enumerate(range(-quant_rho, quant_rho)):  # rho
            val = 0
            for y in range(size[0]):
                high = ceil(np.divide(j + 1, cosArc) + (y - half_y) * tanArc + half_x)
                low = ceil(np.divide(j - 1, cosArc) + (y - half_y) * tanArc + half_x)
                if low < 0:
                    low = 0
                if high >= size[1]:
                    high = size[1]
                for x in range(low, high):
                    val += img[y, x]
            val *= 0.5
            if val > maxi:
                maxi = val
            output[i, ind] = val

    for i in range(half_deg + 1, quant_deg):  # from 90+deg to 180-deg
        arc = const_arc * i
        cosArc = cos(arc)
        tanArc = tan(arc)
        for ind, j in enumerate(range(-quant_rho, quant_rho)):  # rho
            val = 0
            for y in range(size[0]):
                high = ceil(np.divide(j - 1, cosArc) + (y - half_y) * tanArc + half_x)
                low = ceil(np.divide(j + 1, cosArc) + (y - half_y) * tanArc + half_x)
                if low < 0:
                    low = 0
                if high >= size[1]:
                    high = size[1]
                for x in range(low, high):
                    val += img[y, x]
            val *= 0.5
            if val > maxi:
                maxi = val
            output[i, ind] = val

    # 0度(水平)
    output[0, : quant_rho - half_x] = 0
    output[0, quant_rho + size[1] - half_x :] = 0
    for ind, j in enumerate(
        range(0, size[1]), quant_rho - half_y
    ):  # range(quantRho-halfY,quantRho+size[1]-halfX) # rho
        val = np.sum(img[:, j])
        if val > maxi:
            maxi = val
        output[0, ind] = val
    # 90度(垂直射線)
    output[half_deg, : quant_rho - half_y] = 0
    output[half_deg, quant_rho + size[0] - half_y :] = 0
    for ind, j in enumerate(range(0, size[0]), quant_rho - half_y):  # rho
        val = np.sum(img[j])
        if val > maxi:
            maxi = val
        output[half_deg, ind] = val
    return output


@njit(
    [
        'float64[:,:](uint8[:,:],UniTuple(int64,2))',
        'float64[:,:](float32[:,:],UniTuple(int64,2))',
        'float64[:,:](float64[:,:],UniTuple(int64,2))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def inv_randon_transform_standard(img: IMG_GRAY, size):
    """Inverse Randon Transform"""
    projSize = img.shape
    # const
    quantDeg = projSize[0]
    quantRho = projSize[1] // 2
    constArc = np.divide(pi, quantDeg)  # 每次增加的弳度
    halfY, halfX = size[0] // 2, size[1] // 2

    output = np.zeros(size, dtype=np.float64)
    matCos = np.empty(quantDeg, dtype=np.float64)
    matSin = np.empty(quantDeg, dtype=np.float64)
    for k in range(quantDeg):
        matCos[k] = cos(k * constArc)
        matSin[k] = sin(k * constArc)

    for k in range(quantDeg):
        for y in range(size[0]):
            y_ = (
                np.multiply(halfX, matCos[k])
                + np.multiply(y - halfY, matSin[k])
                - quantRho
            )
            rho = -y_
            for x in range(size[1]):
                output[y, x] += img[k, int(rho)]
                rho += matCos[k]
    return np.divide(output, np.amax(output))


@njit(
    [
        'float32[:](int64,float64)',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def window_kernel(ksize, c):
    """Hamming window(c=0.54) or a Hann window(c=0.5) in spatial domain."""
    # size is odd, c=0.5 or c=0.54
    # 頻率順序同numpy.fft.fft， 0, 1, 2,..., size//2, -size/2, -size/2+1, ..., -1
    output = np.empty(ksize, np.float32)
    # const
    const = 2 * pi / ksize
    quant = ksize // 2 + 1
    output[0] = 0

    for i in range(1, quant):
        temp = np.multiply(c - 1, cos(np.multiply(const, i)))
        output[i] = output[-i] = np.multiply(c - temp, i)
    return output


@njit(
    [
        'float32[:](int64,float64)',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def window_kernel_half(rfftSize, c):
    """Hamming window(c=0.54) or a Hann window(c=0.5) in frequency domain."""
    output = np.empty(rfftSize, np.float32)
    # const
    const = pi / rfftSize  # 2*pi / (2*rfftSize), rfft後的size只有原本一半
    for i in range(rfftSize):
        temp = np.multiply(c - 1, cos(np.multiply(const, i)))
        output[i] = np.multiply(c - temp, i)
    return output


def filtered_inv_randon_transform(img, size, c: float = 0.54):
    proj_size = img.shape  # Size of Projection Image

    rfft_projection = rfft2(img)
    kernel = window_kernel_half(rfft_projection.shape[1], c)

    for i in range(proj_size[0]):
        rfft_projection[i] = np.multiply(rfft_projection[i], kernel)

    rfft_projection = irfft2(rfft_projection)
    return inv_randon_transform_standard(rfft_projection, size)
