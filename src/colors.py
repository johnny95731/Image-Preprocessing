from operator import setitem, getitem
from typing import Tuple, Union, Literal

from numba import njit

import cv2
import numpy as np

from src.utils.img_type import IMG_COLOR, IMG_ARRAY, IMG_GRAY, IMG_8U, Arr8U3D


__SIGNATURE_COLOR_SLICING = [
    'uint8[:,:,:](uint8[:,:,:],int32[:,:],int64,int64)',
    'uint8[:,:,:](uint8[:,:,:],uint8[:,:],int64,int64)',
    'uint8[:,:](uint8[:,:],int32[:,:],int64,int64)',
    'uint8[:,:](uint8[:,:],uint8[:,:],int64,int64)',
]


@njit(__SIGNATURE_COLOR_SLICING, nogil=True, cache=True, fastmath=True)
def color_slicing_by_dist(
    img: IMG_COLOR,
    dist: IMG_GRAY,
    radius: Union[float, int],
    bg: Union[float, int] = 127,
) -> IMG_COLOR:
    bg = np.uint8(bg)
    size = img.shape
    output = np.empty_like(img)
    for y in range(size[0]):
        for x in range(size[1]):
            if dist[y, x] > radius:
                # output[y,x] = bg
                setitem(getitem(output, y), x, bg)
            else:
                # Equals output[y,x] = img[y,x], but faster.
                setitem(getitem(output, y), x, getitem(getitem(img, y), x))
    return output


type_center = Union[Tuple[float, float, float], Tuple[int, int, int]]


def color_dist_supnorm(img: IMG_COLOR, center: type_center) -> IMG_COLOR:
    # The sup-norm distance between point and center.
    # numpy version:
    # dist = np.amax(np.abs(np.subtract(img, center)), axis=2)
    dist = cv2.absdiff(img[:, :, 0], center[0])
    dist = cv2.max(cv2.absdiff(img[:, :, 1], center[1]), dist)
    dist = cv2.max(cv2.absdiff(img[:, :, 2], center[2]), dist)
    return dist


def color_dist_2norm(img: IMG_COLOR, center: type_center) -> IMG_COLOR:
    # The 2-norm distance between point and center.
    dist = cv2.add(
        cv2.pow(cv2.subtract(img[:, :, 0], center[0], dtype=cv2.CV_32S), 2),
        cv2.pow(cv2.subtract(img[:, :, 1], center[1], dtype=cv2.CV_32S), 2),
    )
    dist = cv2.add(
        cv2.pow(cv2.subtract(img[:, :, 2], center[2], dtype=cv2.CV_32S), 2), dist
    )
    return dist
    # standard: return cv2.sqrt(dist)


def color_dist_1norm(img: IMG_COLOR, center: type_center) -> IMG_COLOR:
    dist = cv2.add(
        cv2.absdiff(img[:, :, 0], center[0]),
        cv2.absdiff(img[:, :, 1], center[1]),
        dtype=cv2.CV_32S,
    )
    dist = cv2.add(cv2.absdiff(img[:, :, 2], center[2]), dist, dtype=cv2.CV_32S)
    return dist


def color_slicing(
    img: IMG_COLOR,
    center: type_center,
    radius: Union[float, int],
    bg: Union[float, int] = 127,
    norm: Literal[0, 1, 2] = 2,
) -> IMG_COLOR:
    if not norm:
        dist = color_dist_supnorm(img, center)
    elif norm == 1:
        dist = color_dist_1norm(img, center)
    elif norm == 2:
        dist = color_dist_2norm(img, center)
        radius **= 2
    return color_slicing_by_dist(img, dist, radius, bg)


# 色彩空間轉換 Color Space Conversions


def bgr_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return img
    return img[:, :, ::-1]


def rgba_to_rgb(
    img: IMG_ARRAY, bg: Union[Literal['w'], Literal['k']] = 'w'
) -> IMG_ARRAY:
    """
    Convert rgba image to rgb image.
    """
    shape = img.shape
    if len(shape) != 3 or shape[2] != 4:
        return img
    output = np.empty((*shape[:2], 3))
    alpha = np.divide(img[:, :, 3], 255, dtype=np.float32)
    if bg == 'w':  # white
        b = np.subtract(255, img[:, :, 3], dtype=np.float32)
        for i in range(3):
            output[:, :, i] = np.add(
                np.multiply(img[:, :, i], alpha, dtype=np.float32), b, dtype=np.float32
            )
    elif bg == 'k':  # black
        for i in range(3):
            output[:, :, i] = np.multiply(img[:, :, i], alpha, dtype=np.float32)
    return output.astype(np.uint8)


__SIGNATURE_RGB_TO_GRAY = [
    'uint8[:,:](uint8[:,:,:],UniTuple(float32,3))',
    'uint8[:,:](uint8[:,:,:],Omitted((0.299,0.587,0.114)))',
]


@njit(__SIGNATURE_RGB_TO_GRAY, nogil=True, cache=True, fastmath=True)
def rgb_to_gray(
    img: Arr8U3D, weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)
) -> Arr8U3D:
    if img.ndim == 2:
        return img
    size = img.shape
    output = np.empty(size[:2], dtype=img.dtype)
    for y in range(size[0]):
        for x in range(size[1]):
            val = 0
            for d in range(3):
                val += img[y, x, d] * weights[d]
            output[y, x] = val
    return output


def rgb_to_cmy(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return img
    return np.subtract(255, img, dtype=np.uint8)


def cmy_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    return rgb_to_cmy(img)


def cmy_to_cmyk(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return img
    size = img.shape
    output = np.empty((*size[:2], 4), dtype=np.uint8)
    k = output[:, :, 3] = np.amin(img, axis=2)  # k
    for i in range(3):
        output[:, :, i] = np.subtract(img[:, :, i], k)
    return output


def cmyk_to_cmy(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return img
    size = img.shape
    output = np.empty((*size[:2], 3), dtype=np.float32)
    k = output[:, :, 3]
    for i in range(3):
        output[:, :, i] = np.add(img[:, :, i], k, dtype=np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def rgb_to_yiq(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    weight = np.array(
        [[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]],
        np.float32,
    )
    for i in range(3):
        output[:, :, i] = np.dot(img, weight[i])
    return output


def yiq_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    weight = np.array(
        [[1, 0.95569, 0.61986], [1, -0.27158, -0.64687], [1, -1.10818, 1.70506]],
        np.float32,
    )
    for i in range(3):
        output[:, :, i] = np.dot(img, weight[i])
    return output


def rgb_to_yuv(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    weight = np.array(
        [[0.299, 0.587, 0.114], [-0.169, -0.331, 0.5], [0.5, -0.419, -0.081]],
        np.float32,
    )
    for i in range(3):
        output[:, :, i] = np.dot(img, weight[i])
    output[:, :, 1:] += 128
    return output


def yuv_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    weight = np.array(
        [[1, 0.00093, 1.40169], [1, -0.3437, -0.71417], [1, 1.77216, 0.00099]],
        np.float32,
    )
    for i in range(3):
        output[:, :, i] = np.dot(img, weight[i])
    output[:, :, 0] -= 179.53536
    output[:, :, 1] += 135.40736
    output[:, :, 2] -= 226.9632
    return output


def xyz_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:  # CIE 1931 XYZ to RGB
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    weight = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.969266, 1.8760108, 0.041556],
            [0.0556434, -0.2040259, -1.0572252],
        ],
        np.float32,
    )
    for i in range(3):
        output[:, :, i] = np.dot(img, weight[i])
    return output


def rgb_to_hsv(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    img = img.astype(np.int32)

    cMax = np.amax(img, axis=2)
    cMin = np.amin(img, axis=2)
    delta = cMax - cMin
    # Calculate H
    n = cMax == img[:, :, 0]  # cMax = R
    output[n, 0] = np.mod(
        np.divide(img[n, 1] - img[n, 2], delta[n], where=delta[n] != 0), 6
    )

    n = cMax == img[:, :, 1]  # cMax = G
    output[n, 0] = np.divide(img[n, 2] - img[n, 0], delta[n], where=delta[n] != 0) + 2

    n = cMax == img[:, :, 2]  # cMax = B
    output[n, 0] = np.divide(img[n, 0] - img[n, 1], delta[n], where=delta[n] != 0) + 4

    output[:, :, 0] = output[:, :, 0] * 60

    n = output[:, :, 0] < 0
    output[n, 0] = output[n, 0] + 360
    # Calculate S
    n = cMax != 0
    output[n, 1] = np.divide(delta[n], cMax[n], where=cMax[n] != 0)
    # Calculate V
    output[:, :, 2] = cMax
    return output


def hsv_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)
    # pre
    hh = img[:, :, 0] / 60
    i = hh.astype(np.int16)
    ff = hh - i
    p = img[:, :, 2] * (1 - img[:, :, 1])  # V * (1-S)
    q = img[:, :, 2] * (1 - (img[:, :, 1] * ff))  # V * ( 1 - S * ff )
    t = img[:, :, 2] * (1 - (img[:, :, 1] * (1 - ff)))  # V * ( 1 - S * (1-ff) )

    output[:, :, 0], output[:, :, 1], output[:, :, 2] = img[:, :, 2], p, q

    index = i == 0
    output[index, 0], output[index, 1], output[index, 2] = (
        img[index, 2],
        t[index],
        q[index],
    )

    index = i == 1
    output[index, 0], output[index, 1], output[index, 2] = (
        q[index],
        img[index, 2],
        p[index],
    )

    index = i == 2
    output[index, 0], output[index, 1], output[index, 2] = (
        p[index],
        img[index, 2],
        t[index],
    )

    index = i == 3
    output[index, 0], output[index, 1], output[index, 2] = (
        p[index],
        q[index],
        img[index, 2],
    )

    index = i == 4
    output[index, 0], output[index, 1], output[index, 2] = (
        t[index],
        p[index],
        img[index, 2],
    )

    index = i == 5
    output[index, 0], output[index, 1], output[index, 2] = (
        img[index, 2],
        p[index],
        q[index],
    )

    return output.astype(np.uint8)


def rgb_to_hsl(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    img = img.astype(np.int16)
    output = np.empty_like(img, dtype=np.float32)

    cMax = np.amax(img, axis=2)
    cMin = np.amin(img, axis=2)
    delta = cMax - cMin
    s = cMax + cMin
    # Calculate H
    index = cMax == img[:, :, 0]  # cMax = R
    output[index, 0] = np.mod(
        np.divide(img[index, 1] - img[index, 2], delta[index], where=delta[index] != 0),
        6,
    )

    index = cMax == img[:, :, 1]  # cMax = G
    output[index, 0] = (
        np.divide(img[index, 2] - img[index, 0], delta[index], where=delta[index] != 0)
        + 2
    )

    index = cMax == img[:, :, 2]  # cMax = B
    output[index, 0] = (
        np.divide(img[index, 0] - img[index, 1], delta[index], where=delta[index] != 0)
        + 4
    )

    output[:, :, 0] = output[:, :, 0] * 60

    index = output[:, :, 0] < 0
    output[index, 0] = output[index, 0] + 360
    # Calculate L
    output[:, :, 2] = s / 2
    # Calculate S
    output[:, :, 1] = np.divide(delta, (2 * s), where=s != 0)
    index = output[:, :, 2] > 0.5
    output[index, 2] = delta / (2 - 2 * s)
    return output


def hsl_to_rgb(img: IMG_ARRAY) -> IMG_ARRAY:
    if img.ndim == 2:
        return
    output = np.empty_like(img, dtype=np.float32)

    deg = (img[:, :, 0] / 60).astype(np.int16)
    C = (1 - np.abs(2 * img[:, :, 2] - 1)) * img[:, :, 1]  # (1-|2L-1|)*S
    X = C * (
        1 - np.abs(np.mod(img[:, :, 0] / 60, 2) - 1)
    )  # C × (1 - |(H / 60°) mod 2 - 1|)
    # m = img[:,:,2] - C/2 # L - C / 2

    index = deg == 0
    output[index, 0], output[index, 1], output[index, 2] = C[index], X[index], 0

    index = deg == 1
    output[index, 0], output[index, 1], output[index, 2] = X[index], C[index], 0

    index = deg == 2
    output[index, 0], output[index, 1], output[index, 2] = 0, C[index], X[index]

    index = deg == 3
    output[index, 0], output[index, 1], output[index, 2] = 0, X[index], C[index]

    index = deg == 4
    output[index, 0], output[index, 1], output[index, 2] = X[index], 0, C[index]

    index = deg == 5
    output[index, 0], output[index, 1], output[index, 2] = C[index], 0, X[index]
    return output.astype(np.uint8)


def complement(img: IMG_8U, maximum=1) -> IMG_ARRAY:  # 補色, 負片
    if img.dtype.kind == 'i':
        return cv2.bitwise_not(img)
    else:
        return np.subtract(maximum, img)
