from typing import Optional

import cv2
import numpy as np


from numba import njit

from src.utils.img_type import IMG_8U, Arr8U2D, ARR_2D


"""
In this document, we present some morphology operation that is not provide by
OpenCV directly.

In Hit-or-Miss Operation:
    If the dtype of kernel is uint, then: 0=background, 1=foreground.
    If the dtype of kernel is int, then: -1=background, 0=ignore, 1=foreground.
"""


DEFAULT_KER = np.ones((3, 3), dtype=np.uint8)


def boundary_extraction(
    img: IMG_8U,
    kernel: ARR_2D,
):
    eroded = cv2.erode(img, kernel)
    # Note that the erode operation takes the minimum in the neighborhood of
    # img. Thus, img >= eroded and img - eroded will not overflow.
    return np.subtract(img, eroded, dtype=np.uint8)


def region_filling(
    img: Arr8U2D, marker: Arr8U2D, kernel: ARR_2D = DEFAULT_KER, max_iter: int = 10
):
    """
    Filling the hole of image that be marked in marker.
    The value of img should either be 255 or be 0.
    """
    if not max_iter:
        return img
    neg = cv2.bitwise_not(img)
    # X_1
    filled = cv2.bitwise_and(cv2.dilate(marker, kernel), neg)
    for _ in range(max_iter - 1):
        temp = cv2.bitwise_and(cv2.dilate(filled, kernel), neg)
        if not cv2.countNonZero(cv2.absdiff(temp, filled)):
            # Nothing changed after iteration.
            break
        filled = temp
    return cv2.bitwise_or(temp, img)


def connected_component(
    img: Arr8U2D, marker: Arr8U2D, kernel: ARR_2D = DEFAULT_KER, max_iter: int = 20
) -> Arr8U2D:
    """
    Extracting connected-component, that connecting the points in marker, of
    img. The value of img should either be 255 or be 0. Return component.
    marker: A image that mark the starting point(s) or index/indices of
        starting points. If marker is indices, it should have the shape (n,2),
        where 1 <= n <= img.shape[0].
    """
    if max_iter < 0:
        return img
    if marker.shape != img.shape:
        temp = np.zeros(marker.shape, dtype=np.uint8)
        for p in marker:
            temp[p[0], p[1]] = 255
        marker = temp
    # X_1
    filled = cv2.bitwise_and(cv2.dilate(marker, kernel), img)
    if max_iter:
        for _ in range(max_iter - 1):
            temp = cv2.bitwise_and(cv2.dilate(filled, kernel), img)
            if not cv2.countNonZero(cv2.absdiff(temp, filled)):
                # Nothing changed after iteration.
                break
            filled = temp
    else:
        while True:
            temp = cv2.bitwise_and(cv2.dilate(filled, kernel), img)
            if not cv2.countNonZero(cv2.absdiff(temp, filled)):
                # Nothing changed after iteration.
                break
            filled = temp
    return filled


def connected_component_region(
    img: Arr8U2D, marker: Arr8U2D, kernel: ARR_2D = DEFAULT_KER, max_iter: int = 20
) -> Arr8U2D:
    """
    Extracting connected-component, that connecting the points in marker, of
    img. The value of img should either be 255 or be 0. Return component and
    indices of component (top-left and bottom-right).
    marker: A image that mark the starting point(s) or index/indices of
        starting points. If marker is indices, it should have the shape (n,2),
        where 1 <= n <= img.shape[0].
    """
    if max_iter < 0:
        return img
    if marker.shape != img.shape:
        temp = np.zeros(marker.shape, dtype=np.uint8)
        for p in marker:
            temp[p[0], p[1]] = 255
        marker = temp
    # X_1
    filled = cv2.bitwise_and(cv2.dilate(marker, kernel), img)
    if max_iter:
        for _ in range(max_iter - 1):
            temp = cv2.bitwise_and(cv2.dilate(filled, kernel), img)
            if not cv2.countNonZero(cv2.absdiff(temp, filled)):
                # Nothing changed after iteration.
                break
            filled = temp
    else:
        while True:
            temp = cv2.bitwise_and(cv2.dilate(filled, kernel), img)
            if not cv2.countNonZero(cv2.absdiff(temp, filled)):
                # Nothing changed after iteration.
                break
            filled = temp
    indices = np.argwhere(filled > 0)
    indices = (np.min(indices, axis=0), np.max(indices, axis=0))  # row-column
    indices = (
        indices[0][::-1] - 1,
        indices[1][::-1] + 1,  # column-row
    )
    return filled, indices


__SIGNATURE_HISTOGRAM_QUANT = [
    'UniTuple(int64,2)(uint8[:,:])',
]


@njit(__SIGNATURE_HISTOGRAM_QUANT, nogil=True, cache=True, fastmath=True)
def __nonzero_index(img):
    """
    Return the first index that img[index] != 0.
    If all values in img is zero, return (-1,-1).
    """
    size = img.shape
    for y in range(size[0]):
        for x in range(size[1]):
            if img[y, x]:
                return y, x
    return -1, -1


def all_connected_components(
    img: Arr8U2D, kernel: ARR_2D = DEFAULT_KER, max_iter: int = 100
):
    """
    Extracting all connected-component of img.
    The value of img should either be 255 or be 0.
    """
    if max_iter < 0:
        return img
    components = []
    remain = img
    index = __nonzero_index(remain)
    while index != (-1, -1):
        marker = np.zeros(remain.shape, dtype=np.uint8)
        marker[index[0], index[1]] = 255
        components.append(connected_component(remain, marker, kernel, max_iter))
        remain = cv2.bitwise_xor(remain, components[-1])
        index = __nonzero_index(remain)
    return components


def all_connected_components_region(
    img: Arr8U2D, kernel: ARR_2D = DEFAULT_KER, max_iter: int = 100
):
    """
    Extracting all connected-component of img.
    The value of img should either be 255 or be 0.
    """
    if max_iter < 0 or not cv2.countNonZero(img):
        return []
    regions = []
    remain = img
    index = __nonzero_index(remain)
    while index != (-1, -1):
        marker = np.zeros(remain.shape, dtype=np.uint8)
        marker[index[0], index[1]] = 255
        component_, indices = connected_component_region(
            remain, marker, kernel, max_iter
        )
        regions.append(indices)
        remain = cv2.bitwise_xor(remain, component_)
        index = __nonzero_index(remain)
    return regions


def draw_connect_components(
    img: Arr8U2D,
    kernel: ARR_2D = DEFAULT_KER,
    max_iter: int = 100,
    overlap: bool = True,
    copy: bool = True,
):
    """
    Extracting all connected-component of img and .
    The value of img should either be 255 or be 0.
    """
    regions = all_connected_components_region(img, kernel, max_iter)
    if copy:
        drawed = img.copy()
    else:
        drawed = img
    if not overlap:
        remove_indices = []
        for i, reg in enumerate(regions, 1):
            for j, reg2 in enumerate(regions[i:], i):
                # reg contains in reg2
                if (
                    reg[0][0] >= reg2[0][0]
                    and reg[1][0] <= reg2[1][0]  # x
                    and reg[0][1] >= reg2[0][1]
                    and reg[1][1] <= reg2[1][1]  # y
                ):
                    remove_indices.append(i - 1)
                elif (  # reg2 contains in reg
                    reg2[0][0] >= reg[0][0]
                    and reg2[1][0] <= reg[1][0]  # x
                    and reg2[0][1] >= reg[0][1]
                    and reg2[1][1] <= reg[1][1]  # y
                ):
                    remove_indices.append(j)
        remove_indices.sort(reverse=True)
        for i in remove_indices:
            regions.pop(i)
    for i, reg in enumerate(regions):
        drawed = cv2.rectangle(drawed, reg[0], reg[1], 255)
    return drawed


def thinning(img: Arr8U2D, max_iter: int = 1):
    # 細線化
    kernel1 = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]], dtype=np.int16)
    kernel2 = np.array([[0, -1, -1], [1, 1, -1], [1, 1, 0]], dtype=np.int16)
    output = img
    for _ in range(4 * max_iter):
        temp = cv2.absdiff(output, cv2.morphologyEx(output, cv2.MORPH_HITMISS, kernel1))
        output = cv2.absdiff(temp, cv2.morphologyEx(temp, cv2.MORPH_HITMISS, kernel2))
        if not cv2.countNonZero(cv2.absdiff(temp, output)):
            return output
        kernel1 = cv2.rotate(kernel1, cv2.ROTATE_90_CLOCKWISE)
        kernel2 = cv2.rotate(kernel2, cv2.ROTATE_90_CLOCKWISE)
    return output


def thickening(img: Arr8U2D, max_iter: int = 1):
    # 厚化
    # Equivalent:
    # bitwise_not(
    #     thinning(bitwise_not(img))
    # )
    kernel1 = np.array([[1, 1, 1], [0, -1, 0], [-1, -1, -1]], dtype=np.int16)
    kernel2 = np.array([[0, 1, 1], [-1, -1, 1], [-1, -1, 0]], dtype=np.int16)
    output = img
    for _ in range(4 * max_iter):
        # Since the result of HMT is containing in original image,
        # the "subtract" in set and in numerical are the same.
        output = cv2.absdiff(
            output, cv2.morphologyEx(output, cv2.MORPH_HITMISS, kernel1, borderValue=0)
        )
        output = cv2.absdiff(
            output, cv2.morphologyEx(output, cv2.MORPH_HITMISS, kernel2, borderValue=0)
        )
        kernel1 = cv2.rotate(kernel1, cv2.ROTATE_90_CLOCKWISE)
        kernel2 = cv2.rotate(kernel2, cv2.ROTATE_90_CLOCKWISE)
    return output


def skelton(img: Arr8U2D, kernel: Arr8U2D) -> tuple[Arr8U2D, int]:
    # 骨架
    output = np.zeros_like(img)
    temp = img
    k = 0
    while cv2.countNonZero(temp):
        opened = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
        output = cv2.bitwise_or(output, cv2.subtract(temp, opened))
        temp = cv2.erode(temp, kernel)
        k += 1
    return output, k


def pruning(img: Arr8U2D, thinning_iter: int = 5) -> Arr8U2D:
    # 剪除
    # step1 = thinning img by following kernels.
    kernel1 = np.array([[0, -1, -1], [1, 1, -1], [0, -1, -1]], dtype=np.int16)
    kernel2 = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, -1]], dtype=np.int16)
    thined = img
    for _ in range(thinning_iter):
        for _ in range(4):
            thined = cv2.absdiff(
                thined, cv2.morphologyEx(thined, cv2.MORPH_HITMISS, kernel1)
            )
            kernel1 = cv2.rotate(kernel1, cv2.ROTATE_90_CLOCKWISE)
        for _ in range(4):
            thined = cv2.absdiff(
                thined, cv2.morphologyEx(thined, cv2.MORPH_HITMISS, kernel2)
            )
            kernel2 = cv2.rotate(kernel2, cv2.ROTATE_90_CLOCKWISE)
    # step2 = find endpoint
    endpoints = np.zeros_like(img)
    for _ in range(4):
        endpoints = cv2.bitwise_or(
            endpoints, cv2.morphologyEx(thined, cv2.MORPH_HITMISS, kernel1)
        )
        endpoints = cv2.bitwise_or(
            endpoints, cv2.morphologyEx(thined, cv2.MORPH_HITMISS, kernel2)
        )
        kernel2 = cv2.rotate(kernel2, cv2.ROTATE_90_CLOCKWISE)
        kernel1 = cv2.rotate(kernel1, cv2.ROTATE_90_CLOCKWISE)
    # step3 = dilate enpoints thinning_iter times
    ker = np.ones((3, 3), dtype=np.uint8)
    endpoints = cv2.dilate(endpoints, ker, iterations=thinning_iter)
    endpoints = cv2.bitwise_and(img, endpoints)
    return cv2.bitwise_or(thined, endpoints)


def geodesic_dilation(
    img: Arr8U2D,
    mask: Arr8U2D,
    kernel: Arr8U2D = DEFAULT_KER,
    iterations: int = 1,
) -> Arr8U2D:
    """
    iterations: Must be positive integer or None. If iterations is None, then
        it will loop until converging.
    Note that iteration=None is equivalent to morphological reconstruction by
    dilation.
    """
    output = cv2.bitwise_and(mask, cv2.dilate(img, kernel))
    if iterations:
        for _ in range(iterations - 1):
            temp = cv2.bitwise_and(mask, cv2.dilate(output, kernel))
            if not cv2.countNonZero(cv2.absdiff(output, temp)):
                break
            output = temp
    else:
        while True:
            temp = cv2.bitwise_and(mask, cv2.dilate(output, kernel))
            if not cv2.countNonZero(cv2.absdiff(output, temp)):
                break
            output = temp
    return output


def geodesic_erosion(
    img: Arr8U2D,
    mask: Arr8U2D,
    kernel: Arr8U2D = DEFAULT_KER,
    iterations: int = 1,
) -> Arr8U2D:
    """
    iterations: Must be positive integer or None. If iterations is None, then
        it will loop until converging.
    Note that iteration=None is equivalent to morphological reconstruction by
    erosion.
    """
    output = cv2.bitwise_or(mask, cv2.erode(img, kernel))
    if iterations:
        for _ in range(iterations - 1):
            temp = cv2.bitwise_or(mask, cv2.erode(output, kernel))
            if not cv2.countNonZero(cv2.absdiff(output, temp)):
                break
            output = temp
    else:
        while True:
            temp = cv2.bitwise_or(mask, cv2.erode(output, kernel))
            if not cv2.countNonZero(cv2.absdiff(output, temp)):
                break
            output = temp
    return output


def opening_by_rec(img, ker_erosion=DEFAULT_KER, ker_rec=DEFAULT_KER, order: int = 1):
    """
    Opening by reconstruction.
    """
    eroded = cv2.erode(img, ker_erosion, iterations=order)
    return geodesic_dilation(eroded, img, ker_rec, None)


def closing_by_rec(img, ker_erosion=DEFAULT_KER, ker_rec=DEFAULT_KER, order: int = 1):
    """
    Closing by reconstruction.
    """
    dilated = cv2.dilate(img, ker_erosion, iterations=order)
    return geodesic_erosion(dilated, img, ker_rec, None)


def auto_region_filling(img, kernel=DEFAULT_KER):
    # Smaller kernel => Fill more region.
    marker = np.zeros_like(img, dtype=np.uint8)
    marker[:, 0] = 255 - img[:, 0]
    marker[:, -1] = 255 - img[:, -1]
    marker[0, :] = 255 - img[0, :]
    marker[-1, :] = 255 - img[-1, :]
    neg = cv2.bitwise_not(img)
    return cv2.bitwise_not(geodesic_dilation(marker, neg, kernel, iterations=None))


def border_clean(img, kernel=DEFAULT_KER):
    # Bigger kernel => Clean more component near border.
    marker = np.zeros_like(img, dtype=np.uint8)
    marker[:, 0] = 255 - img[:, 0]
    marker[:, -1] = 255 - img[:, -1]
    marker[0, :] = 255 - img[0, :]
    marker[-1, :] = 255 - img[-1, :]
    rec = geodesic_dilation(marker, img, kernel, iterations=None)
    return cv2.absdiff(img, rec)
