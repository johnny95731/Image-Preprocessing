__all__ = [
   'threshold_tovalue', 'threshold_tovalue_inv', 'threshold_binary',
   'threshold_binary_inv', 'thresholding', 'threshold_by_mask',
   'threshold_by_mask_inv', 'get_threshold_mean', 'get_threshold_itermean',
   'get_threshold_otsu', 'get_threshold_max_entropy', 'get_threshold_yen',
   'get_threshold_moments', 'get_auto_thresholding_names', 'auto_threshold',
   'auto_threshold_overall', 'kmeans_seg_8UC1', 'kmeans_seg_8UC3',
   'kmeans_seg', 'superpixel_seg', 'superpixel_slic', 'superpixel_lsc',
   'superpixel_seeds', 'grabcut', 'watershed'
]
from typing import Literal, Union, Optional, Tuple, List
from math import floor

from cython import boundscheck, wraparound
from numba import njit
from sklearn.cluster import KMeans
from cv2 import (
    convertScaleAbs, bitwise_and, bitwise_not, grabCut, watershed,
    GC_INIT_WITH_MASK
)
from cv2.ximgproc import (
    createSuperpixelSLIC, createSuperpixelLSC, createSuperpixelSEEDS
)
import numpy as np

import commons
import stats
from img_type import ARR_8U2D, ARR_8U3D, IMG_ARRAY, IMG_GRAY



# Thresholding
__SIGNATURE_THRESHOLD_TOVALUE = [
    "uint8[:,:](uint8[:,:],uint8,Optional(uint8))",
    "uint8[:,:](uint8[:,:],uint8,Omitted(None))",
    "float32[:,:](float32[:,:],float32,Optional(uint8))",
    "float32[:,:](float32[:,:],float32,Omitted(None))",
]
@njit(
    __SIGNATURE_THRESHOLD_TOVALUE,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_tovalue(
        img: IMG_GRAY,
        threshold: Union[int, float],
        value: Optional[int] = None
    ) -> IMG_GRAY:
    """
    output[y,x] = img[y,x], img[y,x] > threshold,
                = value,    otherwise.

    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    threshold : int | float
        Threshold value. Between 0 and 255.
    value : int | None, default=None
        Assigned value. Between 0 and 255.

    Returns
    -------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    if value is None:
        value = 0
    if (0 < threshold < 255):
        output = np.empty_like(img)
        for y, row in enumerate(img):
            for x, img_val in enumerate(row):
                if (img_val > threshold):
                    output[y,x] = img_val
                else:
                    output[y,x] = value
        return output
    elif threshold == 255:
        return np.full_like(img, img_val)
    else:
        return img


@njit(
    __SIGNATURE_THRESHOLD_TOVALUE,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_tovalue_inv(
        img: IMG_GRAY, threshold: Union[int, float],
        value: Optional[int] = None
    ) -> IMG_GRAY:
    """
    output[y,x] = value,    img[y,x] > threshold,
                = img[y,x], otherwise.

    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    threshold : int | float
        Threshold value. Between 0 and 255.
    value : int | None, default=None
        Assigned value. Between 0 and 255.

    Returns
    -------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    if value is None:
        value = 255
    if (0 < threshold < 255):
        output = np.empty_like(img)
        for y, row in enumerate(img):
            for x, img_val in enumerate(row):
                if (img_val > threshold):
                    output[y,x] = value
                else:
                    output[y,x] = img_val
        return output
    elif threshold == 255:
        return img
    else:
        return np.full_like(img, img_val)
    


__SIGNATURE_THRESHOLD_BINARY = [
    "uint8[:,:](uint8[:,:],uint8,Optional(uint8))",
    "uint8[:,:](uint8[:,:],uint8,Omitted(None))",
    "float32[:,:](float32[:,:],float32,Optional(float32))",
    "float32[:,:](float32[:,:],float32,Omitted(None))",
]
@njit(
    __SIGNATURE_THRESHOLD_BINARY,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_binary(
        img: IMG_GRAY, threshold: Union[int, float],
        maximum: Optional[Union[int, float]] = None
    ) -> IMG_GRAY:
    """Segment an 1-channel image by the following formula:
        output[y,x] = maximum, img[y,x] > threshold,
                    = 0,       otherwise.

    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    threshold : int | float
        Threshold value. Between 0 and 255.
    maximum : int | float | None, default=None
        The maximum of new image. Between 1 and 255.

    Returns
    -------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    if maximum is None:
        maximum = 255
    if (0 < threshold < 255):
        output = np.empty_like(img)
        for y, row in enumerate(img):
            for x, img_val in enumerate(row):
                if (img_val > threshold):
                    output[y,x] = maximum
                else:
                    output[y,x] = 0
        return output
    elif threshold == 255:
        return np.full_like(img, maximum)
    else:
        return np.zeros_like(img)


@njit(
    __SIGNATURE_THRESHOLD_BINARY,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_binary_inv(
        img: IMG_GRAY, threshold: Union[int, float],
        maximum: Optional[Union[int, float]] = None
    ) -> IMG_GRAY:
    """Segment an 1-channel image by the following formula:
        output[y,x] = 0,       img[y,x] > threshold,
                    = maximum, otherwise.

    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    threshold : int | float
        Threshold value. Between 0 and 255.
    maximum : int | float | None, default=None
        The maximum of new image. Between 1 and 255.

    Returns
    -------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    if maximum is None:
        maximum = 255
    if (0 < threshold < 255):
        output = np.empty_like(img)
        for y, row in enumerate(img):
            for x, img_val in enumerate(row):
                if (img_val > threshold):
                    output[y,x] = 0
                else:
                    output[y,x] = maximum
        return output
    elif threshold == 255:
        return np.zeros_like(img)
    else:
        return np.full_like(img, maximum)


def thresholding(
        img: IMG_GRAY,
        threshold: Union[int, float],
        value: Optional[Union[int, float]] = None,
        inv: bool = False,
        threshold_type: Literal["binary", "tovalue"] = "binary"
    ) -> IMG_GRAY:
    """Segment an 1-channel image by a fixed intensity value.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    threshold : int | float
        Threshold value. Between 0 and 255.
    value : int | float | None, default=None
        If threshold_type == "binary", turn img into a binary image with 
        values {0, value}.
        If threshold_type == "tovalue" and inv == False, put
            output[y,x] = img[y,x], img[y,x] > threshold;
                        = value, elsewise.
        If threshold_type == "tovalue" and inv == True, put
            output[y,x] = value, img[y,x] > threshold;
                        = img[y,x], elsewise.
        If value == None, then put value=0 when threshold_type == "tovalue" and
            inv == False. Otherwise, value be reassign as 255.
    inv : bool, default=False
        The compare operator is inverted or not.
    threshold_type : {"binary", "tovalue"}, default=binary
        If threshold_type == "binary", turn img into a binary image with 
        values {0, value}. If threshold_type == "tovalue", 
            output[y,x] = value, img[y,x] > threshold;
                        = img[y,x], elsewise.

    Returns
    ------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
        
    Raises
    ------
    ValueError
        If threshold_type is neither "binary" nor "tovalue".
    """
    if (threshold_type == "binary" and not inv):
        return threshold_binary(img, threshold, value)
    elif (threshold_type == "binary" and inv):
        return threshold_binary_inv(img, threshold, value)
    elif (threshold_type == "tovalue" and not inv):
        return threshold_tovalue(img, threshold, value)
    elif (threshold_type == "tovalue" and inv):
        return threshold_tovalue_inv(img, threshold, value)
    else:
        raise ValueError(
            "thresholdTypes should be either \"binary\" or \"tovalue\"."
        )


__SIGNATURE_THRESHOLD_by_MASK = [
    "uint8[:,:](uint8[:,:],uint8[:,:])",
    "uint8[:,:](uint8[:,:],int16[:,:])",
    "uint8[:,:](uint8[:,:],float32[:,:])",
    "uint8[:,:](uint8[:,:],float64[:,:])",
]
@njit(
    __SIGNATURE_THRESHOLD_by_MASK,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_by_mask(
    img: IMG_GRAY, mask: IMG_GRAY
) -> IMG_GRAY:
    """Image thresholding pointwise. Same as np.where(img > mask, 255, 0) but
    faster.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    mask : IMG_GRAY
        An 8UC1 or 32FC1 image.

    Returns
    ------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    output = np.empty_like(img)
    for y, row in enumerate(img):
        for x, img_val in enumerate(row):
            if (img_val > mask[y,x]):
                output[y,x] = 255
            else:
                output[y,x] = 0
    return output


@njit(
    __SIGNATURE_THRESHOLD_by_MASK,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def threshold_by_mask_inv(
    img: IMG_GRAY, mask: IMG_GRAY
) -> IMG_GRAY:
    """Image thresholding pointwise. Same as np.where(img > mask, 0, 255) but
    faster.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    mask : IMG_GRAY
        An 8UC1 or 32FC1 image.

    Returns
    ------
    output : IMG_GRAY
        New image after thresholding. The dtype is the same as img.
    """
    output = np.empty_like(img)
    for y, row in enumerate(img):
        for x, img_val in enumerate(row):
            if (img_val > mask[y,x]):
                output[y,x] = 0
            else:
                output[y,x] = 255
    return output



# -Auto Threshold Value
# ImageJ, auto threshold: https://imagej.net/plugins/auto-threshold
__SIGNATURE_AUTO_THRESHOLD = [
    "float32(uint8[:,:])",
]
@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_mean(img: ARR_8U2D) -> float:
    """Calculate threshold value T of single-channel image where
        T = mean(img)
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    return stats.mean_8UC1(img)[0]


@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_itermean(img: ARR_8U2D) -> float:
    """
    Calculate threshold value T of single-channel image such that
        mean(img[img<=T]) == mean(img[img>T]).
    
    Ridler, T.W., Calvard, S. (1978) Picture Thresholding Using an Iterative
    Selection Method. IEEE Transactions on Systems, Man, and Cybernetics.
    8 (8): 630–632. doi:10.1109/TSMC.1978.4310039.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    hist = stats.histogram_percentage_8UC1(img)
    n = np.arange(256, dtype=np.float32) * hist
    threshold = 0.
    new_thresh = 127.5
    # Iterate
    while abs(threshold-new_thresh) > 1:
        threshold = floor(new_thresh) + 1
        s1 = np.sum(hist[:threshold])
        if s1:
            mean1 = np.sum(n[:threshold]) / s1
        else:
            mean1 = 0
        s2 = 1 - s1
        if s2:
            mean2 = np.sum(n[threshold:]) / s2
        else:
            mean2 = 0
        new_thresh = (mean1+mean2) * 0.5
    return floor(new_thresh)


@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_otsu(img: ARR_8U2D) -> float:
    """
    Calculate threshold value of single-channel image by Otsu's method.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    hist = stats.histogram_percentage_8UC1(img)
    # Calculate mean
    cum_mean = np.empty(256, dtype=np.float32) # cumulative mean
    cum_mean[0] = 0
    for i in range(1, 256):
        cum_mean[i] = i*hist[i] + cum_mean[i-1]
    mean_g = cum_mean[-1] # global mean
    # Find maximum index
    cdf = 0
    index = 0
    maxi = 0
    for i in range(256):
        cdf += hist[i]
        if 0 < cdf < 1:
            pre = (mean_g*cdf-cum_mean[i])**2 / (cdf*(1-cdf))
            if pre > maxi:
                maxi = pre
                index = i
    return index


@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_max_entropy(img):
    """
    Calculate threshold value of single-channel image by maximize the entropy.

    Kapur, J. N., Sahoo, P. K., & Wong, A. K. C. (1985).
    A new method for gray-level picture thresholding using the entropy of the
    histogram. Computer Vision, Graphics, and Image Processing, 29(3), 273-285.
    doi:10.1016/0734-189x(85)90125-2
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    hist = stats.histogram_percentage_8UC1(img)
    # Calculate mean
    entropy = np.empty(256, dtype=np.float32) # cumulative sum entropy
    if np.isinf(np.log(hist[0])): # xln(x) -> 0 as x-> 0+.
        entropy[0] = 0
    else:
        entropy[0] = -hist[0] * np.log(hist[0])
    for i in range(1, 256):
        temp = np.log(hist[0])
        if np.isinf(temp): # xln(x) -> 0 as x-> 0+.
            entropy[i] = entropy[i-1]
        else:
            entropy[i] = -hist[i] * np.log(hist[i]) + entropy[i-1]
    En = entropy[-1]
    # Find maximum index
    cdf = 0
    index = 0
    maxi = -np.inf
    for i in range(256):
        cdf += hist[i]
        if 0 < cdf < 1:
            pre = (
                np.log(cdf*(1-cdf))+entropy[i]/cdf + (En-entropy[i-1])/(1-cdf)
            )
            if pre > maxi:
                maxi = pre
                index = i
    return index


@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_yen(img):
    """
    Calculate threshold value of single-channel image by Yen's method.

    Jui-Cheng Yen, Fu-Juay Chang, & Shyang Chang. (1995).
    A new criterion for automatic multilevel thresholding.
    IEEE Transactions on Image Processing, 4(3), 370-378.
    doi:10.1109/83.366472
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    hist = stats.histogram_percentage_8UC1(img)
    # Calculate mean
    sum_ = np.empty(256, dtype=np.float32) # cumulative sum of square of hist
    sum_[0] = hist[0]**2
    for i in range(1, 256):
        sum_[i] = hist[i]**2 + sum_[i-1]
    # Find maximum index
    cdf = 0
    index = 0
    maxi = -np.inf
    for i in range(256):
        cdf += hist[i]
        prod = sum_[i]*(sum_[-1]-sum_[i])
        if 0 < prod < 1: # implies 0 < cdf < 1.
            pre = (
                -np.log(prod) + 2*np.log(cdf*(1-cdf))
            )
            if pre > maxi:
                maxi = pre
                index = i
    return index


@njit(
    __SIGNATURE_AUTO_THRESHOLD,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def get_threshold_moments(img):
    """
    Calculate threshold value of single-channel image by Tsai's method.

    Tsai, W.-H. (1985). Moment-preserving thresolding: A new approach.
    Computer Vision, Graphics, and Image Processing, 29(3), 377-393.
    doi:10.1016/0734-189x(85)90133-1
    
    Glasbey, C. A. (1993). An Analysis of Histogram-Based Thresholding
    Algorithms. CVGIP: Graphical Models and Image Processing, 55(6), 532-537.
    doi:10.1006/cgip.1993.1040
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.

    Returns
    ------
    threshold : float
        Threshold value.
    """
    hist = stats.histogram_percentage_8UC1(img)
    # Calculate raw moments
    cdf = np.empty(256, dtype=np.float32)
    raw_moments = np.zeros(3, dtype=np.float32)
    cdf[0] = hist[0]
    for i, val in enumerate(hist[1:], 1):
        cdf[i] = val + cdf[i-1]
        raw_moments[0] += i * val
        raw_moments[1] += (i**2) * val
        raw_moments[2] += (i**3) * val
    x1 = (
        (raw_moments[0]*raw_moments[2]-raw_moments[1]**2)
        / (raw_moments[1]-raw_moments[0]**2)
    )
    x2 = (
        (raw_moments[0]*raw_moments[1]-raw_moments[2])
        / (raw_moments[1]-raw_moments[0]**2)
    )
    x0 = (
        0.5 - (raw_moments[0]+x2/2)/np.sqrt(x2**2-4*x1)
    )
    index = 0
    mini = np.inf
    for i, val in enumerate(cdf):
        if abs(val-x0) < mini:
            index = i
            mini = abs(val-x0)
    return index


def get_auto_thresholding_names() -> Tuple[str]:
    """Returns all auto global threshold methods.
    
    Returns
    ------
    auto_type : tuple
        Auto global threshold methods.
    """
    return (
        "mean", "itermean", "otsu", "max_entropy", "yen", "moments"
    )


def auto_threshold(img: ARR_8U2D, auto_type: str) -> float:
    """Get a threshold value automatically.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    auto_type : str
        The auto threshold method. See auto_threshold_types().
    
    Returns
    -------
    threshold : float
        Threshold value.
    """
    if auto_type == "mean":
        return get_threshold_mean(img)
    elif auto_type == "itermean":
        return get_threshold_itermean(img)
    elif auto_type == "otsu":
        return get_threshold_otsu(img)
    elif auto_type == "max_entropy":
        return get_threshold_max_entropy(img)
    elif auto_type == "yen":
        return get_threshold_yen(img)
    elif auto_type == "moments":
        return get_threshold_moments(img)


def auto_threshold_overall(img: ARR_8U2D) -> List[Tuple[ARR_8U2D, float]]:
    """Return all types of thresholding image and threshold value.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    
    Returns
    -------
    output : List[Tuple[new_img, val]]
        The list of image after thresholding and its threshold value.
    """
    values = [auto_threshold(img, type_) for type_ in get_auto_thresholding_names()]
    return [
        (threshold_binary(img, val), val) for val in values
    ]



# Clustering
__SIGNATURE_KMEANS_SEG_8UC1 = [
    "uint8[:,:](uint8[:,:],int64,uint32,float64)",
    "uint8[:,:](uint8[:,:],int64,Omitted(300),float64)",
    "uint8[:,:](uint8[:,:],Omitted(8),uint32,float64)",
    "uint8[:,:](uint8[:,:],int64,uint32,Omitted(1.))",
    "uint8[:,:](uint8[:,:],Omitted(8),uint32,Omitted(1.))",
    "uint8[:,:](uint8[:,:],Omitted(8),Omitted(300),float64)",
    "uint8[:,:](uint8[:,:],int64,Omitted(300),Omitted(1.))",
    "uint8[:,:](uint8[:,:],Omitted(8),Omitted(300),Omitted(1.))",
]
@njit(
    __SIGNATURE_KMEANS_SEG_8UC1,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def kmeans_seg_8UC1(
        img: ARR_8U2D, cl_counts: int = 8, max_iter: int = 300, tol: float = 1.
    ) -> ARR_8U2D:
    """Grayscale image segmentation by using kmeans. The classes only depend on
    intensity. This is about 200x times faster than OpenCV k-means since we
    calculate by histogram.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    cl_counts : int, default=8
        The counts of classes. 1 <= cl_counts <= 255.
    max_iter : int, default=300
        The maximum iteration times.
    tol : float, default=1
        Tolerance, stopping criterion.
    
    Returns
    -------
    output : ARR_8U2D
        The image after segmented.
    """
    if (not 0 < cl_counts < 256 or type(cl_counts) is int):
        raise ValueError("cl_counts should be an integer in [1, 255].")
    hist = stats.histogram_quant_8UC1(img)
    # Create arrays
    # -Current 
    centers = np.zeros(cl_counts, dtype=np.float64)
    clusters = np.zeros(256, dtype=np.uint8) # Intensity s belongs clusters i.
    counts = np.zeros(cl_counts, dtype=np.uint64)
    cost = 0
    # -In last iteration
    last_centers = np.zeros(cl_counts, dtype=np.float64)
    # Initialize
    # -initialize centers
    step = 255 / (cl_counts+1)
    for i in range(cl_counts):
        centers[i] = (i+1) * step
    # -Assign intensity to cluster and calculate cost
    for i in range(256):
        mini_index = -1
        mini = np.inf
        for c in range(cl_counts):
            d = hist[i] * (centers[c]-i)**2
            if d < mini:
                mini_index = c
                mini = d
            cost += d
        clusters[i] = mini_index
        counts[mini_index] += hist[i]
    # --Update centers
    for c in range(cl_counts):
        centers[c] = 0
    for i in range(256):
        centers[clusters[i]] += hist[i] * i
    for c in range(cl_counts):
        if counts[c]:
            centers[c] /= counts[c]
    # -Best iteration
    best_centers = centers.copy()
    best_clusters = clusters.copy()
    best_cost = cost
    
    # Iterate
    for _ in range(max_iter):
        last_centers = centers.copy()
        cost = 0
        counts[:] = 0
        # Assign intensity to cluster and calculate cost
        for i in range(256):
            mini = np.inf
            for c in range(cl_counts):
                d = (centers[c]-i)**2
                if d < mini:
                    mini_index = c
                    mini = d
                cost += d
            clusters[i] = mini_index
            counts[mini_index] += hist[i]
        # -Update centers
        for c in range(cl_counts):
            centers[c] = 0
        for i in range(256):
            centers[clusters[i]] += hist[i] * i
        for c in range(cl_counts):
            if counts[c]:
                centers[c] /= counts[c]
        # -Update best
        if cost < best_cost:
            best_centers = centers.copy()
            best_clusters = clusters.copy()
            best_cost = cost
        centers_move_dist = 0
        for i, val in enumerate(centers):
            centers_move_dist += (val-last_centers[i])**2
        centers_move_dist **= 0.5
        if centers_move_dist < tol:
            break
    # Convert best_clusters into intensity transformation table.
    for i in range(256):
        best_clusters[i] = round(best_centers[best_clusters[i]])
    return commons.transform(img, best_clusters)




__SIGNATURE_KMEANS_SEG_8UC3 = [
    "uint8[:,:,:](uint8[:,:,:],int64,uint32,float64)",
    "uint8[:,:,:](uint8[:,:,:],int64,Omitted(300),float64)",
    "uint8[:,:,:](uint8[:,:,:],Omitted(8),uint32,float64)",
    "uint8[:,:,:](uint8[:,:,:],int64,uint32,Omitted(1.))",
    "uint8[:,:,:](uint8[:,:,:],Omitted(8),uint32,Omitted(1.))",
    "uint8[:,:,:](uint8[:,:,:],Omitted(8),Omitted(300),float64)",
    "uint8[:,:,:](uint8[:,:,:],int64,Omitted(300),Omitted(1.))",
    "uint8[:,:,:](uint8[:,:,:],Omitted(8),Omitted(300),Omitted(1.))",
]
@njit(
    __SIGNATURE_KMEANS_SEG_8UC3,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def kmeans_seg_8UC3(
        img: ARR_8U3D, cl_counts: int = 8, max_iter: int = 300, tol: float = 1.
    ) -> ARR_8U3D:
    """Color image segmentation by using kmeans. Slow.
    
    Parameters
    ----------
    img : ARR_8U3D
        An 8UC3 image.
    cl_counts : int, default=8
        The counts of classes. 1 <= cl_counts <= 255.
    max_iter : int, default=300
        The maximum iteration times.
    tol : float, default=1
        Tolerance, stopping criterion.
    
    Returns
    -------
    output : ARR_8U3D
        The image after segmented.
    """
    # Create arrays
    if (not 0 < cl_counts < 256 or type(cl_counts) is int):
        raise ValueError("cl_counts should be an integer in [1, 255].")
    shape = img.shape
    data_quant = shape[0] * shape[1]
    # -Reshape image
    data = np.empty((data_quant, shape[2]), dtype=np.float32)
    x = shape[1]
    for y in range(shape[0]):
        data[y*x:(y+1)*x] = img[y]
    # -Current 
    new_centers = np.zeros((cl_counts, shape[2]), dtype=np.float32)
    # --clusters[i]=c: data i belongs clusters c
    clusters = np.zeros(shape[0]*shape[1], dtype=np.uint16)
    counts = np.zeros(cl_counts, dtype=np.uint32)
    cost = 0
    # -In last iteration
    last_centers = np.zeros(cl_counts, dtype=np.float32)
    # Initialize
    # -initialize centers
    for i in range(cl_counts):
        for j in range(shape[2]):
            new_centers[i,j] = 255 * np.random.random()
    # -Assign data to cluster and calculate cost
    for i, val in enumerate(data):
        mini_index = -1
        mini = np.inf
        for c in range(cl_counts):
            d = np.sum((new_centers[c]-val)**2)
            if d < mini:
                mini_index = c
                mini = d
            cost += d
        clusters[i] = mini_index
        counts[mini_index] += 1
    # --Update centers
    for c in range(cl_counts):
        new_centers[c] = 0
    for i in range(data_quant):
        new_centers[clusters[i]] += data[i]
    for c in range(cl_counts):
        if counts[c]:
            new_centers[c] /= counts[c]
    # -Best iteration
    best_centers = new_centers.copy()
    best_clusters = clusters.copy()
    best_cost = cost
    
    # Iterate
    for _ in range(max_iter):
        last_centers = new_centers
        cost = 0
        counts[:] = 0
        # Assign intensity to cluster and calculate cost
        for i, val in enumerate(data):
            mini = np.inf
            for c in range(cl_counts):
                d = np.sum((new_centers[c]-val)**2)
                if d < mini:
                    mini_index = c
                    mini = d
                cost += d
            clusters[i] = mini_index
            counts[mini_index] += 1
        # -Update centers
        for c in range(cl_counts):
            new_centers[c] = 0
        for i, val in enumerate(data):
            new_centers[clusters[i]] += val
        for c, val in enumerate(counts):
            if val:
                new_centers[c] /= val
        # -Update best
        if cost < best_cost:
            best_centers = new_centers
            best_clusters = clusters
            best_cost = cost
        centers_move_dist = np.linalg.norm(new_centers-last_centers)
        if centers_move_dist < tol:
            break
    # Intensity Transform
    output = np.empty_like(img)
    new_centers = np.zeros((cl_counts, shape[2]), dtype=np.uint8)
    for c, val in enumerate(best_centers):
        for i in range(shape[2]):
            new_centers[c,i] = round(best_centers[c,i])
    index = 0
    for y in range(shape[0]):
        for x in range(shape[1]):
            output[y,x] = new_centers[best_clusters[index]]
            index += 1
    return output


@boundscheck(False)
@wraparound(False)
def kmeans_seg(
        img: IMG_ARRAY, cl_counts: int = 8, max_iter: int = 300, tol: float = 1.
    ) -> IMG_ARRAY:
    """Image segmentation by using kmeans.
    
    Parameters
    ----------
    img : IMG_ARRAY
        An 8UC1 image.
    cl_counts : int, default=8
        The counts of classes. 1 <= cl_counts <= 255.
    max_iter : int, default=300
        The maximum iteration times.
    tol : float, default=1
        Tolerance, stopping criterion.
    
    Returns
    -------
    output : IMG_ARRAY
        The image after segmented.
    """
    if img.ndim == 2 and img.dtype == np.uint8:
        return kmeans_seg_8UC1(img, cl_counts, max_iter, tol)
    if img.ndim == 2:
        sample = np.float32(img.reshape((-1, 1)))
    elif img.ndim == 3:
        sample = np.float32(img.reshape((-1, 3)))
    model = KMeans(n_clusters=cl_counts, n_init="auto", tol=tol).fit(sample)
    centers = np.uint8(np.round(model.cluster_centers_))
    return centers[model.labels_].reshape(img.shape)



@boundscheck(False)
@wraparound(False)
def superpixel_seg(
        img, n_segments: int = 500, compactness: float = 10,
        sigma: float = 0, max_iter: int = 10, boundary: bool = False,
    ):
    from skimage.segmentation import slic, mark_boundaries
    label = slic(img, n_segments=n_segments, compactness=compactness,
                    sigma=sigma, max_num_iter=max_iter)
    if not boundary:
        output = np.empty_like(img, dtype=np.float32)
        for i in range(n_segments):
            n = label==i
            output[n] = np.mean(img[n], axis=0)
        return convertScaleAbs(output)
    else:
        return mark_boundaries(img, label)


@boundscheck(False)
@wraparound(False)
def superpixel_slic(
        img, region_size: int = 100, ruler: float = 10.,
        algorithm: Literal[100,101,102] = 100,
        iter_: int = 10, boundary: bool = False
    ):
    """Simple Linear Iterative Clustering."""
    model = createSuperpixelSLIC(
        img, algorithm, region_size=region_size, ruler=ruler)
    model.iterate(iter_)
    if not boundary:
        label = model.getLabels()
        output = np.empty_like(img, dtype=np.float32)
        for i in range(model.getNumberOfSuperpixels()):
            n = label==i
            output[n] = np.mean(img[n], axis=0)
        return convertScaleAbs(output)
    else:
        mask_slic = bitwise_not(model.getLabelContourMask())
        img_slic = bitwise_and(img, img, mask=mask_slic)
        return img_slic


@boundscheck(False)
@wraparound(False)
def superpixel_lsc(
        img, region_size: int = 20, ratio: float = 20,
        iter_: int = 10, boundary: bool = False
    ):
    """Linear Spectral Clustering"""
    model = createSuperpixelLSC(
        img, region_size=region_size, ratio=ratio) 
    model.iterate(iter_)
    if not boundary:
        label = model.getLabels()
        output = np.empty_like(img, dtype=np.float32)
        for i in range(model.getNumberOfSuperpixels()):
            n = label==i
            output[n] = np.mean(img[n], axis=0)
        return convertScaleAbs(output)
    else:
        mask_slic = bitwise_not(model.getLabelContourMask())
        img_slic = bitwise_and(img, img, mask=mask_slic)
        return img_slic


@boundscheck(False)
@wraparound(False)
def superpixel_seeds(
        img, n_segments: int = 500, num_levels: int = 15,
        iter_: int = 10, boundary: bool = False
    ):
    """Superpixels Extracted via Energy-Driven Sampling."""
    shape = img.shape
    if img.ndim == 2:
        model = createSuperpixelSEEDS(
            *shape[:2], 1, n_segments, num_levels
        ) 
    model.iterate(iter_)
    if not boundary:
        label = model.getLabels()
        output = np.empty_like(img, dtype=np.float32)
        for i in range(model.getNumberOfSuperpixels()):
            n = label==i
            output[n] = np.mean(img[n], axis=0)
        return convertScaleAbs(output)
    else:
        mask_slic = bitwise_not(model.getLabelContourMask())
        img_slic = bitwise_and(img, img, mask=mask_slic)
        return img_slic


# 
@boundscheck(False)
@wraparound(False)
def grabcut(
        img, mask=None, rect=None,
        iter_: int = 10, mode: Literal[0,1,2,3] = GC_INIT_WITH_MASK
    ):
    shape = img.shape
    if mask is None:
        mask = np.zeros(shape[:2], dtype=np.uint8)
    if rect is None:
        rect = (50, 50, 450, 290)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    grabCut(img, mask, rect, bgdModel, fgdModel, iter_, mode)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    return img * mask2[:,:,np.newaxis]



@boundscheck(False)
@wraparound(False)
def watershed(img, markers, c=(255,0,0), copy=True):
    markers = watershed(img, markers)
    if copy:
        output = img.copy()
    else:
        output = img
    output[markers == -1] = c
    return output
