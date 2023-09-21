__all__ = [
    "get_noises", "get_arguments", "add_uniform_noise", "add_gaussian_noise", "add_rayleigh_noise", "add_rayleigh_noise_8UC1", "add_gamma_noise", "add_gamma_noise_8UC1", "add_exponential_noise", "add_exponential_noise_8UC1", "add_single_color_noise", "add_salt_noise", "add_pepper_noise", "add_salt_and_pepper_noise", "add_beta_noise"
]
import random

from cython import wraparound
from numba import njit

import numpy as np
from img_type import ARR_8U2D, ARR_32F2D, IMG_GRAY


@wraparound(False)
def get_noises() -> tuple[str]:
    """Returns all available noises.

    Returns
    -------
    output : tuple
        Available noises
    """
    return (
        "uniform", "gaussian", "rayleigh", "gamma", "exponential",
        "salt-and-pepper", "salt", "pepper", "single color",
        "beta"
    )


__SIGNATURE_UNIFORM = [
    "float32[:,:](uint8[:,:],int64,int64)",
    "float32[:,:](float32[:,:],int64,int64)",
]
@njit(
    __SIGNATURE_UNIFORM,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_uniform_noise(
        img: IMG_GRAY, low: int, high: int
    ) -> ARR_32F2D:
    """
    Add discrete uniform noise to a given image.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    low : int
        Minimum of noise. Must less than high.
    high : int
        Maximum of noise. Must greater than low.
    
    Returns
    -------
    output : ARR_32F2D
        img with noise.
    """
    # Add discrete uniform noises to a given image.
    # mean = (high+low)/2, var = (high-low)^2 / 12
    # pdf(x) = 1/(high-low), x∈{low, low+1, ..., high}
    size = img.shape
    output = np.empty(size, dtype=np.float32)
    for y in range(size[0]):
        for x in range(size[1]):
            output[y,x] = np.add(img[y,x], random.uniform(low, high))
    return output


SIGNATURE_FLOAT_1ARG = [
    "float32[:,:](uint8[:,:],float32)",
    "float32[:,:](float32[:,:],float32)",
]
SIGNATURE_FLOAT_2ARGS = [
    "float32[:,:](uint8[:,:],float32,float32)",
    "float32[:,:](float32[:,:],float32,float32)",
]
SIGNATURE_FLOAT_1ARG_8UC1 = [
    "uint8[:,:](uint8[:,:],float32)",
]
SIGNATURE_FLOAT_2ARGS_8UC1 = [
    "uint8[:,:](uint8[:,:],float32,float32)",
]
@njit(
    SIGNATURE_FLOAT_2ARGS,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_gaussian_noise(
        img: IMG_GRAY, center: float, sigma: float
    ) -> ARR_32F2D:
    """
    Add discrete uniform noise to a given image.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    center : loat
        The center of gaussian noise.
    sigma : float
        The standard deviation of gaussian noise. Must be possitive.
    
    Returns
    -------
    output : ARR_32F2D
        img with noise.
    """
    # mean=center, var=sigma^2
    # pdf(x)= 1/(sqrt(2*pi)*sigma) * exp(-(x-center)^2 / (2*sigma^2 ))
    size = img.shape
    output = np.empty(size, dtype=np.float32)
    for y in range(size[0]):
        for x in range(size[1]):
            output[y,x] = np.add(img[y,x], np.random.normal(center, sigma))
    return output


@njit(
    SIGNATURE_FLOAT_1ARG,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_rayleigh_noise(
        img: IMG_GRAY, scale: float = 0.1
    ) -> ARR_32F2D:
    """
    Add Rayleigh noise to a given image.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 or 32FC1 image.
    scale : float
        Scale. Must be nonzero. If scale is negative, the noise is multiplied
        by -1.
    
    Returns
    -------
    output : ARR_32F2D
        img with noise.
    """
    # mean=scale*sqrt(pi/2), var=(4-pi)/2 * scale^2
    # pdf(x) = x/(scale^2) * exp(-x^2 / (2*scale^2)), x≥0
    size = img.shape
    
    output = np.empty(size, dtype=np.float32)
    if scale > 0:
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.add(img[y,x], np.random.rayleigh(scale))
    elif scale < 0:
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.subtract(img[y,x], np.random.rayleigh(scale))
    return output


@njit(
    SIGNATURE_FLOAT_1ARG_8UC1,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_rayleigh_noise_8UC1(
        img: ARR_8U2D, scale: float = 0.1
    ) -> ARR_8U2D:
    """Add Rayleigh noise to a given image. The noise will be chosen so that
    img add noise is in [0,255].
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    scale : float
        Scale. Must be nonzero. If scale is negative, the noise is multiplied
        by -1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    # mean=scale*sqrt(pi/2), var=(4-pi)/2 * scale^2
    # pdf(x) = x/(scale^2) * exp(-x^2 / (2*scale^2)), x≥0
    size = img.shape
    
    output = np.empty(size, dtype=np.uint8)
    if scale > 0:
        for y in range(size[0]):
            for x in range(size[1]):
                val = np.add(img[y,x], np.random.rayleigh(scale))
                while not (0 <= val <= 255):
                    val = np.add(img[y,x], np.random.rayleigh(scale))
                output[y,x] = round(val)
    elif scale < 0:
        for y in range(size[0]):
            for x in range(size[1]):
                val = np.subtract(img[y,x], np.random.rayleigh(scale))
                while not (0 <= val <= 255):
                    val = np.subtract(img[y,x], np.random.rayleigh(scale))
                output[y,x] = round(val)
    return output



@njit(
    SIGNATURE_FLOAT_2ARGS,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_gamma_noise(
        img: IMG_GRAY, shape: float, scale: float
    ) -> ARR_32F2D:
    """
    Add Gamma(Erlang) noise to a given image.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    shape : float
        Shape. Must be positive.
    scale : float
        Scale. Must be nonzero. If scale is negative, the noise is multiplied
        by -1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    # Add Gamma(Erlang) noises to a given image.
    # mean=shape * scale, var=shape * scale**2
    # pdf(x): x^{shape-1}/(Γ(shape) * scale^shape) * exp(-x/scale), x≥0
    # where Γ is the gamma function
    size = img.shape
    
    output = np.empty(size, dtype=np.float32)
    if scale > 0:
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.add(img[y,x],
                                     random.gammavariate(shape, scale))
    elif scale < 0:
        scale = -scale
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.subtract(img[y,x],
                                          random.gammavariate(shape, scale))
    return output


@njit(
    SIGNATURE_FLOAT_2ARGS_8UC1,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_gamma_noise_8UC1(
        img: ARR_8U2D, shape: float, scale: float
    ) -> ARR_8U2D:
    """
    Add Gamma(Erlang) noise to a given image. The noise will be chosen so that
    img add noise is in [0,255].
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    shape : float
        Shape. Must be positive.
    scale : float
        Scale. Must be nonzero. If scale is negative, the noise is multiplied
        by -1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    # Add Gamma(Erlang) noises to a given image.
    # mean=shape * scale, var=shape * scale**2
    # pdf(x): x^{shape-1}/(Γ(shape) * scale^shape) * exp(-x/scale), x≥0
    # where Γ is the gamma function
    size = img.shape
    output = np.empty(size, dtype=np.uint8)
    if scale > 0:
        for y in range(size[0]):
            for x in range(size[1]):
                val = np.add(img[y,x], random.gammavariate(shape, scale))
                while not (0 <= val <= 255):
                    val = np.add(img[y,x],
                                 random.gammavariate(shape, scale))
                output[y,x] = round(val)
    elif scale < 0:
        scale = -scale
        for y in range(size[0]):
            for x in range(size[1]):
                val = np.subtract(img[y,x], random.gammavariate(shape, scale))
                while not (0 <= val <= 255):
                    val = np.subtract(img[y,x],
                                      random.gammavariate(shape, scale))
                output[y,x] = round(val)
    return output



@njit(
    SIGNATURE_FLOAT_1ARG,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_exponential_noise(
        img: IMG_GRAY, scale: float
    ) -> ARR_32F2D:
    """
    Add exponential noise to a given image.
    
    Parameters
    ----------
    img : IMG_GRAY
        An 8UC1 image.
    scale : float
        Scale. Must be nonzero. The range is [0,∞) if scale > 0. And the
        range is (-∞,0] if scale < 0.
    
    Returns
    -------
    output : ARR_32F2D
        img with noise.
    """
    # mean=scale, var=scale**2
    # pdf(x): exp(-x/scale) / scale
    size = img.shape
    output = np.empty(size,dtype=np.float32)
    rate = 1 / scale
    
    for y in range(size[0]):
        for x in range(size[1]):
            output[y,x] = np.add(img[y,x], random.expovariate(rate))
    return output


@njit(
    SIGNATURE_FLOAT_1ARG,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_exponential_noise_8UC1(
        img: ARR_8U2D, scale: float
    ) -> ARR_8U2D:
    """
    Add exponential noise to a given image. The noise will be chosen so that
    img add noise is in [0,255].
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    scale : float
        Scale. Must be nonzero. The range is [0,∞) if scale > 0. And the
        range is (-∞,0] if scale < 0.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    # mean=scale, var=scale**2
    # pdf(x): exp(-x/scale) / scale
    size = img.shape
    output = np.empty(size,dtype=np.float32)
    rate = np.reciprocal(scale)
    
    for y in range(size[0]):
        for x in range(size[1]):
            val = np.add(img[y,x], random.expovariate(rate))
            while not (0 <= val <= 255):
                val = np.add(img[y,x], random.expovariate(rate))
            output[y,x] = round(val)
    return output



SIGNATURE_SINGLE_COLOER_NOISE = [
    "uint8[:,:](uint8[:,:],float32,uint8)",
]
@njit(
    SIGNATURE_SINGLE_COLOER_NOISE,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_single_color_noise(
        img: ARR_8U2D, prob: float, color: int
    ) -> ARR_8U2D:
    """Randomly assign a given color to a given image. Generalized case of
    pepper (or, salt) noise.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    prob : float
        The probability that img be add a noise. Must between 0 and 1.
    color : int
        Noise color. Must be uint8.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    size = img.shape
    output = np.empty_like(img)
    
    for y in range(size[0]):
        for x in range(size[1]):
            if (random.random() < prob):
                output[y,x] = color
            else:
                output[y,x] = img[y,x]
    return output
    

@wraparound(False)
def add_salt_noise(
        img: ARR_8U2D, prob: float
    ) -> ARR_8U2D:
    """
    Add salt noises to a given image.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    prob : float
        The probability that img be added salt noise. Must between 0 and 1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    return add_single_color_noise(img, prob, 255)


@wraparound(False)
def add_pepper_noise(
        img: ARR_8U2D, prob: float
    ) -> ARR_8U2D:
    """
    Add pepper noise to a given image.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    prob : float
        The probability that img be added pepper noise. Must between 0 and 1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    return add_single_color_noise(img, prob, 0)


SIGNATURE_SALT_AND_PEPPER = [
    "uint8[:,:](uint8[:,:],float32,float32)",
]
@njit(
    SIGNATURE_SALT_AND_PEPPER,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_salt_and_pepper_noise(
        img: ARR_8U2D, p_salt: float, p_pepper: float
    ) -> ARR_8U2D:
    """Add salt-and-pepper noise to a given image.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    p_salt : float
        The probability that img be add salt noise. Must between 0 and 1 and
        p_salt+p_pepper <= 1.
    p_pepper : float
        The probability that img be add pepper noise. Must between 0 and 1 and
        p_salt+p_pepper <= 1.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    size = img.shape
    output = np.empty_like(img)
    p1 = p_salt+p_pepper
    
    for y in range(size[0]):
        for x in range(size[1]):
            # Generate a random number p in [0,1).
            # If p∈[0,p_salt), then output=255.
            # If p∈[p_salt,p1), then output=0. (Note that length of interval
            # [p_salt,p1) is probPepper.)
            # And if p∈[p1,1), then output=img.
            p = random.random()
            if (p < p_salt): # Prob(output=255) = p_salt
                output[y,x] = 255
            elif (p < p1): # Prob(output=0) = p_pepper
                output[y,x] = 0
            else:
                output[y,x] = img[y,x]
    return output



SIGNATURE_BETA = [
    "float32[:,:](uint8[:,:],float32,float32,float32)",
    "float32[:,:](uint8[:,:],float32,float32,Omitted(255))",
]
@njit(
    SIGNATURE_BETA,
    nogil=True, cache=True, fastmath=True)
@wraparound(False)
def add_beta_noise(
        img: ARR_8U2D, a: float, b: float, maximum: float = 255
    ) -> ARR_8U2D:
    """Add beta noise to a given image.
    
    Parameters
    ----------
    img : ARR_8U2D
        An 8UC1 image.
    a : float
        Shape of incomplete beta function. Must be positive.
    b : float
        Shape of incomplete beta function. Must be nonzero. If scale is negative, the noise is multiplied
        by -1.
    maximum : float, default=255
        The maximum of noise. Muse between 0 and 255.
    
    Returns
    -------
    output : ARR_8U2D
        img with noise.
    """
    # Add Gamma(Erlang) noises to a given image.
    # mean=shape * scale, var=shape * scale**2
    # pdf(x): x^{shape-1}/(Γ(shape) * scale^shape) * exp(-x/scale), x≥0
    # where Γ is the gamma function
    size = img.shape
    
    output = np.empty(size, dtype=np.float32)
    if b > 0:
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.add(img[y,x],
                                     maximum*random.betavariate(a, b))
    elif b < 0:
        b = -b
        for y in range(size[0]):
            for x in range(size[1]):
                output[y,x] = np.subtract(img[y,x],
                                          maximum*random.betavariate(a, b))
    return output
