"""Special functions such as beta incomplete function."""

import math

from numba import njit
import numpy as np
from utils.img_type import Array2D

SQRT2PI = math.sqrt(2 * math.pi)
"""sqrt(2 * pi)"""

LANCZOS_G = 8
LANCZOS_COEFFS = (
    0.9999999999999999,
    1975.373902357885,
    -4397.382392792243,
    3462.632845986272,
    -1156.985143163117,
    154.5381505025278,
    -6.253671612368916,
    0.03464276245473681,
    -7.477617197444298e-7,
    6.30412538218523e-8,
    -2.7405717035683877e-8,
    4.048694881756761e-9,
)
"""Gamma function approximation by Lanczos approximation whith

G = 8 and N = 12.
Only calculate the sumnation part.
"""


@njit(
    ['float64(float64)', 'float32(float32)'],
    nogil=True,
    cache=True,
    fastmath=True,
)
def lanczos_sum(a: float) -> float:
    if math.isnan(a) or a <= 0:
        return float('nan')
    s = LANCZOS_COEFFS[0]
    for coeff in LANCZOS_COEFFS:
        s += coeff / a
        a += 1
    return s


BETA_CONST = SQRT2PI * math.exp(0.5 - LANCZOS_G)


@njit(
    ['float64(float64,float64)', 'float32(float32,float32)'],
    nogil=True,
    cache=True,
    fastmath=True,
)
def beta(a: float, b: float) -> float:
    if math.isnan(a + b) or a <= 0 or b <= 0:
        return float('nan')
    elif a == 1:
        return 1 / b
    elif b == 1:
        return 1 / a
    elif a < 0.5:
        return math.pi / (b * math.sin(math.pi * a) * beta(1 - a, a + b))
    g = LANCZOS_G
    s = lanczos_sum(a) * (lanczos_sum(b) / lanczos_sum(a + b))
    c = a + b + g - 0.5
    a -= 0.5
    b -= 0.5
    return BETA_CONST * ((a + g) / c) ** a * ((b + g) / c) ** b / math.sqrt(c) * s


@njit(
    [
        'float64(float64, float64, float64, bool)',
        'float32(float32, float32, float32, bool)',
        'float64(float64, float64, float64, Omitted(True))',
        'float32(float32, float32, float32, Omitted(True))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def betainc_single_value(
    a: float, b: float, x: float, regularized: bool = True
) -> float:
    if a <= 0 or b <= 0:
        return float('nan')
    elif math.isnan(x) or x < 0 or x > 1:
        return float('nan')
    elif x < 1e-9:
        return 0.0
    elif 1.0 - x < 1e-9:
        return 1.0 if regularized else beta(a, b)
    # Compute directly may be slow.
    # Scale
    scale = (x**a / beta(a, b)) if regularized else x**a

    result = 1.0 / a  # 0-th term
    b = 1.0 - b
    a += 1.0
    n = 1.0
    power = x
    for _ in range(500):
        resultM1 = result
        # calculate new data
        result += b / (n * a) * power
        # update
        b *= b + 1.0
        a += 1.0
        n *= n + 1.0
        power *= x
        if abs(result - resultM1) < abs(1e-9 * (result or 1)):
            break
    return result * scale


@njit(
    [
        'float64[:,:](float64, float64, float64[:,:], bool)',
        'float64[:,:](float64, float64, float64[:,:], Omitted(True))',
        'float32[:,:](float32, float32, float32[:,:], bool)',
        'float32[:,:](float32, float32, float32[:,:], Omitted(True))',
    ],
    nogil=True,
    cache=True,
    fastmath=True,
)
def betainc(
    a: float,
    b: float,
    img: Array2D[np.float32 | np.float64],
    regularized: bool = True,
):
    if a <= 0 or b <= 0:
        return np.full_like(img, np.nan)
    output = np.empty_like(img)

    scale = (1 / beta(a, b)) if regularized else 1
    for y, row in enumerate(img):
        output_row = output[y]
        for x, val in enumerate(row):
            output_row[x] = betainc_single_value(a, b, val, False) * scale

    return output
