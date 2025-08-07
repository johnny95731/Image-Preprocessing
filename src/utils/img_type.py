from typing import TypeVar

from numpy import (
    dtype,
    generic,
    int16,
    int64,
    ndarray,
    uint8,
    float32,
    float64,
    uint32,
    complex64,
    complex128,
)

T = TypeVar('T', bound=generic, covariant=True)
Array1D = ndarray[tuple[int], dtype[T]]
Array2D = ndarray[tuple[int, int], dtype[T]]
Array3D = ndarray[tuple[int, int, int], dtype[T]]
Tensor = ndarray[tuple[int, ...], dtype[T]]


# Array Typeing
Arr8U1D = Array1D[uint8]
Arr8U2D = Array2D[uint8]
Arr8U3D = Array3D[uint8]

Arr32U1D = Array1D[uint32]
Arr32U2D = Array2D[uint32]

Arr16S1D = Array1D[int16]

Arr64S1D = Array1D[int64]

Arr32F1D = Array1D[float32]
Arr32F2D = Array2D[float32]
Arr32F3D = Array3D[float32]

Arr64F1D = Array1D[float64]
Arr64F2D = Array2D[float64]

Arr64C2D = Array2D[complex64]
Arr128C2D = Array2D[complex128]


#
ARR_1D = Arr8U1D | Arr32U1D
ARR_2D = Arr8U2D | Arr32U2D

IMG_8U = Arr8U2D | Arr8U3D
IMG_32F = Arr32F2D | Arr32F3D

# 1-channel image
IMG_GRAY = Arr8U2D | Arr32F2D

# 3-channel image
IMG_COLOR = Arr8U3D | Arr32F3D

IMG_ARRAY = IMG_8U | IMG_32F

IMG_FREQ = Arr64C2D | Arr128C2D

# Kernel Typeing
KER_SIZE = tuple[int, int]
