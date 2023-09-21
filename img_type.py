from typing import Tuple, Literal, Union

from numpy import (
    ndarray, dtype, uint8, float32, float64, uint32, complex64, complex128
)


# Array Typeing
ARR_8U1D = ndarray[Tuple[Literal["L"]], dtype[uint8]]
ARR_8U2D = ndarray[Tuple[Literal["H", "W"]], dtype[uint8]]
ARR_8U3D = ndarray[Tuple[Literal["H", "W", 3]], dtype[uint8]]


ARR_32U1D = ndarray[Tuple[Literal["L"]], dtype[uint32]]
ARR_32U2D = ndarray[Tuple[Literal["H","W"]], dtype[uint32]]

ARR_32F1D = ndarray[Tuple[Literal["L"]], dtype[float32]]
ARR_32F2D = ndarray[Tuple[Literal["H", "W"]], dtype[float32]]
ARR_32F3D = ndarray[Tuple[Literal["H", "W", 3]], dtype[float32]]

ARR_64F2D = ndarray[Tuple[Literal["H", "W"]], dtype[float64]]

ARR_64C2D = ndarray[Tuple[Literal["H", "W"]], dtype[complex64]]
ARR_128C2D = ndarray[Tuple[Literal["H", "W"]], dtype[complex128]]


#
ARR_1D = Union[ARR_8U1D, ARR_32U1D]
ARR_2D = Union[ARR_8U2D, ARR_32U2D]

IMG_8U = Union[ARR_8U2D, ARR_8U3D]
IMG_32F = Union[ARR_32F2D, ARR_32F3D]

# 1-channel image
IMG_GRAY = Union[ARR_8U2D, ARR_32F2D]

# 3-channel image
IMG_COLOR = Union[ARR_8U3D, ARR_32F3D]

IMG_ARRAY = Union[
    IMG_8U, IMG_32F
]

IMG_FREQ = Union[ARR_64C2D, ARR_128C2D]

# Kernel Typeing
KER_SIZE = tuple[int, int]

