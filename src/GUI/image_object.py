from typing import TypeVar, Generic, NewType

import numpy as np
import cv2
from pyfftw.interfaces.scipy_fft import rfft2, irfft2

import src.enhance as enhance
from src.noises import (
    add_uniform_noise,
    add_gaussian_noise,
    add_rayleigh_noise,
    add_gamma_noise,
    add_exponential_noise,
    add_single_color_noise,
    add_salt_and_pepper_noise,
    add_beta_noise,
)
from src.seg import thresholding, get_auto_thresholding_names, auto_threshold
from src.filters.spatial_filters import (
    get_gradient_operator_names,
    get_gradient_operator,
    gradient_norm,
    gradient_uint8_overflow,
    mean_blur,
    gaussian_blur,
    median_blur,
    bilateral_blur,
    unsharp_masking,
)
from src.colors import rgb_to_gray
import src.utils.stats as stats

from src.utils.img_type import Arr8U2D, IMG_8U, KER_SIZE


Mean = NewType('Mean', np.float32)
Var = NewType('Var', np.float32)
Std = NewType('Std', np.float32)
Skewness = NewType('Skewness', np.float32)
Kurtosis = NewType('Kurtosis', np.float32)
Max = NewType('Max', np.uint8)
Min = NewType('Min', np.uint8)
Median = NewType('Median', np.uint8)


class BasicOperators:
    stats_labels = (
        'Mean',
        'Var',
        'Std',
        'Skewness',
        'Kurtosis',
        'Max',
        'Min',
        'Median',
    )

    def get_statistics(
        img: Arr8U2D,
    ) -> tuple[Mean, Var, Std, Skewness, Kurtosis, Max, Min, Median]:
        # Order statistice
        max_, min_ = stats.max_min(img)
        median_ = stats.median(img)
        # Moments
        mean_, var_, skew, kurtosis = stats.moments(img)
        std_ = np.sqrt(var_)
        return mean_, var_, std_, skew, kurtosis, max_, min_, median_

    def histogram_equalization(img: IMG_8U) -> IMG_8U:
        """Applying histogram equalization to img."""
        # Check chanels
        if img.ndim == 3:  # 彩色圖片
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = enhance.histogram_equalization(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif img.ndim == 2:  # 灰階圖片
            return enhance.histogram_equalization(img)

    def rfft2(img: IMG_8U):
        return rfft2(img.astype(np.float32))

    def irfft2(img: IMG_8U):
        return irfft2(img)

    def thresholding(img: IMG_8U, threshold, value):
        if img.ndim == 3:  # 彩色圖片
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif img.ndim == 2:  # 灰階圖片
            return cv2.equalizeHist(img)
        return thresholding(img, threshold, value)


class EnhanceOperators:
    @staticmethod
    def get_intensity_transform_names():
        return (
            'Linear Transformation',
            'Gamma Correction',
            'Log Transformation',
            'Arctan Transformation',
            'Logistic Correction',
            'Beta Correction',
            'Level Slicing Type I',
            'Level Slicing Type II',
            'Level Slicing Type II (Inv.)',
        )

    @staticmethod
    def intensity_transform(img: Arr8U2D, op: int, args: list[float]) -> Arr8U2D:
        if op == 0:  # Linear Transformation
            return enhance.linear_transformation_8UC1(img, *args)
        elif op == 1:  # Gamma Correction
            if args[1] == 0:  # Automatically compute ratio
                args[1] = None
            return enhance.gamma_correction_8UC1(img, *args)
        elif op == 2:  # Log Transformation
            if args[0] == 0:  # Automatically compute ratio
                args[0] = None
            return enhance.log_transformation_8UC1(img, *args)
        elif op == 3:  # Arctan Transformation
            return enhance.arctan_transformation_8UC1(img, *args)
        elif op == 4:  # Logistic Correction
            return enhance.logistic_correction_8UC1(img, *args)
        elif op == 5:  # Beta Correction
            return enhance.beta_correction_8UC1(img, *args)
        elif op == 6:  # Level Slicing Type I
            args = [(round(args[0]), round(args[1])), round(args[2]), round(args[3])]
            return enhance.intensity_level_slicing_type1(img, *args)
        elif op == 7:  # Level Slicing Type II
            args = [(round(args[0]), round(args[1])), round(args[2])]
            return enhance.intensity_level_slicing_type2(img, *args)
        elif op == 8:  # Level Slicing Type II (Inv.)
            args = [(round(args[0]), round(args[1])), round(args[2])]
            return enhance.intensity_level_slicing_type2_inv(img, *args)
        else:
            return img


class SpatialOperators:
    @staticmethod
    def get_blurring_operator_names():
        return (
            'mean',
            'gaussian',
            'median',
            'bilateral',
        )

    @staticmethod
    def blurring_args(op: str) -> dict:
        """args = {
        - "ksize": 1 | 2, # Number of ksize. Either 1 or 2.
        - "arguments": (`name`), # Argument names except ksize.
        - "ranges": (
            - (min, max), # Minimum and maximum of argument `name`.
            ),
        - "default": [
            - ksize, # Kernel size. int | tuple[int, int]
            - val, # Value of `name` float.
            ]
        }
        """
        if op == 'mean':
            return {
                'ksize': 2,
                'arguments': (),
                'ranges': (),
                'default': [(5, 5)],  # ksize Y, ksize X
            }
        elif op == 'gaussian':
            return {
                'ksize': 2,
                'arguments': ('sigma Y', 'sigma X'),
                'ranges': (
                    (0, float('inf')),
                    (0, float('inf')),
                ),
                'default': [(7, 7), 0, 0],  # ksize Y, ksize X, sigma Y, sigma X
            }
        elif op == 'median':
            return {
                'ksize': 1,
                'arguments': (),
                'ranges': (),
                'default': [7],  # ksize
            }
        elif op == 'bilateral':
            return {
                'ksize': 1,
                'arguments': ('sigma color', 'sigma space'),
                'ranges': (
                    (0, float('inf')),
                    (0, float('inf')),
                ),
                'default': [11, 50, 10],  # ksize, sigma color, sigma space
            }

    @staticmethod
    def blurring(
        img: IMG_8U, op: str, ksize: int | KER_SIZE, args: list[float]
    ) -> IMG_8U:
        if op == 'mean':
            blurred = mean_blur(
                img,
                ksize,
            )
        elif op == 'gaussian':
            blurred = gaussian_blur(img, ksize, *args)
        elif op == 'median':
            blurred = median_blur(img, ksize)
        elif op == 'bilateral':
            blurred = bilateral_blur(img, ksize, *args)
        else:
            return img
        return cv2.convertScaleAbs(blurred)

    @staticmethod
    def unsharp_masking(
        img: IMG_8U,
        op: str,
        amount: float,
        ksize: int | KER_SIZE,
        args: list[float],
    ) -> IMG_8U:
        if op in ('mean', 'gaussian'):
            return unsharp_masking(img, ksize, op, amount, *args)
        elif op == 'median':
            blurred = median_blur(img, ksize)
        elif op == 'bilateral':
            blurred = bilateral_blur(img, ksize, *args)
        else:
            return img
        mask = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0, dtype=cv2.CV_32F)
        return cv2.convertScaleAbs(mask)

    @staticmethod
    def get_border_names() -> tuple[str]:
        return ('Constant', 'Replicate', 'Reflect', 'Reflect_101')

    @staticmethod
    def get_gradient_operator_names() -> tuple[str]:
        return get_gradient_operator_names()

    @staticmethod
    def get_gradient_norm() -> tuple[str]:
        return ('Sup-Norm', 'Abs. Sum', 'Euclidean Norm', 'Abs. Mean')

    @staticmethod
    def get_overflow_deal_names() -> tuple[str]:
        return ('Clip', 'Normalization')

    @staticmethod
    def gradient(
        img: IMG_8U,
        op: str,
        bordertype: int,
        norm_type: int,
        to_grayscale: bool,
        overflow_type: int,
        overflow_val: float,
        threshold_val: int,
    ) -> IMG_8U:
        kernels = get_gradient_operator(op)
        edges = np.empty([len(kernels), *img.shape], dtype=np.float32)
        # Convolution
        if op != 'roberts':
            for i, kernel in enumerate(kernels):
                edges[i] = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=bordertype)
        else:
            for i, kernel in enumerate(kernels):
                edges[i] = cv2.filter2D(
                    img, cv2.CV_32F, kernel, anchor=(0, 0), borderType=bordertype
                )
        # Take norm
        output = gradient_norm(edges, norm_type)  # shape = img.shape
        if to_grayscale:
            output = rgb_to_gray(output)
        # Deal with overflow
        output = gradient_uint8_overflow(output, overflow_type, overflow_val, np.uint8)
        # Thresholding
        if threshold_val > 0:
            output = cv2.threshold(output, threshold_val, 255, cv2.THRESH_TOZERO)[1]
        return output

    @staticmethod
    def sharpening(
        img: IMG_8U,
        op: str,
        bordertype: int,
        norm_type: int,
        to_grayscale: bool,
        scaling_val: float,
        threshold_val: int,
    ) -> IMG_8U:
        if to_grayscale:  # Convert to grayscale
            temp = rgb_to_gray(img)
        else:
            temp = img
        kernels = get_gradient_operator(op)
        edges = np.empty([len(kernels), *temp.shape], dtype=np.float32)
        # Convolution
        if op != 'roberts':
            for i, kernel in enumerate(kernels):
                edges[i] = cv2.filter2D(temp, cv2.CV_32F, kernel, borderType=bordertype)
        else:
            for i, kernel in enumerate(kernels):
                edges[i] = cv2.filter2D(
                    temp, cv2.CV_32F, kernel, anchor=(0, 0), borderType=bordertype
                )
        # Take norm
        if len(kernels) > 1:
            if not norm_type:
                maximum = np.max(edges, axis=0)
                minimum = np.min(edges, axis=0)
                gradient = np.where(np.abs(maximum) > np.abs(minimum), maximum, minimum)
            elif norm_type == 1:
                gradient = np.sum(edges, axis=0, dtype=np.float32)
            elif norm_type == 2:
                maximum = np.max(edges, axis=0)
                minimum = np.min(edges, axis=0)
                sign = np.sign(
                    np.where(np.abs(maximum) > np.abs(minimum), maximum, minimum)
                )
                gradient = np.linalg.norm(edges, axis=0)
                gradient *= sign
            elif norm_type == 3:
                gradient = np.mean(edges, axis=0, dtype=np.float32)
        else:
            gradient = edges[0]
        # Boosting edge
        if scaling_val != 1:
            gradient *= scaling_val
        # Thresholding
        if threshold_val > 0:
            gradient[np.abs(gradient) <= threshold_val] = 0
        # Sharpening, f-f''
        if to_grayscale:  # Convert to grayscale
            output = np.empty_like(img, dtype=np.float32)
            for i in range(3):
                output[:, :, i] = np.subtract(img[:, :, i], gradient, dtype=np.float32)
        else:
            output = np.subtract(img, gradient, dtype=np.float32)
        return cv2.convertScaleAbs(output)

    # @staticmethod
    # def marr_hildreth(
    #         img: IMG_8U, ksize: int, sigma: float, bordertype: int,
    #         norm_type: int, to_grayscale: bool,
    #         overflow_type: int, overflow_val: float,
    #         threshold_val: int
    #     ) -> IMG_8U:
    #     # marr_hildreth:
    #     #   Step 1. Gaussian blur
    #     #   Step 2. Gradient.
    #     # Get kernels
    #     ksize, sigma, _ = check_gaussian_kernel_arg(ksize, sigma, sigma)
    #     ksize = (ksize[0]+2, ksize[1]+2) # Fill border
    #     gaussian = get_2d_gaussian_kernel(ksize, sigma, sigma)
    #     sobels = get_gradient_operator("sobel")
    #     # Convolution is commutative. Convolve gaussian kernel and gradient
    #     # kernel can reduce computation.
    #     kernels = []
    #     for i, ker in enumerate(sobels):
    #         temp = cv2.filter2D(gaussian, cv2.CV_32F, ker,
    #                             borderType=bordertype)[1:-1,1:-1]
    #         temp -= np.sum(temp) / temp.size
    #         kernels.append(temp)
    #     # Gradient
    #     edges = np.empty(
    #         [len(kernels), *img.shape], dtype=np.float32
    #     )
    #     img = img.astype(np.float32) # Convert to float32 will be faster.
    #     for i, ker in enumerate(kernels):
    #         edges[i] = cv2.filter2D(
    #             img, cv2.CV_32F, ker, borderType=bordertype
    #         )
    #     # Take norm
    #     output = gradient_norm(edges, norm_type)
    #     if to_grayscale: # Convert gradient to grayscale
    #         kernel = np.array((0.299,0.587,0.114), dtype=np.float32)
    #         output = np.dot(output, kernel)
    #     # Deal with overflow
    #     output = gradient_uint8_overflow(
    #         output, overflow_type, overflow_val, np.uint8
    #     )
    #     # Thresholding
    #     if (threshold_val > 0):
    #         output = cv2.threshold(
    #             output, threshold_val, 255, cv2.THRESH_TOZERO
    #         )[1]
    #     return output

    @staticmethod
    def canny(
        img: IMG_8U,
        ksize: int,
        sigma: float,
        threshold_val: int,
        threshold2_val: int,
        l2_gradient: bool,
    ) -> IMG_8U:
        # -Gaussian Blur
        if ksize > 1:
            img = gaussian_blur(img, ksize, sigma, sigma)
            img = cv2.convertScaleAbs(img)
        # -Canny
        return cv2.Canny(
            img, threshold_val, threshold2_val, apertureSize=3, L2gradient=l2_gradient
        )


class NoiseOperators:
    @staticmethod
    def get_noises_names() -> tuple[str]:
        return (
            'uniform',
            'gaussian',
            'rayleigh',
            'gamma',
            'exponential',
            'beta',
            'salt-and-pepper',
            # "single color"
        )

    def add_noise(img: Arr8U2D, noise: str, args: list[float]) -> Arr8U2D:
        if noise == 'uniform':
            args = [int(val) for val in args]
            if args[0] > args[1]:
                return img
            img = add_uniform_noise(img, *args)
            return cv2.convertScaleAbs(img)
        elif noise == 'gaussian':
            if args[1] == 0:
                return img
            img = add_gaussian_noise(img, *args)
            return cv2.convertScaleAbs(img)
        elif noise == 'rayleigh':
            if args[0] == 0:
                return img
            img = add_rayleigh_noise(img, *args)
            return cv2.convertScaleAbs(img)
        elif noise == 'gamma':
            if 0 in args:
                return img
            img = add_gamma_noise(img, *args)
            return cv2.convertScaleAbs(img)
        elif noise == 'exponential':
            if 0 in args:
                return img
            img = add_exponential_noise(img, *args)
            return cv2.convertScaleAbs(img)
        elif noise == 'salt-and-pepper':
            for val in args:
                if val < 0:
                    return img
            args = [val / 100 for val in args]
            s = sum(args)
            if s > 1:
                args = [val / s for val in args]
            elif s == 0:
                return img
            return add_salt_and_pepper_noise(img, *args)
        # elif noise == "single color":
        #     if args[0] == 0:
        #         return img
        #     args = [args[0]/100, round(args[1])]
        #     if args[1] < 0:
        #         args[1] = 0
        #     elif args[1] > 255:
        #         args[1] = 255
        #     return add_single_color_noise(img, *args)
        elif noise == 'beta':
            if args[0] == 0:
                return img
            args = [args[0] / 100, round(args[1])]
            if args[1] < 0:
                args[1] = 0
            elif args[1] > 255:
                args[1] = 255
            return add_beta_noise(img, *args)
        else:
            if img.dtype != np.uint8:
                return img.astype(dtype=np.uint8)
            else:
                return img


class SegmentationOperators:
    @staticmethod
    def get_thresholding_names() -> tuple[str]:
        return (
            'Binary',
            'Binary Inv',
            'Trunc',
            'To Zero',
            'To Zero Inv',
        )

    @staticmethod
    def get_auto_global_thresholding_names() -> int:
        return get_auto_thresholding_names()

    @staticmethod
    def auto_threshold(img: Arr8U2D, name: int) -> int:
        return auto_threshold(img, name)

    @staticmethod
    def get_local_thresholding_names():
        return (
            'Adaptive Mean',
            'Adaptive Gaussian',
            'Adaptive Median',
            'niBlack',
            'Sauvola',
            'Wolf',
            'Nick',
        )

    @staticmethod
    def local_thresholding(
        img: Arr8U2D, op: int, threshold_type: int, block_size: int, C: int
    ) -> Arr8U2D:
        if op < 2:  # Adaptive Mean/Gaussian
            return cv2.adaptiveThreshold(img, 255, op, threshold_type, block_size, C)
        if op == 2:  # Adaptive Median
            from seg import threshold_by_mask

            mask = np.subtract(cv2.medianBlur(img, block_size), C, dtype=np.float32)
            return threshold_by_mask(img, mask)
        if 3 <= op <= 6:  # niBlack, Sauvola, Wolf, Nick
            return cv2.ximgproc.niBlackThreshold(
                img, 255, threshold_type, block_size, C, binarizationMethod=op - 3
            )


class MorpologyOperators:
    @staticmethod
    def get_operaions_names():
        return (
            # OpenCV
            'Dilation',
            'Erosion',
            'Opening',
            'Closing',
            'Gradient',
            'Top Hat',
            'Black Hat',
            'Hit-or-Miss',
            # 1 SE
            'Pruning',
            'Connect Components',
            'Thinning',
            'Thickening',
            'Boundary Extraction',
            'Region Filling',
            'Border Clean',
            'Skelton',
            # 2 SE
            'Opening by Rec.',
            'Closing by Rec.',
        )

    @staticmethod
    def basic_operation(
        img: IMG_8U,
        op: int,
        SE: np.ndarray,
        iter_: int,
        bordertype: int,
        SE2: np.ndarray | None,
    ) -> IMG_8U:
        if op < 8:
            return cv2.morphologyEx(
                img, op, SE, iterations=iter_, borderType=bordertype
            )
        elif op == 8:
            from morphology import pruning

            return pruning(img, iter_)
        elif op == 9:
            from morphology import draw_connect_components

            return draw_connect_components(img, SE, iter_)
        elif op == 10:
            from morphology import thinning

            return thinning(img, iter_)
        elif op == 11:
            from morphology import thickening

            return thickening(img, iter_)
        elif op == 12:
            from morphology import boundary_extraction

            return boundary_extraction(img, SE)
        elif op == 13:
            from morphology import auto_region_filling

            return auto_region_filling(img, SE)
        elif op == 14:
            from morphology import border_clean

            return border_clean(img, SE)
        elif op == 15:
            from morphology import skelton

            return skelton(img, SE)[0]
        elif op == 16:
            from morphology import opening_by_rec

            return opening_by_rec(img, SE, SE2, iter_)
        elif op == 17:
            from morphology import closing_by_rec

            return closing_by_rec(img, SE, SE2, iter_)


IMG = TypeVar('Image')


class Image(Generic[IMG], BasicOperators):
    support_color_formats = ('RGB888', 'Grayscale8')

    support_file_extensions = (
        '*.jpeg',
        '*.jpg',
        '*.jpe',  # JPEG
        '*.jp2',  # JPEG 2000
        '*.png',  # PNG (Portable Network Graphics)
        # "*.tiff", "*.tif", # TIFF
        '*.webp',  # WebP
        # "*.avif", # AVIF
        # "*.pbm", "*.pgm", "*.ppm", "*.pxm", "*.pnm", # Portable image format
        # "*.pfm", # PFM
        # "*.hdr", "*.pic" # Radiance HDR
    )

    support_database = (
        'astronaut_512',
        'astronaut_1024x821',
        'astronaut_3000x2406',
        'brick_512',
        'bricks_1024_color',
        'bricks_1024_gray',
        'bricks_2048_color',
        'bricks_2048_gray',
        'bricks_4096_color',
        'bricks_4096_gray',
        'bricks_8192_color',
        'bricks_8192_gray',
        'camera',
        'cell',
        'chelsea',
        'chessboard',
        'clock_motion',
        'coffee',
        'coins',
        'colorwheel',
        'grass',
        'gravel',
        'horse',
        'hubble_deep_field',
        'immunohistochemistry',
        'microaneurysms',
        'moon',
        'motorcycle_left',
        'motorcycle_right',
        'page',
        'retina',
        'rocket',
        'shepp-logan_phantom',
        'simple',
        'text',
    )

    history_limit: int = 20  # Limit of history length of Image object

    def __init__(
        self, img: IMG_8U, name: str, file_extension: str = '.png', path: str = None
    ):
        if not isinstance(img, np.ndarray):
            img = np.array(img, dtype=np.uint8)
        self.__original = img  # Original image
        # 影像現在的尺寸(self.shape)、顏色格式(self.__color)、資料(self.img)
        dim = self.__channels = img.ndim
        if dim == 3:  # 彩色圖片
            self.__color = 'RGB888'
        elif dim == 2:  # 灰階圖片
            self.__color = 'Grayscale8'
        self.name = name
        self.file_extension = file_extension
        if isinstance(path, str) and path[-1] not in ('\\', '/'):
            path = ''.join([path, '\\'])
        self.path = path
        # Editing history
        self.__history = [
            img,
        ]
        self.__history_index = 0  # 現在的影像為第幾張, 0,1,2,..., Limit-1
        # 亮度（灰階）統計資訊
        self.__brightness_statistics_update()

    def read_image(path: str) -> np.ndarray:
        img = cv2.imdecode(
            # cv2 does not support utf-8 character in path.
            np.fromfile(path, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        if img.ndim == 3:
            img = img[:, :, ::-1]  # BGR2RGB
        return img

    def save_image(self, option: dict | None = None) -> None:
        path = ''.join([self.path, self.name, self.file_extension])
        if option is not None:
            buf = cv2.imencode(self.file_extension, self.image, option)[1]
        else:
            buf = cv2.imencode(self.file_extension, self.image)[1]
        buf.tofile(path)

    def __check_history_length(self):
        if len(self.__history) > Image.history_limit:
            self.__history = self.__history[1:]
            self.__history_index = Image.history_limit - 1

    def __update_dim(self):
        img = self.image
        if img.ndim == 3:
            self.__channels = 3
            self.__color = 'RGB888'
        else:
            self.__channels = 2
            self.__color = 'Grayscale8'

    # 返回self.currentImg
    @property
    def image(self) -> IMG_8U:
        return self.__history[self.__history_index]

    @image.setter
    def image(self, new_img: IMG_8U):
        if not isinstance(new_img, np.ndarray):
            try:
                new_img = np.array(new_img, dtype=np.uint8)
            except:  # noqa
                print('img should be type NDArray[np.uint8],', f'not {type(new_img)}')
                raise ValueError
        # Update history and check length
        self.__history_index += 1
        if self.__history_index < len(self.__history):
            # index後還有圖片
            del self.__history[self.__history_index :]
        self.__history.insert(self.__history_index, new_img)
        self.__check_history_length()
        # Update attribute
        self.__update_dim()
        # Update statistics
        self.__brightness_statistics_update()
        #
        from gc import collect

        collect(2)

    # 復原, Ctrl+Z
    def undo(self) -> None:
        if self.__history_index > 0:
            self.__history_index -= 1
            self.__update_dim()
            self.__brightness_statistics_update()

    # 取消復原, Ctrl+Y
    def redo(self) -> None:
        if self.__history_index < len(self.__history) - 1:
            self.__history_index += 1
            self.__update_dim()
            self.__brightness_statistics_update()

    # 重置
    def reset(self) -> None:
        self.image = self.__original  # 原始影像

    # 轉為灰階
    def to_grayscale(self) -> None:
        if self.__channels == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    # 轉為彩色
    def to_color(self) -> None:
        if self.__channels == 2:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)

    def equalize_hist(self):
        """Applying histogram equalization to img."""
        # Check chanels
        if self.__channels == 3:  # 彩色圖片
            hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
            hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
            self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        elif self.__channels == 2:  # 灰階圖片
            self.image = cv2.equalizeHist(self.image)

    def auto_gamma_correction_PB_PZ(self):
        # Check chanels
        self.image = enhance.auto_gamma_correction_PB_PZ(self.image)

    def is_grayscale(img: IMG_8U) -> bool:
        if img.ndim == 2:
            return True
        a = cv2.absdiff(img[:, :, 0], img[:, :, 1])
        b = cv2.absdiff(img[:, :, 0], img[:, :, 2])
        return (cv2.countNonZero(a) + cv2.countNonZero(b)) == 0

    # Statistics about Brightness
    @property
    def brightness_statistics(
        self,
    ) -> tuple[Mean, Var, Std, Skewness, Kurtosis, Max, Min, Median]:
        """Return the mean, var, std, max, min of brightness of the image."""
        return self.__brightness

    def __brightness_statistics_update(self) -> None:
        # Check chanels
        if self.__channels == 3:  # Color
            img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:  # Grayscale
            img = self.image
        # Update
        self.__brightness = Image.get_statistics(img)
