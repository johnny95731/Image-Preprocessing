from typing import Optional

import numpy as np
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QToolTip, QWidget

from interface import MplCanvas, BasicDialog, LabelThread
from image_object import (
    Image, SpatialOperators, SegmentationOperators, NoiseOperators,
    MorpologyOperators
)



"""
Dialogs for functions in menu: Spatial (see mainwindow.py,
self.__create_main_menubar).
"""


# Blur
class BlurDialog(BasicDialog):
    """A dialog for bluring image."""
    
    def __init__(self, parent: QWidget, mode: int = 0):
        self.mode = mode # 0=blur, 1=unsharp masking
        if not mode:
            super().__init__(parent, "Bluring")
            self.main_work = self.bluring
        else:
            super().__init__(parent, "Unsharp Masking")
            self.main_work = self.unsharp_masking
        
        # Main region
        self.__create_ksize_region()
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        self.connect_value_changed(is_connect=False)
        
        self.blur_op = "mean" # Default blur operator.
        self.blur_info = {}
        self.blur_info[self.blur_op] = SpatialOperators.blurring_args(self.blur_op)
        self.__update_args_enable(self.blur_op)
        self.update_view()
        self.show()
        
    def __create_ksize_region(self):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: ksize
        title = QtWidgets.QLabel("ksize (kernel size)", region)
        title.setFont(self.fonts["title"])
        # -Toggle sizes
        self.button["size"] = QtWidgets.QCheckBox("Same ksize", region)
        layout_region.addRow(title, self.button["size"])
        self.button["size"].toggled.connect(self.__same_ksize)
        # -Labels and spin boxes.
        img_shape = self.original_image.shape[:2] # (height, width)
        for index, direction in enumerate(["height", "width"]):
            self.widget[direction] = QtWidgets.QLabel(direction, region)
            self.widget[direction].setFont(self.fonts["default"])
            spin_box = self.spin_box[direction] = QtWidgets.QSpinBox(region)
            spin_box.setRange(1, img_shape[1-index])
            spin_box.setValue(3)
            spin_box.setSingleStep(2)
            layout_region.addRow(self.widget[direction], spin_box)
        
    def __create_args_region(self):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Arguments
        title = QtWidgets.QLabel("Arguments", region)
        title.setFont(self.fonts["title"])
        # -Combo box.
        blur_list = SpatialOperators.get_blurring_operator_names()
        self.combo_box["arg"] = QtWidgets.QComboBox()
        self.combo_box["arg"].addItems(blur_list)
        self.combo_box["arg"].currentTextChanged.connect(self.blur_op_changed)
        layout_region.addRow(title, self.combo_box["arg"])
        
        # -Labels and double spin boxes.
        q = 2 if not self.mode else 3
        for j in range(q):
            self.widget[f"arg{j}_label"] = QtWidgets.QLabel()
            self.widget[f"arg{j}_label"].setFont(self.fonts["default"])
            self.spin_box[f"arg{j}"] = QtWidgets.QDoubleSpinBox()
            self.spin_box[f"arg{j}"].setMinimumWidth(80)
            layout_region.addRow(self.widget[f"arg{j}_label"],
                                 self.spin_box[f"arg{j}"])
        if self.mode:
            self.widget["arg2_label"].setText("Amount")
            self.spin_box["arg2"].setRange(-float("inf"), float("inf"))
            self.spin_box["arg2"].setValue(1)
    
    def connect_value_changed(self, is_connect: bool):
        """update_view may be called many times when operator changed. Hence,
        the signals should be disconneted and re-connect at the beginning and
        at the end of `__update_args_enable()`, respectively.
        """
        q = 2 if not self.mode else 3
        if is_connect:
            self.spin_box["height"].valueChanged.disconnect()
            self.spin_box["width"].valueChanged.disconnect()
            for j in range(q):
                self.spin_box[f"arg{j}"].valueChanged.disconnect()
        else:
            self.spin_box["height"].valueChanged.connect(self.check_ker_height)
            self.spin_box["width"].valueChanged.connect(self.check_ker_width)
            for j in range(q):
                self.spin_box[f"arg{j}"].valueChanged.connect(self.update_view)
            if self.mode:
                self.spin_box["arg2"].valueChanged.connect(self.update_view)
    
    # Functions
    def check_ker_height(self, val: int):
        if not val % 2: # kernel size must be odd.
            val -= 1
            self.spin_box["height"].setValue(val)
            return # emit valueChange again
        if self.button["size"].isChecked():
            self.spin_box["width"].setValue(val)
        self.update_view()
    
    def check_ker_width(self, val: int):
        if not self.button["size"].isChecked():
            if not (val % 2):# kernel size must be odd.
                self.spin_box["width"].setValue(val-1)
            # If button["size"].isChecked(), spin_box["width"] can't be changed
            # by user.
            self.update_view()
    
    def __same_ksize(self, checked: bool):
        if checked:
            val = min(self.spin_box["height"].value(),
                      self.spin_box["width"].value())
            self.widget["height"].setText("size")
            self.spin_box["height"].setValue(val)
            self.widget["width"].hide()
            self.spin_box["width"].hide()
            self.update_view()
        else:
            self.widget["height"].setText("height")
            self.widget["width"].show()
            self.spin_box["width"].show()
    
    def bluring(self):
        # Get ksize
        if self.blur_op in ("median", "bilateral"):
            ksize = self.spin_box["height"].value()
        else:
            ksize = [
                self.spin_box[direction].value()
                    for direction in ("height", "width")
            ]
        # Get args
        args = []
        if self.blur_op not in ("mean", "median"):
            for j in range(2):
                args.append(self.spin_box[f"arg{j}"].value())
        #
        return SpatialOperators.blurring(
            self.original_image, self.blur_op, ksize, args
        )
    
    def unsharp_masking(self):
        # Get ksize
        if self.blur_op in ("median", "bilateral"):
            ksize = self.spin_box["height"].value()
        else:
            ksize = [
                self.spin_box[direction].value()
                    for direction in ("height", "width")
            ]
        # Get args
        args = []
        if self.blur_op not in ("mean", "median"):
            for j in range(2):
                args.append(self.spin_box[f"arg{j}"].value())
        amount = self.spin_box["arg2"].value()
        #
        return SpatialOperators.unsharp_masking(
            self.original_image, self.blur_op, amount, ksize, args
        )
    
    def __update_args_enable(self, blur):
        self.connect_value_changed(is_connect=True)
        info = self.blur_info[blur]["default"]
        arg_names = self.blur_info[blur]["arguments"]
        # Change ksize label and spin box.
        if blur in ("median", "bilateral"):
            self.widget["height"].setText("diameter")
            self.spin_box["height"].setValue(info[0])
            self.widget["width"].hide()
            self.spin_box["width"].hide()
        else:
            self.widget["height"].setText("height")
            self.spin_box["height"].setValue(info[0][0])
            self.widget["width"].show()
            self.spin_box["width"].show()
            self.spin_box["width"].setValue(info[0][1])
        # Change arguments label and spin box.
        if not arg_names:
            for j in range(2):
                self.widget[f"arg{j}_label"].clear()
                self.spin_box[f"arg{j}"].hide()
        else:
            arg_ranges = self.blur_info[blur]["ranges"]
            for j, name in enumerate(arg_names):
                self.widget[f"arg{j}_label"].setText(name)
                self.spin_box[f"arg{j}"].show()
                self.spin_box[f"arg{j}"].setRange(*arg_ranges[j])
                self.spin_box[f"arg{j}"].setValue(info[1+j])
        self.connect_value_changed(is_connect=False)
        
    def blur_op_changed(self, blur: Optional[str] = None):
        # Record past value
        info = self.blur_info[self.blur_op]["default"]
        arg_names = self.blur_info[self.blur_op]["arguments"]
        if self.blur_op in ("median", "bilateral"):
            info[0] = self.spin_box["width"].value()
            for j, _ in enumerate(arg_names):
                info[j+1] = self.spin_box[f"arg{j}"].value()
        else:
            info[0] = (
                self.spin_box["height"].value(),
                self.spin_box["width"].value()
            )
            for j, _ in enumerate(arg_names):
                info[j+1] = self.spin_box[f"arg{j}"].value()
        
        self.blur_op = blur
        if blur not in self.blur_info.keys(): # operator never be selected.
            self.blur_info[blur] = SpatialOperators.blurring_args(blur)
        # Update label and spin box.
        self.__update_args_enable(blur)
        self.update_view()
        
    def show_effect(self, checked):
        super().show_effect(checked)
        q = 3 if self.mode else 2
        self.combo_box["arg"].setEnabled(checked)
        for j in range(q):
            self.widget[f"arg{j}_label"].setEnabled(checked)
            self.spin_box[f"arg{j}"].setEnabled(checked)



# Edge Detection
class CannyDialog(BasicDialog):
    """A dialog for applying Canny edge detection to image."""
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Canny")
            
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Arguments
        title = QtWidgets.QLabel("Blur Args", region)
        title.setFont(self.fonts["title"])
        layout_region.addRow(title)
        # -Ksize
        label = QtWidgets.QLabel("Ksize", region)
        label.setFont(self.fonts["default"])
        self.spin_box["arg"] = QtWidgets.QSpinBox()
        self.spin_box["arg"].setRange(1, 255)
        self.spin_box["arg"].setSingleStep(2)
        self.spin_box["arg"].setValue(5)
        self.spin_box["arg"].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box["arg"])
        # -Sigma
        label = QtWidgets.QLabel("Sigma", region)
        label.setFont(self.fonts["default"])
        self.spin_box["arg2"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["arg2"].setRange(0, float("inf"))
        self.spin_box["arg2"].setSingleStep(0.5)
        self.spin_box["arg2"].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box["arg2"])
        # Canny Argument
        title = QtWidgets.QLabel("Canny Args", region)
        title.setFont(self.fonts["title"])
        layout_region.addRow(title)
        # -Threshold
        label2 = QtWidgets.QLabel("Threshold", region)
        label2.setFont(self.fonts["title"])
        self.spin_box["threshold"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["threshold"].setRange(0, float("inf"))
        self.spin_box["threshold"].setValue(60)
        self.spin_box["threshold"].valueChanged.connect(self.update_view)
        layout_region.addRow(label2, self.spin_box["threshold"])
        # -Threshold 2
        label2 = QtWidgets.QLabel("Threshold 2", region)
        label2.setFont(self.fonts["title"])
        self.spin_box["threshold2"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["threshold2"].setRange(0, float("inf"))
        self.spin_box["threshold2"].setValue(100)
        self.spin_box["threshold2"].valueChanged.connect(self.update_view)
        layout_region.addRow(label2, self.spin_box["threshold2"])
        # -Aperture Size
        label2 = QtWidgets.QLabel("Aperture Size", region)
        label2.setFont(self.fonts["title"])
        self.spin_box["aperture_size"] = QtWidgets.QSpinBox()
        self.spin_box["aperture_size"].setRange(3, 7)
        self.spin_box["aperture_size"].setSingleStep(2)
        self.spin_box["aperture_size"].valueChanged.connect(self.update_view)
        layout_region.addRow(label2, self.spin_box["aperture_size"])
        # -L2gradient
        self.button["L2"] = QtWidgets.QCheckBox("L2 Gradient")
        self.button["L2"].clicked.connect(self.update_view)
        self.button["L2"].setChecked(False)
        layout_region.addRow(self.button["L2"])
    
    # Functions
    def main_work(self):
        # Get args
        # -Blur Arg
        ksize = self.spin_box["arg"].value()
        if not ksize % 2:
            ksize -= 1
        sigma = self.spin_box["arg2"].value()
        # -Canny Arg
        threshold_val = self.spin_box["threshold"].value()
        threshold2_val = self.spin_box["threshold2"].value()
        aperture_size = self.spin_box["aperture_size"].value()
        if not aperture_size % 2:
            aperture_size -= 1
        l2_gradient = self.button["L2"].isChecked()
        return SpatialOperators.canny(
            self.original_image, ksize, sigma,
            threshold_val, threshold2_val, l2_gradient)


class GradientDialog(BasicDialog):
    """A dialog for applying gradient operator to image."""
    
    def __init__(self, parent: QWidget, mode: int = 0):
        self.mode = mode # mode: 0=Gradient, 1=Sharpening
        if not mode:
            super().__init__(parent, "Gradient")
            self.main_work = self.gradient
        else:
            super().__init__(parent, "Sharpening")
            self.main_work = self.sharpening

        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        # -Title: Arguments
        title = QtWidgets.QLabel("Gradient Operators", region)
        title.setFont(self.fonts["title"])
        # -Combo box.
        grad_list = SpatialOperators.get_gradient_operator_names()
        self.combo_box["arg"] = QtWidgets.QComboBox()
        self.combo_box["arg"].addItems(grad_list)
        self.combo_box["arg"].setCurrentIndex(grad_list.index("sobel"))
        self.combo_box["arg"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(title, self.combo_box["arg"])
        # -Border Type
        label = QtWidgets.QLabel("Border Type", region)
        label.setFont(self.fonts["default"])
        border_list = SpatialOperators.get_border_names()
        self.combo_box["border"] = QtWidgets.QComboBox()
        self.combo_box["border"].addItems(border_list)
        self.combo_box["border"].setCurrentIndex(
            border_list.index("Reflect_101")
        )
        self.combo_box["border"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["border"])
        # -Norm
        label = QtWidgets.QLabel("Norm", region)
        label.setFont(self.fonts["default"])
        norm_list = SpatialOperators.get_gradient_norm()
        self.combo_box["norm"] = QtWidgets.QComboBox()
        self.combo_box["norm"].addItems(norm_list)
        self.combo_box["norm"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["norm"])
        # -Overflow: handling option
        if not self.mode:
            label = QtWidgets.QLabel("Overflow", region)
            label.setFont(self.fonts["default"])
            overflow_list = SpatialOperators.get_overflow_deal_names()
            self.combo_box["overflow"] = QtWidgets.QComboBox()
            self.combo_box["overflow"].addItems(overflow_list)
            self.combo_box["overflow"].setCurrentIndex(1)
            self.combo_box["overflow"].currentIndexChanged.connect(
                self.overflow_changed)
            layout_region.addRow(label, self.combo_box["overflow"])
        # -Overflow: arg
        text = "Coeff" if self.mode else "Coeff (%)"
        self.widget["overflow_label"] = QtWidgets.QLabel(text, region)
        self.widget["overflow_label"].setFont(self.fonts["default"])
        self.spin_box["overflow"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["overflow"].setDecimals(3)
        self.spin_box["overflow"].setRange(0.001, 100)
        if not self.mode:
            self.spin_box["overflow"].setValue(100)
        else:
            self.spin_box["overflow"].setValue(0.8)
            self.spin_box["overflow"].setSingleStep(0.05)
        self.spin_box["overflow"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["overflow_label"],
                             self.spin_box["overflow"])
        # -Threshold
        label2 = QtWidgets.QLabel("Threshold", region)
        label2.setFont(self.fonts["title"])
        self.spin_box["threshold"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["threshold"].setRange(0, float("inf"))
        self.spin_box["threshold"].valueChanged.connect(self.update_view)
        layout_region.addRow(label2, self.spin_box["threshold"])
        # -Output Grascale
        self.button["grayscale"] = QtWidgets.QCheckBox("Grayscale")
        self.button["grayscale"].clicked.connect(self.update_view)
        self.button["grayscale"].setChecked(True)
        if not self.is_color:
            self.button["grayscale"].hide()
        layout_region.addRow(self.button["grayscale"])
        
    # Functions
    def gradient(self):
        # Get args
        op = self.combo_box["arg"].currentText()
        bordertype = self.combo_box["border"].currentIndex()
        # reflex101 is index 3 but cv2.BORDER_REFLECT101 is 4
        if bordertype == 3: 
            bordertype = 4
        norm_type = self.combo_box["norm"].currentIndex()
        overflow_type = self.combo_box["overflow"].currentIndex()
        overflow_val = self.spin_box["overflow"].value()
        threshold_val = self.spin_box["threshold"].value()
        #
        to_grayscale = self.is_color and self.button["grayscale"].isChecked()
        return SpatialOperators.gradient(
            self.original_image, op, bordertype, norm_type,
            to_grayscale, overflow_type, overflow_val, threshold_val
        )
    
    def sharpening(self):
        # Get args
        op = self.combo_box["arg"].currentText()
        bordertype = self.combo_box["border"].currentIndex()
        # reflex101 is index 3 but cv2.BORDER_REFLECT101 is 4
        if bordertype == 3: 
            bordertype = 4
        scaling_val = self.spin_box["overflow"].value()
        norm_type = self.combo_box["norm"].currentIndex()
        threshold_val = self.spin_box["threshold"].value()
        # 
        to_grayscale = self.is_color and self.button["grayscale"].isChecked()
        return SpatialOperators.sharpening(
            self.original_image, op, bordertype, norm_type,
            to_grayscale, scaling_val, threshold_val
        )
    
    def overflow_changed(self, index: int):
        if index:
            self.widget["overflow_label"].show()
            self.spin_box["overflow"].show()
        else:
            self.widget["overflow_label"].hide()
            self.spin_box["overflow"].hide()
        self.update_view()


class MarrHildrethDialog(BasicDialog):
    """A dialog for applying Marr-Hidreth edge detection to image."""
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Marr-Hidreth")
        
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Arguments
        title = QtWidgets.QLabel("Blur Args", region)
        title.setFont(self.fonts["title"])
        layout_region.addRow(title)
        # -Ksize
        label = QtWidgets.QLabel("Ksize", region)
        label.setFont(self.fonts["default"])
        self.spin_box["arg"] = QtWidgets.QSpinBox()
        self.spin_box["arg"].setRange(3, 255)
        self.spin_box["arg"].setSingleStep(2)
        self.spin_box["arg"].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box["arg"])
        # -Sigma
        label = QtWidgets.QLabel("Sigma", region)
        label.setFont(self.fonts["default"])
        self.spin_box["arg2"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["arg2"].setRange(0, float("inf"))
        self.spin_box["arg2"].setSingleStep(0.5)
        self.spin_box["arg2"].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box["arg2"])
        # -Border Type
        label = QtWidgets.QLabel("Border Type", region)
        label.setFont(self.fonts["default"])
        border_list = SpatialOperators.get_border_names()
        self.combo_box["border"] = QtWidgets.QComboBox()
        self.combo_box["border"].addItems(border_list)
        self.combo_box["border"].setCurrentIndex(
            border_list.index("Reflect_101")
        )
        self.combo_box["border"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["border"])
        # -Norm
        label = QtWidgets.QLabel("Norm", region)
        label.setFont(self.fonts["default"])
        norm_list = ("Abs. Max", "Sum", "Mean")
        self.combo_box["norm"] = QtWidgets.QComboBox()
        self.combo_box["norm"].addItems(norm_list)
        self.combo_box["norm"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["norm"])
        # -Deal overflow, combo Box
        label = QtWidgets.QLabel("Overflow", region)
        label.setFont(self.fonts["default"])
        overflow_list = SpatialOperators.get_overflow_deal_names()
        self.combo_box["overflow"] = QtWidgets.QComboBox()
        self.combo_box["overflow"].addItems(overflow_list)
        self.combo_box["overflow"].setCurrentIndex(1)
        self.combo_box["overflow"].currentIndexChanged.connect(
            self.overflow_changed)
        layout_region.addRow(label, self.combo_box["overflow"])
        # -Overflow arg
        self.widget["overflow_label"] = QtWidgets.QLabel("Coeff (%)", region)
        self.widget["overflow_label"].setFont(self.fonts["default"])
        self.spin_box["overflow"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["overflow"].setRange(0.001, 100)
        self.spin_box["overflow"].setDecimals(3)
        self.spin_box["overflow"].setValue(90)
        self.spin_box["overflow"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["overflow_label"],
                             self.spin_box["overflow"])
        # -Threshold
        label2 = QtWidgets.QLabel("Threshold", region)
        label2.setFont(self.fonts["title"])
        self.spin_box["threshold"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["threshold"].setRange(0, float("inf"))
        self.spin_box["threshold"].valueChanged.connect(self.update_view)
        layout_region.addRow(label2, self.spin_box["threshold"])
        # -Output Grascale
        self.button["grayscale"] = QtWidgets.QCheckBox("Grayscale")
        self.button["grayscale"].clicked.connect(self.update_view)
        self.button["grayscale"].setChecked(True)
        if not self.is_color:
            self.button["grayscale"].hide()
        layout_region.addRow(self.button["grayscale"])
    
    # Functions
    def main_work(self):
        # Get args
        ksize = self.spin_box["arg"].value()
        if not ksize % 2:
            ksize -= 1
        sigma = self.spin_box["arg2"].value()
        bordertype = self.combo_box["border"].currentIndex()
        # reflex101 is index 3 but cv2.BORDER_REFLECT101 is 4
        if bordertype == 3: 
            bordertype = 4
        norm_type = self.combo_box["norm"].currentIndex()
        overflow_type = self.combo_box["overflow"].currentIndex()
        overflow_val = self.spin_box["overflow"].value()
        threshold_val = self.spin_box["threshold"].value()
        #
        to_grayscale = self.is_color and self.button["grayscale"].isChecked()
        return SpatialOperators.marr_hildreth(
            self.original_image, ksize, sigma, bordertype,
            norm_type, to_grayscale, overflow_type, overflow_val,
            threshold_val)
    
    def overflow_changed(self, index: int):
        if index:
            self.widget["overflow_label"].show()
            self.spin_box["overflow"].show()
        else:
            self.widget["overflow_label"].hide()
            self.spin_box["overflow"].hide()
        self.update_view()



# Noise
class NoiseDialog(BasicDialog):
    """
    A dialog for creating a new image(with noise or not) or for adding
    noise to image.
    """
    
    noise_args = {
        "none": {
            "arguments": (),
            "ranges": (),
            "default": (),
            "pdf": None,
        },
        "uniform": {
            "arguments": (r"a", r"b"),
            "ranges": (
                (-float("inf"), float("inf")),
                (-float("inf"), float("inf"))
            ),
            "default": [[-1,1] for _ in range(3)],
            "pdf": r"$\displaystyle\frac{1}{b-a},\ a < x < b.$",
        },
        "gaussian": {
            "arguments": (r"$\mu$(center)", r"$\sigma$"),
            "ranges": ((-float("inf"), float("inf")), (0, float("inf"))),
            "default": [[0,1] for _ in range(3)],
            "pdf": "".join((
                r"$\displaystyle\frac{1}{\sqrt{2\pi}\/\sigma}$",
                r"exp",
                r"$\displaystyle\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}$"
            )),
        },
        "rayleigh": {
            "arguments": (r"$\sigma$", ),
            "ranges": ((0, float("inf")), ),
            "default": [[5] for _ in range(3)],
            "pdf": "".join((
                r"$\displaystyle\frac{x}{\sigma^2}$",
                r"exp$\displaystyle\left\{-\frac{x^2}{2 \sigma^2}\right\},\ $",
                r"$x,\ \sigma > 0$", 
            )),
        },
        "gamma": {
            "arguments": (r"$k$(shape)", r"$\theta$(scale)"),
            "ranges": ((0, float("inf")), (-float("inf"), float("inf"))),
            "default": [[1,1] for _ in range(3)],
            "pdf": "".join((
                r"$\displaystyle\frac{x^{k-1}}{\Gamma(k) * \theta^k}$",
                r"exp$\displaystyle\left\{-\frac{x}{\theta}\right\},\ $",
                r"$x,\ k, \theta > 0$", 
            )),
        },
        "exponential": {
            "arguments": (r"$\beta$(scale)", ),
            "ranges": ((-float("inf"), float("inf")), ),
            "default": [[1] for _ in range(3)],
            "pdf": "".join((
                r"$\displaystyle\frac{1}{\beta}$",
                r"exp$\displaystyle\left\{-\frac{x}{\beta}\right\},\ $",
                r"$\beta > 0,\ x\geq0$",
            )),
        },
        "salt-and-pepper": {
            "arguments": (r"salt(\%)", r"pepper(\%)"),
            "ranges": ((0, 100), (0, 100)),
            "default": [[.5, .5] for _ in range(3)],
            "pdf": None,
        },
        "single color": {
            "arguments": (r"prob(\%)", "color"),
            "ranges": ((0, 100), (0, 255)),
            "default": [[.5, 0] for _ in range(3)],
            "pdf": None,
        },
        "beta": {
            "arguments": ("a", "b", "max"),
            "ranges": (
                (0, float("inf")), (-float("inf"), float("inf")),
                (0, 255)
            ),
            "default": [[1, 1, 255] for _ in range(3)],
            "pdf": None,
        },
    }
    
    def __init__(self, parent: QWidget = None, mode: str="new"):
        # mode: new=new image, add=add noise
        if mode not in ("new", "add"):
            print(f"mode must be \"new\" or \"add\", not {mode}")
            raise ValueError
        self.__mode: str = mode
        # Main Dialog
        title = "Create Image" if mode == "new" else "Add Noise"
        super().__init__(parent, title)
        
        # Attributes about image
        if (mode == "add" and not self.is_color):
            self.original_image = self.original_image[:,:,np.newaxis]
        
        # Main region layout
        if self.__mode == "new":
            self.__create_new_region()
        self.__create_args_region()
        self.__create_info_region()
        self.central_layout.addStretch(3)
        self.create_basic_buttons()
        self.connect_value_changed(is_connect=False)
        
        self.noise_changed(self.combo_box["arg"].currentText())
        self.show()
    
    def __create_new_region(self,):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)
        layout_region.setVerticalSpacing(10)
        row_index = 0
        
        # File name and format
        layout_name = QtWidgets.QGridLayout()
        layout_region.addLayout(layout_name, row_index, 0, 1, 3)
        row_index += 1
        # -Title: File Name
        title = QtWidgets.QLabel("Filename", region)
        layout_region.addWidget(title, 0, 0)
        title.setFont(self.fonts["title"])
        # -Default filename
        default_name = self.parentWidget().filename_repeat("New File.jpeg")
        # -Line Edit
        self.widget["name"] = QtWidgets.QLineEdit(default_name, region)
        layout_name.addWidget(self.widget["name"], 1, 0, 1, 3)
        # --ToolTip
        self.widget["name"].setToolTip(BasicDialog.filename_tooltip())
        # --Block some special word.
        validator = QRegularExpressionValidator(
            BasicDialog.filename_reg_exp(), self.widget["name"])
        self.widget["name"].setValidator(validator)
        # --Default formats: .jpeg
        self.combo_box["file_format"] = QtWidgets.QComboBox(region)
        layout_name.addWidget(self.combo_box["file_format"], 1, 3)
        self.combo_box["file_format"].addItems(Image.support_file_extensions)
        
        # Image size
        layout_size = QtWidgets.QFormLayout()
        layout_region.addLayout(layout_size, row_index, 0, 1, 2)
        layout_size.setVerticalSpacing(12)
        row_index += 1
        # -Title: Size
        title = QtWidgets.QLabel("Size", region)
        title.setFont(self.fonts["title"])
        # -Toggle sizes
        self.button["size"] = QtWidgets.QCheckBox("Fixed Ratio", region)
        layout_size.addRow(title, self.button["size"])
        self.button["size"].toggled.connect(self.__fix_size_ratio)
        self.button["size"].setChecked(False)
        # -Labels and spin boxes.
        # --Qt use int32 and size should be positive. 
        size_range = (1, 10000)
        for direction in ("height", "width"):
            label = QtWidgets.QLabel(direction, region)
            label.setFont(self.fonts["default"])
            self.spin_box[direction] = QtWidgets.QSpinBox(region)
            self.spin_box[direction].setRange(*size_range)
            self.spin_box[direction].setValue(500)
            layout_size.addRow(label, self.spin_box[direction])
        self.spin_box["height"].valueChanged.connect(self.height_changed)
        self.spin_box["width"].valueChanged.connect(self.width_changed)
        
        # Color dimensions
        layout_color = QtWidgets.QGridLayout()
        layout_region.addLayout(layout_color, row_index, 0, 1, 1)
        row_index += 1
        # -Title: Color
        title = QtWidgets.QLabel("Color", region)
        layout_color.addWidget(title, 0, 0)
        title.setFont(self.fonts["title"])
        # -options
        self.combo_box["color"] = QtWidgets.QComboBox(region)
        layout_color.addWidget(self.combo_box["color"], 0, 1, 1, 2)
        self.combo_box["color"].addItems(Image.support_color_formats)
        self.combo_box["color"].currentIndexChanged.connect(self.color_changed)
        self.channels = ("Red", "Green", "Blue")
        
        # Background color
        layout_bg_color = QtWidgets.QGridLayout()
        layout_region.addLayout(layout_bg_color, row_index, 0, 1, 3)
        layout_bg_color.setHorizontalSpacing(6)
        row_index += 1
        # -Labels and spin boxes.
        for i, c in enumerate(self.channels):
            # chanel
            self.widget[f"bg_{c}_label"] = QtWidgets.QLabel(c, region)
            layout_bg_color.addWidget(self.widget[f"bg_{c}_label"], 0, i)
            
            self.spin_box[f"bg_{c}"] = QtWidgets.QSpinBox(region)
            layout_bg_color.addWidget(self.spin_box[f"bg_{c}"], 1, i)
            self.spin_box[f"bg_{c}"].setRange(0, 255)
            self.spin_box[f"bg_{c}"].setValue(0)
            self.spin_box[f"bg_{c}"].valueChanged.connect(self.update_view)

    def __create_args_region(self):
        # Noises
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)
        
        layout_options = QtWidgets.QHBoxLayout()
        layout_region.addLayout(layout_options, 0, 0, 1, 4)
        # -Title: Noise
        title = QtWidgets.QLabel("Noise", region)
        layout_options.addWidget(title)
        title.setFont(self.fonts["title"])
        # -Combo box.
        if self.__mode == "new":
            noises_list = ("none", *NoiseOperators.get_noises_names())
        else:
            noises_list = NoiseOperators.get_noises_names()
        self.combo_box["arg"] = QtWidgets.QComboBox()
        layout_options.addWidget(self.combo_box["arg"], 2)
        self.combo_box["arg"].addItems(noises_list)
        self.combo_box["arg"].currentTextChanged.connect(self.noise_changed)
        self.noise = noises_list[0]
        self.noise_info = NoiseDialog.noise_args[self.noise]
        
        layout_args = QtWidgets.QGridLayout()
        layout_region.addLayout(layout_args, 1, 0, 1, 7)
        # -Labels and double spin boxes.
        for j in range(3):
            for i, c in enumerate(self.channels):
                self.widget[f"arg{c}{j}_label"] = QtWidgets.QLabel()
                layout_args.addWidget(self.widget[f"arg{c}{j}_label"], 2*j, i)
                self.widget[f"arg{c}{j}_label"].setFont(self.fonts["default"])
                
                self.spin_box[f"arg{c}{j}"] = QtWidgets.QDoubleSpinBox()
                layout_args.addWidget(self.spin_box[f"arg{c}{j}"], 2*j+1, i)
                self.spin_box[f"arg{c}{j}"].setMinimumWidth(80)
                self.spin_box[f"arg{c}{j}"].setDecimals(4)
                 
    def __create_info_region(self):   
        region = QtWidgets.QWidget()
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QHBoxLayout(region)
        
        if self.__mode == "add":
            self.central_layout.addSpacing(10)
        # --pdf(particle distribution function)
        pdf = QtWidgets.QLabel(region)
        layout_region.addWidget(pdf)
        pdf.setPixmap(MplCanvas.latex(r"f($x$)=", 12))
        
        self.widget["formula"] = QtWidgets.QLabel(region)
        layout_region.addWidget(self.widget["formula"])
        
        layout_region.addStretch(1)
    
    def connect_value_changed(self, is_connect: bool):
        """update_view may be called many times when operator changed. Hence,
        the signals should be disconneted and re-connect at the beginning and
        at the end of `__update_args_enable()`, respectively.
        """
        for _, c in enumerate(self.channels):
            for j in range(3):
                if is_connect:
                    self.spin_box[f"arg{c}{j}"].valueChanged.disconnect()
                else:
                    self.spin_box[f"arg{c}{j}"].valueChanged.connect(
                        self.update_view)
    
    # Functions
    def confirm(self, e=None):
        if self.__mode == "new":
            # Get filename
            filename = self.widget["name"].text()
            if not filename:
                # If filename is empty, show tool tip.
                edit_pos = self.widget["name"].mapToGlobal(
                    self.widget["name"].pos())
                edit_size = QPoint(
                    0, self.widget["name"].size().height())
                QToolTip.showText(
                    edit_pos - edit_size, 
                    BasicDialog.filename_tooltip())
                return None
            filename = "".join([
                filename, self.combo_box["file_format"].currentText()
            ])
            filename = self.parentWidget().filename_repeat(filename)
        else:
            if not self.button["show"].isChecked():
                self.show_effect(True)
        
        img = self.parentWidget().current_image
        if self.__mode == "new":
            self.parentWidget().add_new_tab(Image(img), filename)
            self.parentWidget().select_last_tab()
        else:
            self.parentWidget().image = img
        
        self.close_dialog()
        
    def main_work(self):
        """Load args and add noise to an image."""
        if self.__mode == "new":
            output = img = self.__create_background()
        else:
            img = self.original_image
            output = np.empty_like(img)
        # Add noise
        if self.noise != "none":
            for i, c in enumerate(self.channels):
                args = []
                for j, _ in enumerate(self.noise_info["arguments"]):
                    args.append(self.spin_box[f"arg{c}{j}"].value())
                output[:,:,i] = NoiseOperators.add_noise(
                    img[:,:,i], self.noise, args)
        if len(self.channels) == 1:
            return output[:,:,0]
        else:
            return output
    
    def __fix_size_ratio(self, checked: bool):
        if checked:
            width = self.spin_box["width"].value()
            height = self.spin_box["height"].value()
            self.ratio: float = width / height
        
    def height_changed(self, val: int):
        if self.button["size"].isChecked():
            width = round(val / self.ratio)
            self.spin_box["width"].setValue(width)
        self.update_view()
    
    def width_changed(self, val: int):
        if self.button["size"].isChecked():
            height = round(val / self.ratio)
            self.spin_box["height"].setValue(height)
        else:
            self.update_view()
    
    def __update_args_enable(self):
        # self.connect_value_changed(is_connect=True)
        len_arg = len(self.noise_info["arguments"])
        for i, c in enumerate(self.channels):
            for j in range(3):
                if j < len_arg:
                    self.spin_box[f"arg{c}{j}"].show()
                    self.spin_box[f"arg{c}{j}"].setRange(
                        *self.noise_info["ranges"][j])
                    self.spin_box[f"arg{c}{j}"].setValue(
                        self.noise_info["default"][i][j])
                else:
                    self.widget[f"arg{c}{j}_label"].clear()
                    self.spin_box[f"arg{c}{j}"].hide()
        # self.connect_value_changed(is_connect=False)
    
    def update_args_label(self):
        for c in self.channels:
            for j, name in enumerate(self.noise_info["arguments"]):
                self.widget[f"arg{c}{j}_label"].setPixmap(
                    MplCanvas.latex(name, 10))
        
        if self.noise_info["pdf"] is None:
            self.widget["formula"].clear()
        else:
            self.widget["formula"].setPixmap(
                MplCanvas.latex(self.noise_info["pdf"], 12))
        
    def noise_changed(self, noise: str):
        if self.noise != "none":
            for i, c in enumerate(self.channels):
                for j in range(len(self.noise_info["arguments"])):
                    self.noise_info["default"][i][j] = (
                        self.spin_box[f"arg{c}{j}"].value()
                    )
        self.noise = noise
        self.noise_info = NoiseDialog.noise_args[noise]
        
        thread = LabelThread(self)
        thread.start()
        self.__update_args_enable()
        self.update_view()
    
    # Mode: New image
    def __create_background(self):
        # Read size
        size = (
            self.spin_box["height"].value(),
            self.spin_box["width"].value(), 
            len(self.channels)
        )
        img = np.empty(size, dtype=np.uint8)
        # Read background
        for i, c in enumerate(self.channels):
            img[:,:,i] = self.spin_box[f"bg_{c}"].value()
        return img
    
    def color_changed(self, index: int):
        if index == 0:
            self.channels = ("Red", "Green", "Blue")
            chanel = 3
        elif index == 1:
            self.channels = ("Red", ) # Grayscale
            chanel = 1
        
        red = "Brightness" if chanel==1 else "Red"
        self.widget[f"bg_Red_label"].setText(red)
        
        for c in ["Green", "Blue"]:
            if index == 0:
                self.widget[f"bg_{c}_label"].setText(c)
                self.spin_box[f"bg_{c}"].setEnabled(True)
                for j in range(3):
                    self.spin_box[f"arg{c}{j}"].show()
            elif index == 1: # Grayscale only one channel
                self.widget[f"bg_{c}_label"].clear()
                self.spin_box[f"bg_{c}"].setEnabled(False)
                for j in range(3):
                    self.widget[f"arg{c}{j}_label"].clear()
                    self.spin_box[f"arg{c}{j}"].hide()
        
        self.__update_args_enable()
        # Call update_args_label(). Using thread because the process is slow.
        thread = LabelThread(self)
        thread.start()
        self.update_view()
        
    # Mode: Add noise
    def show_effect(self, checked):
        if not checked:
            self.temp = self.parentWidget().current_image
            if self.__mode == "new":
                self.parentWidget().update_all(self.__create_background())
                for direction in ("height", "width"):
                    self.spin_box[direction].setEnabled(False)
                self.combo_box["color"].setEnabled(False)
            else:
                self.parentWidget().update_all()
            self.combo_box["arg"].setEnabled(False)
        else:
            self.parentWidget().update_all(self.temp)
            if self.__mode == "new":
                for direction in ("height", "width"):
                    self.spin_box[direction].setEnabled(True)
                self.combo_box["color"].setEnabled(True)
            self.combo_box["arg"].setEnabled(True)



# Threshold
class ThresholdDialog(BasicDialog):
    """A dialog for applying the thresholding to image."""
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Global Thresholding")
        
        if self.is_color:
            self.grayscale = cv2.cvtColor(self.original_image,
                                          cv2.COLOR_RGB2GRAY)
        
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.combo_box["auto"].setCurrentIndex(3) # Otsu
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Threshold Type
        title = QtWidgets.QLabel("Global Method", region)
        title.setFont(self.fonts["title"])
        auto_list = [
            "None", *SegmentationOperators.get_auto_global_thresholding_names()
        ]
        self.combo_box["auto"] = QtWidgets.QComboBox()
        self.combo_box["auto"].addItems(auto_list)
        self.combo_box["auto"].currentTextChanged.connect(self.auto_changed)
        layout_region.addRow(title, self.combo_box["auto"])
        # -Threshold Type
        label = QtWidgets.QLabel("Threshold Type", region)
        label.setFont(self.fonts["default"])
        type_list = SegmentationOperators.get_thresholding_names()
        self.combo_box["arg"] = QtWidgets.QComboBox()
        self.combo_box["arg"].addItems(type_list)
        self.combo_box["arg"].setCurrentIndex(0)
        self.combo_box["arg"].currentIndexChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["arg"])
        # -Threshold Value
        label2 = QtWidgets.QLabel("Threshold", region)
        label2.setFont(self.fonts["default"])
        self.spin_box["threshold"] = QtWidgets.QSpinBox()
        self.spin_box["threshold"].setRange(0, 255)
        self.spin_box["threshold"].valueChanged.connect(self.type_changed)
        self.spin_box["threshold"].editingFinished.connect(
            self.threashold_edited)
        layout_region.addRow(label2, self.spin_box["threshold"])
        # -Output Grascale
        self.button["grayscale"] = QtWidgets.QCheckBox("Grayscale")
        self.button["grayscale"].clicked.connect(self.update_view)
        if not self.is_color:
            self.button["grayscale"].setChecked(False)
            self.button["grayscale"].hide()
        layout_region.addRow(self.button["grayscale"])
    
    # Functions
    def main_work(self):
        threshold_type = self.combo_box["arg"].currentIndex()
        threshold = self.spin_box["threshold"].value()
        if self.button["grayscale"].isChecked():
            img = self.grayscale
        else:
            img = self.original_image
        return cv2.threshold(img, threshold, 255, threshold_type)[1]
    
    def threashold_edited(self):
        self.combo_box["auto"].setCurrentIndex(0)
        
    def type_changed(self, index: int):
        if index > 1:
            self.combo_box["auto"].setCurrentIndex(0)
        self.update_view()
    
    def auto_changed(self, text: int):
        if text == "None":
            return
        if self.is_color:
            img = self.grayscale
        else:
            img = self.original_image
        self.spin_box["threshold"].setValue(
            round(SegmentationOperators.auto_threshold(img, text))
        )


class AdaptiveThresholdDialog(BasicDialog):
    """A dialog for applying the adaptive thresholding to image."""
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Adaptive Thresholding")
        
        if self.original_image.ndim == 3:
            self.original_image = cv2.cvtColor(self.original_image,
                                               cv2.COLOR_RGB2GRAY)
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.combo_box["auto"].setCurrentIndex(1) # Adaptive Gaussian
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Adaptive
        title = QtWidgets.QLabel("Adaptive", region)
        title.setFont(self.fonts["title"])
        auto_list = SegmentationOperators.get_local_thresholding_names()
        self.combo_box["auto"] = QtWidgets.QComboBox()
        self.combo_box["auto"].addItems(auto_list)
        self.combo_box["auto"].currentIndexChanged.connect(self.update_view)
        layout_region.addRow(title, self.combo_box["auto"])
        # -Adaptive method
        label = QtWidgets.QLabel("Threshold Type", region)
        label.setFont(self.fonts["default"])
        # type_list = ["Binary", "Binary Inv"]
        type_list = SegmentationOperators.get_thresholding_names()[:2]
        self.combo_box["arg"] = QtWidgets.QComboBox()
        self.combo_box["arg"].addItems(type_list)
        self.combo_box["arg"].setCurrentIndex(0)
        self.combo_box["arg"].currentIndexChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["arg"])
        # -Const
        self.widget["const"] = QtWidgets.QLabel("Block Size", region)
        self.widget["const"].setFont(self.fonts["default"])
        self.spin_box["const"] = QtWidgets.QSpinBox()
        self.spin_box["const"].setRange(3, 999)
        self.spin_box["const"].setValue(7)
        self.spin_box["const"].setSingleStep(2)
        self.spin_box["const"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["const"], self.spin_box["const"])
        # -Const 2
        self.widget["const2"] = QtWidgets.QLabel("C", region)
        self.widget["const2"].setFont(self.fonts["default"])
        self.spin_box["const2"] = QtWidgets.QDoubleSpinBox()
        self.spin_box["const2"].setRange(-255, 255)
        self.spin_box["const2"].setDecimals(3)
        self.spin_box["const2"].setSingleStep(0.2)
        self.spin_box["const2"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["const2"], self.spin_box["const2"])
    
    # Functions
    def main_work(self):
        # Get Args
        threshold_type = self.combo_box["arg"].currentIndex()
        op = self.combo_box["auto"].currentIndex()
        block_size = self.spin_box["const"].value()
        if not block_size % 2:
            block_size -= 1
        C = self.spin_box["const2"].value()
        # Thresholg
        return SegmentationOperators.local_thresholding(
            self.original_image, op, threshold_type, block_size, C)



# Morphology
class BasicMorphologyDialog(BasicDialog):
    """A dialog for applying OpenCV Morphologic operation to image"""
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Morphology")
        
        if self.original_image.ndim == 3:
            self.original_image = cv2.cvtColor(self.original_image,
                                               cv2.COLOR_RGB2GRAY)
        # Contains only two different values
        self.binary = True
        if np.unique(self.original_image).size != 2:
            self.binary = False
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Operation
        title = QtWidgets.QLabel("Operation", region)
        title.setFont(self.fonts["title"])
        op_list = MorpologyOperators.get_operaions_names()
        self.combo_box["op"] = QtWidgets.QComboBox()
        self.combo_box["op"].addItems(op_list)
        if not self.binary:
            self.combo_box["op"].removeItem(15) # Skelton
        self.combo_box["op"].currentIndexChanged.connect(self.op_changed)
        layout_region.addRow(title, self.combo_box["op"])
        # -Iteration
        self.widget["iter"] = QtWidgets.QLabel("Iteration", region)
        self.widget["iter"].setFont(self.fonts["default"])
        self.spin_box["iter"] = QtWidgets.QSpinBox()
        self.spin_box["iter"].setRange(0, 999)
        self.spin_box["iter"].setValue(1)
        self.spin_box["iter"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["iter"], self.spin_box["iter"])
        # -Border Type
        label = QtWidgets.QLabel("Border Type", region)
        label.setFont(self.fonts["default"])
        border_list = SpatialOperators.get_border_names()
        self.combo_box["border"] = QtWidgets.QComboBox()
        self.combo_box["border"].addItems(border_list)
        self.combo_box["border"].setCurrentIndex(0) # Constant
        self.combo_box["border"].currentTextChanged.connect(self.update_view)
        layout_region.addRow(label, self.combo_box["border"])
        
        # -Title: Structuring Element Args
        title = QtWidgets.QLabel("SE(s) Args", region)
        title.setFont(self.fonts["title"])
        layout_region.addRow(title)
        se_list = ("Rect", "Cross", "Ellipse")
        # -Shape
        self.widget["shape"] = QtWidgets.QLabel("Shape", region)
        self.widget["shape"].setFont(self.fonts["default"])
        self.combo_box["shape"] = QtWidgets.QComboBox()
        self.combo_box["shape"].addItems(se_list)
        self.combo_box["shape"].currentIndexChanged.connect(self.update_view)
        layout_region.addRow(self.widget["shape"], self.combo_box["shape"])
        # -Ksize
        self.widget["ksize"] = QtWidgets.QLabel("Ksize", region)
        self.widget["ksize"].setFont(self.fonts["default"])
        self.spin_box["ksize"] = QtWidgets.QSpinBox()
        self.spin_box["ksize"].setRange(3, 255)
        self.spin_box["ksize"].setSingleStep(2)
        self.spin_box["ksize"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["ksize"], self.spin_box["ksize"])
        # -Shape 2
        self.widget["shape2"] = QtWidgets.QLabel("Shape 2", region)
        self.widget["shape2"].setFont(self.fonts["default"])
        self.combo_box["shape2"] = QtWidgets.QComboBox()
        self.combo_box["shape2"].addItems(se_list)
        self.combo_box["shape2"].currentIndexChanged.connect(self.update_view)
        layout_region.addRow(self.widget["shape2"], self.combo_box["shape2"])
        # -Ksize 2
        self.widget["ksize2"] = QtWidgets.QLabel("Ksize 2", region)
        self.widget["ksize2"].setFont(self.fonts["default"])
        self.spin_box["ksize2"] = QtWidgets.QSpinBox()
        self.spin_box["ksize2"].setRange(3, 255)
        self.spin_box["ksize2"].setSingleStep(2)
        self.spin_box["ksize2"].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget["ksize2"], self.spin_box["ksize2"])
        
    # Functions
    def main_work(self):
        # Get args
        op = self.combo_box["op"].currentIndex()
        if not self.binary and op > 14:
            op += 1
        iter_ = self.spin_box["iter"].value()
        bordertype = self.combo_box["border"].currentIndex()
        # reflex101 is index 3 but cv2.BORDER_REFLECT101 is 4
        if bordertype == 3: 
            bordertype = 4
        # Create structure element 1
        se1_shape = self.combo_box["shape"].currentIndex()
        se1_ksize = self.spin_box["ksize"].value()
        if not se1_ksize % 2:
            se1_ksize -= 1
        SE = cv2.getStructuringElement(se1_shape, (se1_ksize,se1_ksize))
        # Structure element 2 (se2)
        se2_shape = self.combo_box["shape2"].currentIndex()
        se2_ksize = self.spin_box["ksize"].value()
        if not se2_ksize % 2:
            se2_ksize -= 1
        SE2 = cv2.getStructuringElement(se2_shape, (se1_ksize, se1_ksize))
        return MorpologyOperators.basic_operation(
            self.original_image, op, SE, iter_, bordertype, SE2
        )
    
    # Operator Changed
    def op_changed(self, op):
        if not self.binary and op > 14:
            op += 1
        if op < 12 or op > 15:
            self.widget["iter"].show()
            self.spin_box["iter"].show()
        else:
            self.widget["iter"].hide()
            self.spin_box["iter"].hide()
        if op in (8, 10, 11):
            self.widget["shape"].hide()
            self.combo_box["shape"].hide()
            self.widget["ksize"].hide()
            self.spin_box["ksize"].hide()
        else:
            self.widget["shape"].show()
            self.combo_box["shape"].show()
            self.widget["ksize"].show()
            self.spin_box["ksize"].show()
        if op < 16:
            self.widget["shape2"].hide()
            self.combo_box["shape2"].hide()
            self.widget["ksize2"].hide()
            self.spin_box["ksize2"].hide()
        else:
            self.widget["shape2"].show()
            self.combo_box["shape2"].show()
            self.widget["ksize2"].show()
            self.spin_box["ksize2"].show()
        self.update_view()
