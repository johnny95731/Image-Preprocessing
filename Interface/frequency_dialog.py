import sys

import numpy as np
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget

from interface import BasicDialog
sys.path.append('./Image_Processing')
from image_object import Image
from img_type import ARR_8U2D, ARR_8U3D, IMG_8U



"""
Dialogs for functions in menu: Frequency (see mainwindow.py,
self.__create_main_menubar).
"""


class BasicFilterDialog(BasicDialog):
    """A dialog for bluring image in frequency domain."""
    
    support_type = ("Gaussian", "Butterworth")
    support_filter = ("Low-Pass", "High-Pass", "Band-Pass", "Band-Reject")
    blur_args = [
        {
            "args": ("Sigma X", "Sigma Y"),
            "ranges": (
                (0, float("inf")),
                (0, float("inf")),
            )
        },
        {
            "args": ("Cutoff", "Order"),
            "ranges": (
                (0, float("inf")),
                (0, float("inf")),
            )
        },
    ]
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Blur Image")
        
        self.is_color =  self.original_image.ndim == 3
        if self.is_color:
            temp = self.original_image.astype(np.float32)
            self.rfft = [np.fft.rfft2(temp[:,:,i]) for i in range(3)]
            self.rfft_size = self.rfft[0].shape[:2]
        else:
            self.rfft = np.fft.rfft2(self.original_image.astype(np.float32))
            self.rfft_size = self.rfft.shape[:2]
        
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.type_changed()
        self.update_view()
        self.show()
        
    def __create_args_region(self):
        # Blur
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Filter Type
        title = QtWidgets.QLabel("Types", region)
        title.setFont(self.fonts["title"])
        # -Combo box.
        filter_type_list = BasicFilterDialog.support_type
        self.combo_box["type"] = QtWidgets.QComboBox()
        self.combo_box["type"].addItems(filter_type_list)
        self.combo_box["type"].currentTextChanged.connect(self.type_changed)
        layout_region.addRow(title, self.combo_box["type"])
        
        # -Title: Filter(Low-Pass, High-Pass, etc.)
        label = QtWidgets.QLabel("Filters", region)
        label.setFont(self.fonts["default"])
        # -Combo box.
        filter_list = BasicFilterDialog.support_filter
        self.combo_box["arg"] = QtWidgets.QComboBox()
        self.combo_box["arg"].addItems(filter_list)
        self.combo_box["arg"].currentTextChanged.connect(self.type_changed)
        layout_region.addRow(label, self.combo_box["arg"])
        
        # -Labels and double spin boxes.
        for j in range(3):
            self.widget[f"arg{j}_label"] = QtWidgets.QLabel()
            self.widget[f"arg{j}_label"].setFont(self.fonts["default"])
            self.spin_box[f"arg{j}"] = QtWidgets.QDoubleSpinBox()
            self.spin_box[f"arg{j}"].setMinimumWidth(80)
            self.spin_box[f"arg{j}"].setRange(0, float("inf"))
            self.spin_box[f"arg{j}"].valueChanged.connect(self.update_view)
            layout_region.addRow(self.widget[f"arg{j}_label"],
                                 self.spin_box[f"arg{j}"])
    
    # Functions
    def main_work(self):
        import rfft2_filters
        # Get Args
        type_ = self.combo_box["type"].currentIndex()
        filter_ = self.combo_box["arg"].currentIndex()
        args = [
            np.float32(self.spin_box[f"arg{j}"].value()) for j in range(3)
        ]
        # Filtering
        if not type_: # Gaussian
            if not filter_:
                mask = rfft2_filters.gaussian_lowpass(
                    self.rfft_size, args[0], args[1])
            elif filter_ == 1:
                mask = rfft2_filters.gaussian_highpass(
                    self.rfft_size, args[0], args[1])
            elif filter_ == 2:
                mask = rfft2_filters.gaussian_bandpass(
                    self.rfft_size, args[0], args[1])
            elif filter_ == 3:
                mask = rfft2_filters.gaussian_bandreject(
                    self.rfft_size, args[0], args[1])
        elif type_ == 1: # Butterworth
            if not filter_:
                mask = rfft2_filters.butterworth_lowpass(
                    self.rfft_size, args[0], args[1])
            elif filter_ == 1:
                mask = rfft2_filters.butterworth_highpass(
                    self.rfft_size, args[0], args[1])
            elif filter_ == 2:
                mask = rfft2_filters.butterworth_bandpass(
                    self.rfft_size, args[1], args[2], args[0])
            elif filter_ == 3:
                mask = rfft2_filters.butterworth_bandreject(
                    self.rfft_size, args[1], args[2], args[0])
        if self.is_color:
            output = []
            for i in range(3):
                temp = np.multiply(self.rfft[i], mask)
                output.append(cv2.convertScaleAbs(np.fft.irfft2(temp)))
            output = cv2.merge(output)
        else:
            output = rfft2_filters.filter(self.rfft, mask)
            output = cv2.convertScaleAbs(np.fft.irfft2(output))
        return output
    
    def type_changed(self, e=None):
        type_ = self.combo_box["type"].currentIndex()
        filter_ = self.combo_box["arg"].currentIndex()
        if filter_ < 2:
            args = BasicFilterDialog.blur_args[type_]
            for j in range(2):
                self.widget[f"arg{j}_label"].setText(
                    BasicFilterDialog.blur_args[type_]["args"][j])
                self.spin_box[f"arg{j}"].setRange(*args["ranges"][j])
            self.widget["arg2_label"].hide()
            self.spin_box["arg2"].hide()
        elif type_ == 0:
            for j, text in enumerate(["Band Center", "Band Width"]):
                self.widget[f"arg{j}_label"].setText(text)
            self.widget["arg2_label"].hide()
            self.spin_box["arg2"].hide()
        elif type_ == 1:
            for j, text in enumerate(["Order", "Band Center", "Band Width"]):
                self.widget[f"arg{j}_label"].show()
                self.widget[f"arg{j}_label"].setText(text)
                self.spin_box[f"arg{j}"].show()
