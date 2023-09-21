import sys

import cv2
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget

from interface import BasicDialog


# Dialogs for functions in menu: edit (see mainwindow.py, 
# self.__create_main_menubar).


class ResizeDialog(BasicDialog):
    """Resizing image."""
    
    support_inter = ( # interpolation
        "Nearest", "Linear", "Cubic", "Area", "Lanczos4"
    )
    
    def __init__(self, parent: QWidget):
        super().__init__(parent, "Resizing")
        
        # Main region
        self.__create_ksize_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
        
    def __create_ksize_region(self):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Size
        title = QtWidgets.QLabel("Size", region)
        title.setFont(self.fonts["title"])
        # -Toggle sizes
        self.button["size"] = QtWidgets.QCheckBox("Fixed Ratio", region)
        layout_region.addRow(title, self.button["size"])
        self.button["size"].toggled.connect(self.__fix_size_ratio)
        # Combo Box
        label = QtWidgets.QLabel("Flag", region)
        label.setFont(self.fonts["default"])
        self.combo_box["inter"] = QtWidgets.QComboBox(region)
        self.combo_box["inter"].addItems(ResizeDialog.support_inter)
        self.combo_box["inter"].setCurrentIndex(3) # Area
        self.combo_box["inter"].currentIndexChanged.connect(self.main_work)
        layout_region.addRow(label, self.combo_box["inter"])
        # -Labels and spin boxes.
        # --Qt use int32 and size should be positive. 
        size_range = (1, 10000)
        size = self.original_image.shape
        for i, direction in enumerate(("width", "height")):
            label = QtWidgets.QLabel(direction, region)
            label.setFont(self.fonts["default"])
            self.spin_box[direction] = QtWidgets.QSpinBox(region)
            self.spin_box[direction].setRange(*size_range)
            self.spin_box[direction].setValue(size[1-i])
            self.spin_box[direction].editingFinished.connect(
                lambda called_from=direction: self.size_changed(called_from))
            layout_region.addRow(label, self.spin_box[direction])
        self.button["size"].setChecked(True)
    
    # Functions
    def __fix_size_ratio(self, checked: bool):
        if checked:
            width = self.spin_box["width"].value()
            height = self.spin_box["height"].value()
            self.ratio: float = width / height
    
    def size_changed(self, called_from):
        if self.button["size"].isChecked():
            if called_from == "width":
                val = self.spin_box["width"].value()
                height = round(val / self.ratio)
                self.spin_box["height"].setValue(height)
            else:
                val = self.spin_box["height"].value()
                width = round(val * self.ratio)
                self.spin_box["width"].setValue(width)
        self.update_view()
    
    def main_work(self, e=None):
        return cv2.resize(
            self.original_image,
            (self.spin_box["width"].value(), self.spin_box["height"].value()),
            interpolation=self.combo_box["inter"].currentIndex()
        )
