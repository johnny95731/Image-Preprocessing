import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QRegularExpressionValidator
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QToolTip, QWidget
from img_type import IMG_8U

from interface import BasicDialog, MplCanvas, LabelThread
from image_object import Image, NoiseOperators



# Dialogs for functions in menu: file
# (see mainwindow.py, self.__create_main_menubar).


class NewFileDialog(BasicDialog):
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
        # "single color": {
        #     "arguments": (r"prob(\%)", "color"),
        #     "ranges": ((0, 100), (0, 255)),
        #     "default": [[.5, 0] for _ in range(3)],
        #     "pdf": None,
        # },
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
    
    def __init__(self, parent: QWidget = None):
        # Main Dialog
        super().__init__(parent, "Create Image")
        
        # Main region layout
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
        default_name = self.parentWidget().filename_repeat("New File")
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
        file_ext = [
            ext[1:] for ext in Image.support_file_extensions
        ]
        self.combo_box["file_format"].addItems(file_ext)
        
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
        noises_list = ("none", *NoiseOperators.get_noises_names())
        self.combo_box["arg"] = QtWidgets.QComboBox()
        layout_options.addWidget(self.combo_box["arg"], 2)
        self.combo_box["arg"].addItems(noises_list)
        self.combo_box["arg"].currentTextChanged.connect(self.noise_changed)
        self.noise = noises_list[0]
        self.noise_info = NewFileDialog.noise_args[self.noise]
        
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
    # Common
    def confirm(self, e=None):
        # Get filename
        filename = self.widget["name"].text()
        if not filename: # Creating can't be done if no file name.
            # If filename is empty, show tool tip.
            edit_pos = self.widget["name"].mapToGlobal(
                self.widget["name"].pos())
            edit_size = QPoint(
                0, self.widget["name"].size().height())
            QToolTip.showText(
                edit_pos - edit_size, 
                BasicDialog.filename_tooltip())
            return None
        file_extension = self.combo_box["file_format"].currentText()
        filename = self.parentWidget().filename_repeat(filename)
        
        img = self.parentWidget().current_image
        self.parentWidget().add_new_tab(img, filename, file_extension)
        self.parentWidget().select_last_tab()
        
        self.close_dialog()
        
    def main_work(self):
        """Load args and add noise to an image."""
        output = img = self.__create_background()
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
        self.noise_info = NewFileDialog.noise_args[noise]
        
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
            self.parentWidget().update_all(self.__create_background())
            for direction in ("height", "width"):
                self.spin_box[direction].setEnabled(False)
            self.combo_box["color"].setEnabled(False)
            self.combo_box["arg"].setEnabled(False)
        else:
            self.parentWidget().update_all(self.temp)
            for direction in ("height", "width"):
                self.spin_box[direction].setEnabled(True)
            self.combo_box["color"].setEnabled(True)
            self.combo_box["arg"].setEnabled(True)


class SaveFileDialog(BasicDialog):
    """A dialog for saving image."""
    
    jpeg_flags = {
        "quality": {
            "flag": 1,
            "range": (0, 100),
            "default": 95
        }
    }
    jpeg2000_flags = {
        "compression": {
            "flag": 272,
            "range": (0, 1000),
            "default": 1000
        }
    }
    png_flags = {
        "compression": {
            "flag": 16,
            "range": (0, 9),
            "default": 1
        }
    }
    webp_flags = {
        "quality": {
            "flag": 64,
            "range": (0, 100),
            "default": 95
        }
    }
    
    def __init__(self, parent: QWidget, image_object: Image):
        # mode: 0=blur, 1=unsharp masking
        super().__init__(parent, "Save")
        
        self.image_object = image_object
        
        # Main region
        self.__create_name_region()
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()
        
        self.update_view()
        self.show()
    
    def __create_name_region(self,):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        layout_region.setVerticalSpacing(10)
        
        # File name and format
        # -Title: File Name
        title = QtWidgets.QLabel("Filename", region)
        layout_region.addRow(title)
        title.setFont(self.fonts["title"])
        # -Line Edit
        self.widget["name"] = QtWidgets.QLineEdit(
            self.image_object.name, region)
        self.widget["name"].setMinimumWidth(
            int(self.widget["name"].size().width()*2.5))
        # --ToolTip
        self.widget["name"].setToolTip(BasicDialog.filename_tooltip())
        # --Block some special word.
        validator = QRegularExpressionValidator(
            BasicDialog.filename_reg_exp(), self.widget["name"])
        self.widget["name"].setValidator(validator)
        # -Available file extensions
        file_ext = [
            ext[2:] for ext in Image.support_file_extensions
        ]
        self.combo_box["file_format"] = QtWidgets.QComboBox(region)
        self.combo_box["file_format"].addItems(file_ext)
        # --Default: self.image_object's file extensions
        index = file_ext.index(self.image_object.file_extension[1:])
        self.combo_box["file_format"].setCurrentIndex(index)
        self.combo_box["file_format"].currentTextChanged.connect(
            self.__update_args_enable)
        layout_region.addRow(self.widget["name"],
                             self.combo_box["file_format"])
        
        # -Title: File Path
        title = QtWidgets.QLabel("Path", region)
        layout_region.addRow(title)
        title.setFont(self.fonts["default"])
        # -Line Edit
        path = self.image_object.path
        if path is None:
            path = self.parentWidget().save_directory
        if path[-1] not in ("\\", "/"):
            path = "".join([path, "\\"])
        self.widget["path"] = QtWidgets.QLineEdit(path, region)
        self.widget["path"].setMinimumWidth(
            self.widget["name"].size().width())
        # -Open directory 
        self.button["path"] = QtWidgets.QPushButton("Path")
        self.button["path"].clicked.connect(self.get_path)
        layout_region.addRow(self.widget["path"], self.button["path"])
        
    def __create_args_region(self):
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)
        
        # -Title: Arguments
        title = QtWidgets.QLabel("Arguments", region)
        title.setFont(self.fonts["title"])
        
        # -Labels and double spin boxes.
        for j in range(1):
            self.widget[f"arg{j}_label"] = QtWidgets.QLabel()
            self.widget[f"arg{j}_label"].setFont(self.fonts["default"])
            self.spin_box[f"arg{j}"] = QtWidgets.QSpinBox()
            self.spin_box[f"arg{j}"].setMinimumWidth(80)
            layout_region.addRow(self.widget[f"arg{j}_label"],
                                 self.spin_box[f"arg{j}"])
        
        file_extension = self.combo_box["file_format"].currentText()
        self.__update_args_enable(file_extension)
    
    # Functions
    def __get_option_info(self, file_extension: str):
        if file_extension in ("jpeg", "jpg", "jpe"):
            info = SaveFileDialog.jpeg_flags
        elif file_extension == "jp2":
            info = SaveFileDialog.jpeg2000_flags
        elif file_extension == "png":
            info = SaveFileDialog.png_flags
        elif file_extension == "webp":
            info = SaveFileDialog.webp_flags
        return info
        
    def __update_args_enable(self, file_extension: str):
        info = self.__get_option_info(file_extension)
        for j, (key, val) in enumerate(info.items()):
            self.widget[f"arg{j}_label"].setText(key.capitalize())
            self.spin_box[f"arg{j}"].setRange(*val["range"])
            self.spin_box[f"arg{j}"].setValue(val["default"])
    
    def get_path(self, e=None):
        path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Path", self.widget["name"].text())
        if not path:
            return
        path = "".join([path, "/"])
        self.widget["path"].setText(path)
    
    def main_work(self) -> IMG_8U:
        return self.image_object.image
    
    def confirm(self):
        # Get option val
        options = []
        file_extension = self.combo_box["file_format"].currentText()
        info = self.__get_option_info(file_extension)
        for j, val in enumerate(info.values()):
            options.extend([
                val["flag"], self.spin_box[f"arg{j}"].value()
            ])
        # Update path+filename
        self.image_object.name = self.widget["name"].text()
        self.image_object.file_extension = "".join([".", file_extension])
        self.image_object.path = self.widget["path"].text()
        
        self.image_object.save_image(options)
        self.close_dialog()
        
