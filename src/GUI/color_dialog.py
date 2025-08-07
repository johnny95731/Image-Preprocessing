__all__ = [
    'FuncTransDialog',
    'PiecewiseLinearDialog',
    'SlicingDialog',
    'kMeansDialog',
    'HistogramMatchingDialog',
    'CLAHEDialog',
]

from math import ceil, floor

import numpy as np
import cv2
from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QWidget, QTableWidgetItem
from PyQt6.QtCore import QPoint, Qt, QVariant
from PyQt6.QtWidgets import QToolTip

from numexpr import evaluate

from src.GUI.interface import MplCanvas, BasicDialog, LabelThread
from src.GUI.image_object import EnhanceOperators

from src.utils.helpers import transform
import src.enhance as enhance
from src.colors import color_slicing
from src.seg import kmeans_seg


# Dialogs for actions in menu: Color.
# (see mainwindow.py, self.__create_main_menubar).


# Intensity Transform
class FuncTransDialog(BasicDialog):
    """
    A dialog for adjusting the contract of current image by applying some
    function (see EnhanceOperators.get_intensity_transform_names()).
    """

    func_trans_args = {
        0: {  # Linear Transformation
            'arguments': ('c(ratio)', 'b(bias)'),
            'ranges': ((-float('inf'), float('inf')), (-float('inf'), float('inf'))),
            'default': [[1, 0] for _ in range(3)],
            'formula': r'$ratio*s + bias$',
        },
        1: {  # Gamma Correction
            'arguments': (r'$\gamma$', 'c(ratio)', 'b(bias)'),
            'ranges': (
                (-float('inf'), float('inf')),
                (0, float('inf')),
                (-float('inf'), float('inf')),
            ),
            'default': [[1, 0, 0] for _ in range(3)],
            'formula': r'$c * s^\gamma + b$',
        },
        2: {  # Log Transformation
            'arguments': ('c(ratio)', 'b(bias)'),
            'ranges': ((0, float('inf')), (-255, 255)),
            'default': [[0, 0] for _ in range(3)],
            'formula': r'$c * \mathrm{ln}(1+s) + b$',
        },
        3: {  # Arctan Transformation
            'arguments': (r'$\gamma(\%)$', r'$\alpha$(center)'),
            'ranges': ((0, float('inf')), (0, 255)),
            'default': [[1, 127.5] for _ in range(3)],
            'formula': ''.join((
                r'$c * \mathrm{arctan}(\gamma*s-b) + d, $',
                '\n',
                r'$c = 255/(\mathrm{arctan}(255*\gamma-b)+\mathrm{arctan}(b))$,',
                '\n',
                r'$b = c * \gamma$, ',
                r'$d = \alpha*\mathrm{arctan}(b)$',
            )),
        },
        4: {  # Logistic Correction
            'arguments': (r'$\sigma$', r'$\alpha$(center)'),
            'ranges': ((0, float('inf')), (0, 255)),
            'default': [[7, 127.5] for _ in range(3)],
            'formula': ''.join((
                r'$\displaystyle 255\frac{P(s)-P(0)}{P(255)-P(0)}$, ',
                r'$\displaystyle P(s)=\frac{1}{1+e^{-(s-\alpha)/{\sigma^2}}}$.',
            )),
        },
        5: {  # Beta Correction
            'arguments': ('a', 'b'),
            'ranges': ((0, float('inf')), (0, float('inf'))),
            'default': [[2, 2] for _ in range(3)],
            'formula': ''.join((
                r'$\displaystyle \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$',
                r'$\displaystyle \int_0^s t^{a-1}(1-t)^{b-1}\,\mathrm{d}t$',
            )),
        },
        6: {  # Level Slicing Type I
            'arguments': ('low', 'high', 'fg', 'bg'),
            'ranges': ((0, 255), (0, 255), (0, 255), (0, 255)),
            'default': [[100, 154, 0, 255] for _ in range(3)],
            'formula': ''.join((
                r'fg, if $low\le s\le high,$',
                '\n',
                r'bg, elsewise.',
            )),
        },
        7: {  # Level Slicing Type II
            'arguments': ('low', 'high', 'level'),
            'ranges': ((0, 255), (0, 255), (0, 255)),
            'default': [[100, 154, 255] for _ in range(3)],
            'formula': ''.join((
                r'level, if $low\le s\le high,$',
                '\n',
                r's, elsewise.',
            )),
        },
        8: {  # Level Slicing Type II (Inv)
            'arguments': ('low', 'high', 'level'),
            'ranges': ((0, 255), (0, 255), (0, 255)),
            'default': [[100, 154, 255] for _ in range(3)],
            'formula': ''.join((
                r's, if $low\le s\le high,$',
                '\n',
                r'level, elsewise.',
            )),
        },
    }

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'Intensity Transformation')

        # Attributes about image
        if not self.is_color:
            self.original_image = self.original_image[:, :, np.newaxis]

        # Main region
        self.__create_info_region()
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.func = 0
        self.func_info = FuncTransDialog.func_trans_args[0]
        self.__update_args_enable()
        thread = LabelThread(self)
        thread.start()
        self.update_view()
        self.show()

    def __create_info_region(self):
        region = QtWidgets.QWidget()
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        self.widget['canvas'], self.widget['plot'] = MplCanvas(
            region, (1.125, 2), visible=True
        ).widget
        layout_region.addWidget(self.widget['canvas'], 0, 0, 1, 4)

        # --pdf(particle distribution function)
        formula = QtWidgets.QLabel(region)
        layout_region.addWidget(formula, 1, 0, 1, 1)
        formula.setPixmap(MplCanvas.latex(r'T(s)=', 12))

        self.widget['formula'] = QtWidgets.QLabel(region)
        layout_region.addWidget(self.widget['formula'], 1, 1, 1, 3)

    def __create_args_region(self):
        # Function
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        layout_options = QtWidgets.QHBoxLayout()
        layout_region.addLayout(layout_options, 0, 0, 1, 4)
        # -Title: Function
        title = QtWidgets.QLabel('Function', region)
        layout_options.addWidget(title)
        title.setFont(self.fonts['title'])
        # -Combo box.
        funcs_list = EnhanceOperators.get_intensity_transform_names()
        self.combo_box['arg'] = QtWidgets.QComboBox()
        layout_options.addWidget(self.combo_box['arg'], 2)
        self.combo_box['arg'].addItems(funcs_list)
        self.combo_box['arg'].currentIndexChanged.connect(self.func_changed)

        layout_vals = QtWidgets.QGridLayout()
        layout_region.addLayout(layout_vals, 1, 0, 1, 7)
        # -Labels and double spin boxes.
        for i, c in enumerate(self.channels):
            if self.is_color:
                label = QtWidgets.QLabel(c, region)
                layout_vals.addWidget(label, 0, i)
            for j in range(4):
                self.widget[f'arg{c}{j}_label'] = QtWidgets.QLabel()
                self.widget[f'arg{c}{j}_label'].setFont(self.fonts['default'])

                self.spin_box[f'arg{c}{j}'] = QtWidgets.QDoubleSpinBox()
                self.spin_box[f'arg{c}{j}'].setMinimumWidth(80)
                self.spin_box[f'arg{c}{j}'].valueChanged.connect(self.update_view)
                self.spin_box[f'arg{c}{j}'].setDecimals(4)
                self.spin_box[f'arg{c}{j}'].setSingleStep(0.5)
                if self.is_color:
                    # Horizontal: Arguments of R, G, B, respectively.
                    # Vertical: The 1st, 2nd, 3rd argument.
                    layout_vals.addWidget(self.widget[f'arg{c}{j}_label'], 2 * j + 1, i)
                    layout_vals.addWidget(self.spin_box[f'arg{c}{j}'], 2 * j + 2, i)
                else:
                    # Horizontal: The 1st, 2nd, 3rd argument.
                    layout_vals.addWidget(
                        self.widget[f'arg{c}{j}_label'], 1 + 2 * (j // 2), j % 2
                    )
                    layout_vals.addWidget(
                        self.spin_box[f'arg{c}{j}'], 2 + 2 * (j // 2), j % 2
                    )

    # Functions
    def main_work(self):
        self.widget['plot'].clear()
        self.widget['plot'].set_ylim((0, 255))
        self.widget['plot'].set_xlim((0, 255))
        output = np.empty_like(self.original_image)
        plot_x = np.arange(256, dtype=np.uint8).reshape((1, 256))
        # Transform
        for i, c in enumerate(self.channels):
            args = []
            for j, _ in enumerate(self.func_info['arguments']):
                args.append(self.spin_box[f'arg{c}{j}'].value())
            plot_y = EnhanceOperators.intensity_transform(plot_x, self.func, args)[0]
            self.widget['plot'].plot(plot_x[0], plot_y, color=c[0].lower(), alpha=0.7)
            output[:, :, i] = EnhanceOperators.intensity_transform(
                self.original_image[:, :, i], self.func, args
            )
        self.widget['canvas'].draw()
        if self.is_color:
            return output
        else:
            return output[:, :, 0]

    def __update_args_enable(self):
        len_arg = len(self.func_info['arguments'])
        for i, c in enumerate(self.channels):
            for j in range(4):
                if j < len_arg:
                    self.spin_box[f'arg{c}{j}'].show()
                    self.spin_box[f'arg{c}{j}'].setRange(*self.func_info['ranges'][j])
                    self.spin_box[f'arg{c}{j}'].setValue(
                        self.func_info['default'][i][j]
                    )
                else:
                    self.widget[f'arg{c}{j}_label'].clear()
                    self.spin_box[f'arg{c}{j}'].hide()

    def update_args_label(self):
        for c in self.channels:
            for j, name in enumerate(self.func_info['arguments']):
                self.widget[f'arg{c}{j}_label'].setPixmap(MplCanvas.latex(name, 10))
        self.widget['formula'].setPixmap(MplCanvas.latex(self.func_info['formula'], 12))

    def func_changed(self, func: int | None = None):
        self.func = func
        self.func_info = FuncTransDialog.func_trans_args[func]

        LabelThread(self).start()
        self.__update_args_enable()
        self.update_view()

    def show_effect(self, checked):
        super().show_effect(checked)
        if not checked:
            for c in self.channels:
                for j in range(4):
                    self.spin_box[f'arg{c}{j}'].setEnabled(False)
            for widget in self.combo_box.values():
                widget.setEnabled(False)
        else:
            for c in self.channels:
                for j in range(4):
                    self.spin_box[f'arg{c}{j}'].setEnabled(True)
            for widget in self.combo_box.values():
                widget.setEnabled(True)


class PLTableWidgetItem(QTableWidgetItem):
    """QTableWidgetItem for PiecewiseLinearDialog."""

    def __lt__(self, other):
        my_value = float(self.data(Qt.EditRole))
        other_value = float(other.data(Qt.EditRole))
        return my_value < other_value


class PiecewiseLinearDialog(BasicDialog):
    """
    A dialog for applying piecewise linear function to transform intensity of
    image.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'Piecewise Linear Trensform')

        # Main region
        self.__create_info_region()
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.show()

    def __create_info_region(self):
        region = QtWidgets.QWidget()
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        self.widget['canvas'], self.widget['axis'] = MplCanvas(
            region, (1.125, 2), visible=True
        ).widget
        layout_region.addWidget(self.widget['canvas'], 0, 0, 1, 4)
        n = np.arange(256)
        self.widget['plot'] = self.widget['axis'].plot(n, n, color='r')[0]

    def __create_args_region(self):
        # Hist
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        layout_options = QtWidgets.QHBoxLayout()
        layout_region.addLayout(layout_options, 0, 0, 1, 3)
        # -Title: Hist
        title = QtWidgets.QLabel('Breakpoints', region)
        layout_options.addWidget(title)
        title.setFont(self.fonts['title'])
        # -Combo box.
        # hist_list = PiecewiseLinearDialog.support_hist
        self.combo_box['arg'] = QtWidgets.QComboBox()
        layout_options.addWidget(self.combo_box['arg'], 2)
        # self.combo_box["arg"].addItems(hist_list)
        # self.combo_box["arg"].currentIndexChanged.connect(self.func_changed)
        # -Table
        self.widget['table'] = QtWidgets.QTableWidget(256, 2, region)
        layout_region.addWidget(self.widget['table'], 0, 0, 1, 7)
        self.widget['table'].setHorizontalHeaderLabels(['x', 'y'])
        self.widget['table'].setColumnWidth(0, 90)
        self.widget['table'].setColumnWidth(1, 90)

        vals = ('0', '255')
        for i in range(4):
            row, col = divmod(i, 2)
            item = PLTableWidgetItem()
            item.setData(Qt.EditRole, QVariant(vals[row]))
            self.widget['table'].setItem(row, col, item)
        self.widget['table'].cellChanged.connect(self.cell_changed)

    # Functions
    def main_work(self):
        self.widget['table'].sortItems(0)
        # Get values
        x_points = list()
        y_heights = list()
        for i in range(256):
            x_item = self.widget['table'].item(i, 0)
            y_item = self.widget['table'].item(i, 1)
            if x_item is None or y_item is None:
                break
            x_points.append(float(x_item.text()))
            y_heights.append(float(y_item.text()))
        x_points = np.array(x_points, np.float32)
        y_heights = np.array(y_heights, np.float32)
        # Calculate Table
        table = enhance.piecewise_linear_function(x_points, y_heights)
        # Plot
        self.widget['plot'].set_ydata(table)
        self.widget['canvas'].draw()
        #
        if len(self.channels) == 1:
            return transform(self.original_image, table)
        else:
            output = np.empty_like(self.original_image)
            for i in range(3):
                output[:, :, i] = transform(self.original_image[:, :, i], table)
            return output

    def cell_changed(self, row, col):
        item = self.widget['table'].item(row, col)
        text = item.text()
        if (
            not text or not text.replace('.', '', 1).isdigit()  # Not a numerical text.
        ):  # Empty or can not convert to float/int
            self.widget['table'].cellChanged.disconnect()
            self.widget['table'].takeItem(row, col)
            self.widget['table'].cellChanged.connect(self.cell_changed)
            if self.widget['table'].item(row, (col + 1) % 2) is None:
                self.update_view()
            return
        self.widget['table'].cellChanged.disconnect()  # Avoid infinity loop
        item = PLTableWidgetItem()
        item.setData(Qt.EditRole, QVariant(text))
        self.widget['table'].takeItem(row, col)
        self.widget['table'].setItem(row, col, item)
        self.widget['table'].cellChanged.connect(self.cell_changed)

        if self.widget['table'].item(row, (col + 1) % 2) is None:
            return
        self.update_view()


# Slicing or Segmentation
class SlicingDialog(BasicDialog):
    """A dialog for applying color slicing to image."""

    support_norm = ('Sup-Norm', 'Abs. Sum', 'Euclidean Norm')

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'Color Slicing')

        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.update_view()
        self.show()

    def __create_args_region(self):
        # Function
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        if self.is_color:
            layout_options = QtWidgets.QHBoxLayout()
            layout_region.addLayout(layout_options, 0, 0, 1, 4)
            # -Title: Function
            title = QtWidgets.QLabel('Norm', region)
            layout_options.addWidget(title)
            title.setFont(self.fonts['title'])
            # -Combo box.
            norms_list = SlicingDialog.support_norm
            self.combo_box['arg'] = QtWidgets.QComboBox()
            layout_options.addWidget(self.combo_box['arg'], 2)
            self.combo_box['arg'].addItems(norms_list)
            self.combo_box['arg'].currentIndexChanged.connect(self.update_view)

            # -Labels and double spin boxes.
            layout_vals = QtWidgets.QGridLayout()
            layout_region.addLayout(layout_vals, 1, 0, 1, 7)
            # background
            j = 0
            i = 0
            label = QtWidgets.QLabel('background(R,G,B)', region)
            label.setFont(self.fonts['default'])
            layout_vals.addWidget(label, 0, 0, 1, 2)
            self.spin_box[f'arg{i}-{j}'] = QtWidgets.QSpinBox()
            self.spin_box[f'arg{i}-{j}'].setMinimumWidth(80)
            self.spin_box[f'arg{i}-{j}'].setRange(0, 255)
            self.spin_box[f'arg{i}-{j}'].setValue(127)
            self.spin_box[f'arg{i}-{j}'].valueChanged.connect(self.update_view)
            layout_vals.addWidget(self.spin_box[f'arg{i}-{j}'], 0, 2)
            # center
            j = 1
            for i, _ in enumerate(self.channels):
                label = QtWidgets.QLabel('center(R,G,B)', region)
                label.setFont(self.fonts['default'])
                layout_vals.addWidget(label, 2 * j, 0, 1, 2)
                self.spin_box[f'arg{i}-{j}'] = QtWidgets.QSpinBox()
                self.spin_box[f'arg{i}-{j}'].setMinimumWidth(80)
                self.spin_box[f'arg{i}-{j}'].setRange(0, 255)
                self.spin_box[f'arg{i}-{j}'].setValue(127)
                self.spin_box[f'arg{i}-{j}'].valueChanged.connect(self.update_view)
                # Horizontal: Arguments of R, G, B, respectively.
                # Vertical: The 1st, 2nd, 3rd argument.
                layout_vals.addWidget(self.spin_box[f'arg{i}-{j}'], 2 * j + 1, i)
            # radius
            j = 2
            i = 0
            label = QtWidgets.QLabel('radius', region)
            label.setFont(self.fonts['default'])
            layout_vals.addWidget(label, 2 * j, 0, 1, 2)
            self.spin_box[f'arg{i}-{j}'] = QtWidgets.QDoubleSpinBox()
            self.spin_box[f'arg{i}-{j}'].setMinimumWidth(80)
            self.spin_box[f'arg{i}-{j}'].setDecimals(3)
            self.spin_box[f'arg{i}-{j}'].setSingleStep(0.2)
            self.spin_box[f'arg{i}-{j}'].setRange(0, float('inf'))
            self.spin_box[f'arg{i}-{j}'].setValue(100)
            self.spin_box[f'arg{i}-{j}'].valueChanged.connect(self.update_view)
            # Horizontal: Arguments of R, G, B, respectively.
            # Vertical: The 1st, 2nd, 3rd argument.
            layout_vals.addWidget(self.spin_box[f'arg{i}-{j}'], 2 * j, 2)
        else:
            layout_vals = QtWidgets.QFormLayout()
            layout_region.addLayout(layout_vals, 1, 0, 1, 7)
            i = 0
            for j, arg_name in enumerate(['background', 'center', 'radius']):
                label = QtWidgets.QLabel(arg_name, region)
                label.setFont(self.fonts['default'])
                self.spin_box[f'arg{i}-{j}'] = QtWidgets.QDoubleSpinBox()
                self.spin_box[f'arg{i}-{j}'].setMinimumWidth(80)
                self.spin_box[f'arg{i}-{j}'].setDecimals(1)
                self.spin_box[f'arg{i}-{j}'].setRange(0, 255)
                self.spin_box[f'arg{i}-{j}'].setValue(127.5)
                self.spin_box[f'arg{i}-{j}'].valueChanged.connect(self.update_view)

                layout_vals.addRow(label, self.spin_box[f'arg{i}-{j}'])
            self.spin_box[f'arg{i}-2'].setDecimals(3)
            self.spin_box[f'arg{i}-2'].setSingleStep(0.2)
            self.spin_box[f'arg{i}-2'].setRange(0, float('inf'))
            self.spin_box[f'arg{i}-2'].setValue(100)

    # Functions
    def main_work(self):
        bg = int(self.spin_box['arg0-0'].value())
        radius = self.spin_box['arg0-2'].value()
        if self.is_color:
            center = [
                self.spin_box[f'arg{i}-1'].value() for i, _ in enumerate(self.channels)
            ]
            return color_slicing(
                self.original_image,
                center,
                radius,
                bg,
                self.combo_box['arg'].currentIndex(),
            )
        else:
            center = self.spin_box['arg0-1'].value()
            low = ceil(center - radius)
            high = floor(center + radius)
            table = np.arange(256, dtype=np.uint8)
            table[:low] = bg
            table[high + 1 :] = bg
            return transform(self.original_image, table)


class kMeansDialog(BasicDialog):
    """A dialog for applying k-means to segment image."""

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'k-Means segmentation')

        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.update_view()
        self.show()

    def __create_args_region(self):
        # Function
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)

        label = QtWidgets.QLabel('k', region)
        label.setFont(self.fonts['default'])
        self.spin_box['arg'] = QtWidgets.QSpinBox()
        self.spin_box['arg'].setMinimumWidth(80)
        self.spin_box['arg'].setRange(1, 256)
        self.spin_box['arg'].setValue(16)
        self.spin_box['arg'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['arg'])

        label = QtWidgets.QLabel('Max Iter', region)
        label.setFont(self.fonts['default'])
        self.spin_box['max_iter'] = QtWidgets.QSpinBox()
        self.spin_box['max_iter'].setMinimumWidth(80)
        self.spin_box['max_iter'].setRange(1, 1000)
        self.spin_box['max_iter'].setValue(100)
        self.spin_box['max_iter'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['max_iter'])

        label = QtWidgets.QLabel('Tolerance', region)
        label.setFont(self.fonts['default'])
        self.spin_box['tol'] = QtWidgets.QDoubleSpinBox()
        self.spin_box['tol'].setMinimumWidth(80)
        self.spin_box['tol'].setRange(0.1, 100)
        self.spin_box['tol'].setDecimals(1)
        self.spin_box['tol'].setValue(1)
        self.spin_box['tol'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['tol'])

    # Functions
    def main_work(self):
        return kmeans_seg(
            self.original_image,
            self.spin_box['arg'].value(),
            self.spin_box['max_iter'].value(),
            self.spin_box['tol'].value(),
        )


# Hist
class HistogramMatchingDialog(BasicDialog):
    """
    A dialog for applying histogram matching to image. User can write formula
    of pdf manually or use some
    """

    support_hist = (
        'exp(-(s-127.5)**2/1000)',
        'exp(-(s-127.5)**2/5000)',
        's**2',
        'sqrt(s)',
        '1/sqrt(1+s)',
    )

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'Histogram Matching')

        if self.is_color:
            self.h, self.s, self.v = cv2.split(
                cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
            )
        # Main region
        self.__create_info_region()
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.show()

    def __create_info_region(self):
        region = QtWidgets.QWidget()
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        self.widget['canvas'], self.widget['plot'] = MplCanvas(
            region, (1.125, 2)
        ).widget
        layout_region.addWidget(self.widget['canvas'], 0, 0, 1, 4)

    def __create_args_region(self):
        # Hist
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        layout_options = QtWidgets.QHBoxLayout()
        layout_region.addLayout(layout_options, 0, 0, 1, 3)
        # -Title: Hist
        title = QtWidgets.QLabel('Hist', region)
        layout_options.addWidget(title)
        title.setFont(self.fonts['title'])
        # -Combo box.
        hist_list = ('customization', *HistogramMatchingDialog.support_hist)
        self.combo_box['arg'] = QtWidgets.QComboBox()
        layout_options.addWidget(self.combo_box['arg'], 2)
        self.combo_box['arg'].addItems(hist_list)
        self.combo_box['arg'].currentIndexChanged.connect(self.func_changed)
        # -Line Edit
        layout_vals = QtWidgets.QFormLayout()
        layout_region.addLayout(layout_vals, 1, 0, 1, 7)

        label = QtWidgets.QLabel(region)
        self.widget['edit'] = QtWidgets.QLineEdit()
        layout_vals.addRow(label, self.widget['edit'])

        label.setPixmap(MplCanvas.latex(r'pdf($s$)=', 12))
        self.widget['edit'].setToolTip(
            ''.join((
                'The support of pdf is integers between 0 and 255 inclusive.',
                '\n',
                'The input will automatically normalize to sum(pdf)=1.',
            ))
        )
        # -Matching Saturation channel
        self.button['sat'] = QtWidgets.QCheckBox('Match S channel')
        self.button['sat'].clicked.connect(self.update_view)
        self.button['sat'].setChecked(True)
        if not self.is_color:
            self.button['sat'].hide()
        layout_vals.addRow(self.button['sat'])

    # Functions
    def main_work(self):
        self.widget['plot'].clear()
        support = np.arange(256, dtype=np.float32)
        # Calculate
        formula = self.widget['edit'].text()
        if not formula:
            return None
        try:
            pdf = evaluate(f'abs({formula})', local_dict={'s': support})
            # exec(f"pdf = {formula}", locals={"s": support})
        except Exception as e:
            print(e)
            edit_pos = self.widget['edit'].mapToGlobal(self.widget['edit'].pos())
            edit_size = QPoint(0, self.widget['edit'].size().height())
            QToolTip.showText(
                edit_pos - edit_size, ''.join(f'"{formula}" can\'t be calculate.')
            )
            return None
        pdf /= np.sum(pdf)
        # Plot pdf and cdf
        self.widget['plot'].bar(support, pdf, color='r', alpha=1)
        self.widget['canvas'].draw()
        #
        if len(self.channels) == 1:
            return enhance.histogram_matching(self.original_image, pdf)
        else:
            v = enhance.histogram_matching(self.v, pdf)
            if self.button['sat'].isChecked():
                s = enhance.histogram_matching(self.s, pdf)
            else:
                s = self.s
            return cv2.cvtColor(cv2.merge([self.h, s, v]), cv2.COLOR_HSV2RGB)

    def func_changed(self, func: int | None = None):
        if func:
            self.widget['edit'].setText(HistogramMatchingDialog.support_hist[func - 1])
        else:
            self.widget['edit'].clear()
        self.update_view()


class CLAHEDialog(BasicDialog):
    """
    A dialog for applying Contrast Limited Adaptive Histogram
    Equalization (CLAHE) to image.
    """

    def __init__(self, parent: QWidget):
        super().__init__(parent, 'Histogram Matching')

        if self.is_color:
            self.h, self.s, self.original_image = cv2.split(
                cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HSV)
            )
        # Main region
        self.__create_args_region()
        self.central_layout.addStretch(1)
        self.create_basic_buttons()

        self.sat_changed(False)
        self.show()

    def __create_args_region(self):
        # Hist
        region = QtWidgets.QWidget(self)
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QFormLayout(region)

        # -Title: Arguments
        title = QtWidgets.QLabel('Model Args', region)
        title.setFont(self.fonts['title'])
        layout_region.addRow(title)
        # -Grid Size
        label = QtWidgets.QLabel('Grid Size', region)
        label.setFont(self.fonts['subtitle'])
        layout_region.addRow(label)
        #
        label = QtWidgets.QLabel('Size(X)', region)
        label.setFont(self.fonts['default'])
        self.spin_box['size_x'] = QtWidgets.QSpinBox()
        self.spin_box['size_x'].setRange(0, 999)
        self.spin_box['size_x'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['size_x'])
        label = QtWidgets.QLabel('Size(Y)', region)
        label.setFont(self.fonts['default'])
        self.spin_box['size_y'] = QtWidgets.QSpinBox()
        self.spin_box['size_y'].setRange(0, 999)
        self.spin_box['size_y'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['size_y'])
        # -Clip Limit
        label = QtWidgets.QLabel('Clip Limit', region)
        label.setFont(self.fonts['subtitle'])
        self.spin_box['limit'] = QtWidgets.QDoubleSpinBox()
        self.spin_box['limit'].setRange(0, float('inf'))
        self.spin_box['limit'].setSingleStep(0.5)
        self.spin_box['limit'].valueChanged.connect(self.update_view)
        layout_region.addRow(label, self.spin_box['limit'])
        # -Grid Size 2
        self.widget['size2'] = QtWidgets.QLabel('Grid Size', region)
        self.widget['size2'].setFont(self.fonts['subtitle'])
        layout_region.addRow(self.widget['size2'])
        #
        self.widget['size2_x'] = QtWidgets.QLabel('Size(X)', region)
        self.widget['size2_x'].setFont(self.fonts['default'])
        self.spin_box['size2_x'] = QtWidgets.QSpinBox()
        self.spin_box['size2_x'].setRange(0, 999)
        self.spin_box['size2_x'].hide()
        self.spin_box['size2_x'].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget['size2_x'], self.spin_box['size2_x'])
        self.widget['size2_y'] = QtWidgets.QLabel('Size(Y)', region)
        self.widget['size2_y'].setFont(self.fonts['default'])
        self.spin_box['size2_y'] = QtWidgets.QSpinBox()
        self.spin_box['size2_y'].setRange(0, 999)
        self.spin_box['size2_y'].hide()
        self.spin_box['size2_y'].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget['size2_y'], self.spin_box['size2_y'])
        # -Clip Limit 2
        self.widget['limit2'] = QtWidgets.QLabel('Clip Limit', region)
        self.widget['limit2'].setFont(self.fonts['subtitle'])
        self.spin_box['limit2'] = QtWidgets.QDoubleSpinBox()
        self.spin_box['limit2'].setRange(0, float('inf'))
        self.spin_box['limit2'].setSingleStep(0.5)
        self.spin_box['limit2'].valueChanged.connect(self.update_view)
        layout_region.addRow(self.widget['limit2'], self.spin_box['limit2'])
        # -Matching Saturation channel
        self.button['sat'] = QtWidgets.QCheckBox('Match S channel')
        self.button['sat'].clicked.connect(self.sat_changed)
        self.button['sat'].setChecked(False)
        if not self.is_color:
            self.button['sat'].hide()
        layout_region.addRow(self.button['sat'])

    # Functions
    def main_work(self):
        # Get Args
        grid_size_x = self.spin_box['size_x'].value()
        grid_size_y = self.spin_box['size_y'].value()
        clip_limit = self.spin_box['limit'].value()
        grid_size2_x = self.spin_box['size2_x'].value()
        grid_size2_y = self.spin_box['size2_y'].value()
        clip_limit2 = self.spin_box['limit2'].value()
        # CLAHE
        # -V Channel
        if grid_size_y and grid_size_x:
            model = cv2.createCLAHE(clip_limit, (grid_size_y, grid_size_x))
        else:
            model = cv2.createCLAHE(clip_limit)
        v = model.apply(self.original_image)
        if not self.is_color:
            return v
        # -S Channel
        if self.button['sat'].isChecked():
            if grid_size2_y and grid_size2_x:
                model = cv2.createCLAHE(clip_limit2, (grid_size2_y, grid_size2_x))
            else:
                model = cv2.createCLAHE(clip_limit2)
            s = model.apply(self.s)
        else:
            s = self.s
        return cv2.cvtColor(cv2.merge([self.h, s, v]), cv2.COLOR_HSV2RGB)

    def sat_changed(self, checked: bool):
        if not checked:
            for d in ('x', 'y'):
                self.widget[f'size2_{d}'].hide()
                self.spin_box[f'size2_{d}'].hide()
            self.widget['size2'].hide()
            self.widget['limit2'].hide()
            self.spin_box['limit2'].hide()
        else:
            for d in ('x', 'y'):
                self.widget[f'size2_{d}'].show()
                self.spin_box[f'size2_{d}'].show()
            self.widget['size2'].show()
            self.widget['limit2'].show()
            self.spin_box['limit2'].show()
        self.update_view()
