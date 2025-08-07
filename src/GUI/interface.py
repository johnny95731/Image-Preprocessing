from typing import Literal

import numpy as np
import numpy.typing as npt

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtWidgets import QDialog, QWidget, QComboBox, QMainWindow
from PyQt6.QtGui import QFont
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from src.utils.img_type import IMG_8U


mpl.use('QtAgg')
# mpl.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Arial",
#     'text.latex.preamble': [r'\usepackage{amsmath}'],
#     "mathtext.fontset": "stixsans",
# })


class MplCanvas(FigureCanvasQTAgg):
    # Return canvas and axis directly.
    # def __new__(cls, *args, **kwargs):
    #     inst = super().__new__(cls, *args, **kwargs)

    #     size = kwargs.get("size", (4,4))
    #     dpi = kwargs.get("dpi", 100)
    #     visible = kwargs.get("visible", False)
    #     figure = Figure()
    #     figure.set_figheight(size[0])
    #     figure.set_figwidth(size[1])
    #     figure.set_dpi(dpi)
    #     figure.tight_layout()
    #     figure.patch.set_visible(visible)

    #     super(MplCanvas, inst).__init__(figure)
    #     parent = kwargs.get("parent", None)
    #     inst.setParent(parent)

    #     inst.ax = figure.add_axes([0,0,1,1], frameon=False)
    #     inst.ax.axis('off')
    #     inst.ax.get_xaxis().set_visible(False)
    #     inst.ax.get_yaxis().set_visible(False)

    #     inst.updateGeometry()
    #     return inst, inst.ax

    def __init__(
        self,
        parent: QtWidgets.QWidget = None,
        size: tuple[int, int] | None = None,
        dpi: int = 100,
        scale: Literal['both', 'x', 'y'] = 'both',
        visible: bool = False,
    ):
        figure = Figure()
        if size is not None:
            figure.set_figheight(size[0])
            figure.set_figwidth(size[1])
        if dpi > 0:
            figure.set_dpi(dpi)
        figure.tight_layout()
        figure.patch.set_visible(visible)

        super(MplCanvas, self).__init__(figure)
        self.setParent(parent)

        self.ax = figure.add_axes([0, 0, 1, 1], frameon=False)
        self.ax.axis('off')
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.autoscale(axis=scale, tight=True)
        self.updateGeometry()

    @property
    def widget(self):
        return self, self.ax

    @staticmethod
    def latex(text: str, font_size: float = 12, font_color: str = 'w') -> QtGui.QPixmap:
        """Text rendering with LaTeX."""
        # Create mpl Figure object
        fig = Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvasQTAgg(fig))
        renderer = fig.canvas.get_renderer()

        # Plot the mathTex expression
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(
            0, 0, f'{text}', ha='left', va='bottom', fontsize=font_size, c=font_color
        )

        # Fit figure size to text artist
        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)
        text_bbox = t.get_window_extent(renderer)

        tight_fig_width = fwidth * (text_bbox.width / fig_bbox.width)
        tight_fig_height = fheight * (text_bbox.height / fig_bbox.height)
        fig.set_size_inches(tight_fig_width, tight_fig_height)
        # convert mpl figure to QPixmap
        buf, size = fig.canvas.print_to_buffer()
        qimage = QtGui.QImage.rgbSwapped(
            QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format_ARGB32)
        )
        return QtGui.QPixmap(qimage)


class BasicDialog(QDialog):
    """
    Basic dialog. Contains common methods. Most dialog that deal image
    processing will inherit this class.

    Parameters
    ------------
    parent : Mainwindow
        An instance of Mainwindow which defined in mainwindow.py.
    title : str
        Title of dialog.


    Attributes
    -----------
    original_image : ndarray
        The image in mainwindow.
    channels : tuple[str]
        Index of dictionary. ("Red", "Green", "Blue") if original_image is rgb.
        ("Red", ) if original_image is grayscale.
    is_color : bool
        False if original_image is 2-d array. Otherwise, True.
    fonts : tuple[QFont]
        Fonts for specified typography.
    combo_box : dict[str, QComboBox]
        Dictionary for combo box.
    spin_box : dict[str, QAbstractSpinBox]
        Dictionary for spin nox.
    button : dict[str, QAbstractButton]
        Dictionary for buttons.
    widget : dict[str, QWidget]
        The other widgets.
    """

    def __init__(self, parent: QMainWindow, title: str = 'Dialog', **kwargs):
        # Disable QTabBar in Mainwindow untill close dialog.
        parent.lock_tab_bar(True)

        super().__init__(parent)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WA_DeleteOnClose)  # Delete after close
        # self.setWindowModality(Qt.ApplicationModal) # Block Mainwindow
        self.central_layout = QtWidgets.QVBoxLayout(self)
        self.central_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # minimum_size: tuple[int,int]=(width, height)
        if 'minimum_size' in kwargs:
            self.setMinimumSize(*kwargs['minimum_size'])

        self.original_image: npt.NDArray[np.uint8] = parent.image
        if self.original_image is not None:
            # Attributes about image
            if self.original_image.ndim == 2:
                self.channels = ('Red',)
                self.is_color = False
            elif self.original_image.ndim == 3:
                self.channels = ('Red', 'Green', 'Blue')
                self.is_color = True

        # Fonts
        # -weight: 400=Normal, 800=Bold
        self.fonts: dict[str, QFont] = {
            'title': QFont('Arial', pointSize=12, weight=800),
            'subtitle': QFont('Arial', pointSize=10, weight=600),
            'default': QFont('Arial', pointSize=9, weight=400),
        }

        # Widget containers
        self.widget = {}
        self.combo_box: dict[str, QComboBox] = {}
        self.spin_box: dict[str, QtWidgets.QAbstractSpinBox] = {}
        self.button: dict[str, QtWidgets.QAbstractButton] = {}

        self.closeEvent = self.close_dialog

    def create_basic_buttons(self):
        # Buttons
        region = QWidget()
        self.central_layout.addWidget(region)
        layout_region = QtWidgets.QGridLayout(region)

        # Show effect or not.
        self.button['show'] = QtWidgets.QCheckBox('Preview')
        layout_region.addWidget(self.button['show'], 0, 0)
        self.button['show'].clicked.connect(self.show_effect)
        self.button['show'].setChecked(True)

        # -Update, confirm, and cancel
        for i, key in enumerate(['update', 'confirm', 'cancel']):
            text = key.capitalize()
            self.button[key] = QtWidgets.QPushButton(text)
            layout_region.addWidget(self.button[key], 1, i)
        self.button['update'].setShortcut('enter')
        self.button['update'].clicked.connect(self.update_view)
        self.button['confirm'].clicked.connect(self.confirm)
        self.button['cancel'].clicked.connect(self.close_dialog)

    def main_work(self) -> IMG_8U:
        """The main work of this dialog present for."""
        return np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)

    def update_view(self, e=None):
        if self.button['show'].isChecked():
            img = self.main_work()
            self.parentWidget().update_all(img)
        else:
            self.temp = self.main_work()

    def confirm(self):
        if not self.button['show'].isChecked():
            self.show_effect(True)
        img = self.parentWidget().current_image
        self.parentWidget().image = img
        self.close_dialog()

    def close_dialog(self, e=None):
        import gc

        self.parentWidget().lock_tab_bar(False)
        self.parentWidget().update_all()
        self.close()
        gc.collect(2)

    def show_effect(self, checked: bool):
        """Show the preview of main_work effect or show original image."""
        if not checked:
            self.temp = self.parentWidget().current_image
            self.parentWidget().update_all()
        else:
            self.parentWidget().update_all(self.temp)

    def filename_tooltip():
        return ''.join([
            'filename must be nonempty.',
            'Filename can not involve the following characters:',
            '\\ / : * ? " | < >',
        ])

    def filename_reg_exp():
        """Filename regular expression"""
        return QtCore.QRegularExpression('^[^\\/:*?"\|<>]+$')


class LabelThread(QThread):
    """Update labels."""

    def __init__(self, parent):
        super().__init__(parent)

    def run(self):
        self.parent().update_args_label()
