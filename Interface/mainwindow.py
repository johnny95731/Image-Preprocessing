import os
from typing import Optional, List, Union

import numpy as np
import cv2

import pyqtgraph as pg

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import QEvent

# import PySide6
# dirname = os.path.dirname(PySide6.__file__)
# plugin_path = os.path.join(dirname, "plugins", "platforms")
# os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

# from interface import UI
import spatial_dialog
import color_dialog
from image_object import Image, BasicOperators
from img_type import ARR_8U2D, IMG_8U



# class CustumViewBox(pg.ViewBox):
#     def raiseContextMenu(self, ev):
#         menu = self.getMenu(ev)
#         if menu is not None:
#             # Don't add 'Export' option.
#             # self.scene().addParentContextMenus(self, menu, ev)
#             menu.popup(ev.screenPos().toPoint())



class Mainwindow(QtWidgets.QMainWindow):
    open_directory = save_directory = os.getcwd()
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Image Preprocessing") # 標題
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        # 取得主螢幕解析度
        # The screen which cursor at.
        cursor_screen = QtGui.QGuiApplication.screenAt(QtGui.QCursor.pos())
        cursor_screen_geometry = cursor_screen.availableGeometry()
        
        ratio = np.divide(1, cursor_screen.devicePixelRatio())
        if ratio == 1:
            ratio = 0.8
        self.__height = int(cursor_screen_geometry.height() * ratio)
        self.__width = int(cursor_screen_geometry.width() * ratio)
        self.resize(self.__width, self.__height)
        # Align center
        y = (cursor_screen_geometry.height()-self.__height) // 2
        x = (cursor_screen_geometry.width()-self.__width) // 2
        self.move(cursor_screen_geometry.left()+x, y)
        
        # Central Widget and its Layout
        self.__central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.__central_widget)
        self.__central_layout = QtWidgets.QVBoxLayout(self.__central_widget)

        # 狀態欄（下方）
        self.__status_bar = self.statusBar()
        # 區域設定
        self.__create_main_menubar()
        self.__create_main_tab_bar()
        self.__create_main_image_view()
        self.__create_informations_frame()
        
        self.show()
        self.update_all()
    
    def __create_main_menubar(self):
        # The method `__create_main_menubar`, thgether with the object `events`,
        # will layout the menubar.
        # events is a list of dicts object.
        # These dicts obey the rules:
        #     1. If ("Action", text: str) is an item of dict. It means add an
        # QAction(text) to QMenu (or, QMenuBar). The other items is present for
        # setting attribute of, or, connecting signal to, the QAction.
        #     2. If (text, obj: list) is an item of dict. It means add an
        # QMenu(text) to QMenu (or, QMenuBar).
        #     3. The list "obj" in rule 2 is a list of dict obey rule 1 and 2. 
        events = [
            {"File": [
                {
                    "Action": "New", "triggered": self.__create_new_image, 
                    "shortcut": "Ctrl+N", "toolTip": "Create a new image."
                }, 
                {
                    "Action": "Open File(s)", "triggered": self.open_file, 
                    "shortcut": "Ctrl+O", "toolTip": "Read image(s).", 
                    "whats_this": "Loading Image(s)"
                }, 
                {"Standard Images":[*[
                    {
                        "Action": name,
                        "triggered": lambda e, n=name: self.__read_standard(
                            name=n)
                    } for name in Image.support_database
                ]]}, 
                {"Save": [
                    {
                        "Action": "Current", "triggered": self.save_file, 
                        "shortcut": "Ctrl+S"
                    }, 
                    # {
                    #     "Action": "All", "triggered": self.save_all_file,
                    #     "shortcut": ""
                    # }
                ]}
            ]}, 
            {"Edit": [
                {
                    "Action": "Undo", "triggered": self.__image_undo, 
                    "shortcut": "Ctrl+Z",
                    "toolTip": "Back to last step of this image."
                }, 
                {
                    "Action": "Redo", "triggered": self.__image_redo, 
                    "shortcut": "Ctrl+Y",
                    "toolTip": "Forward to next step of this image."
                }, 
                {
                    "Action": "Reset", "triggered": self.__reset_image,
                    "toolTip": "\n".join((
                        "Back to the original image.",
                        " Using \"Undo(Ctrl+Z)\" can back to last step."
                    ))
                },
                {
                    "Action": "Resize", "triggered": self.__image_resize,
                    "toolTip": "\n".join((
                        "Resize the image."
                    ))
                },
            ]}, 
            {"Colors": [
                {"Color Formats": [
                    {
                        "Action": color,
                        "triggered": lambda e, c=color: self.__color_format(c)
                    } for color in Image.support_color_formats]
                },
                {"Hist": [
                    {
                        "Action": "CLAHE",
                        "triggered": self.__clahe,
                        "toolTip": " ".join((
                        "Applying Contrast Limited Adaptive Histogram",
                        "Equalization to image."
                        ))
                    },
                    {
                        "Action": "Hist Equalize",
                        "triggered": self.__equalize_hist,
                        "toolTip": "Applying Histogram Equalization to image."
                    },
                    {
                        "Action": "Hist Matching",
                        "triggered": self.__match_hist,
                        "toolTip": "Applying Histogram Matching to image."
                    },
                ]},
                {"Intensity Trans.": [
                    {
                        "Action": "Auto Gamma Correction",
                        "triggered": self.__auto_gamma_correction,
                        "toolTip": "".join((
                            "Auto Gamma correction ",
                            "introduced by P. Babakhani1 and P. Zarei."
                        ))
                    },
                    {
                        "Action": "Function Trans.",
                        "triggered": self.__func_trans, "shortcut":"Ctrl+T",
                        "toolTip": "".join((
                            "Intensity transform by some simple functions.",
                            "\n",
                            "For examples, Linear Transformation, Gamma ",
                            "correction, Beta Correction, etc."
                        ))
                    },
                    {
                        "Action": "Piecewise Linear Trans.",
                        "triggered": self.__PL_trans,
                        "toolTip": 
                            "Intensity transform by piecewise linear function."
                    },
                ]},
                {"Segmentation": [
                    {
                        "Action": "k-Means", "triggered": self.__k_means, 
                    },
                    # {
                    #     "Action": "Superpixel", "triggered": self.__k_means, 
                    # },
                ]},
                {
                    "Action": "Slicing",
                    "triggered": self.__color_slicing,
                    "toolTip": "Slice specific range of color."
                },
            ]},  
            {"Spatial": [
                {
                    "Action": "Blur", "triggered": self.__blur, 
                },
                {
                    "Action": "Unsharp Mask", "triggered": self.__unsharp_mask, 
                },
                {"Edge": [
                    {
                        "Action": "Canny", "triggered": self.__canny, 
                    },
                    {
                        "Action": "Gradiant", "triggered": self.__apply_grad, 
                    },
                    {
                        "Action": "Marr-Hildreth",
                        "triggered": self.__marr_hildreth, 
                    },
                ]},
                {"Morphology": [
                    {
                        "Action": "Basic", "triggered": self.__basic_morph, 
                    },
                ]},
                {
                    "Action": "Noise", "triggered": self.__add_noise, 
                },
                {
                    "Action": "Sharpening",
                    "triggered": self.__apply_sharpening, 
                },
                {"Thresholding": [
                    {
                        "Action": "Global",
                        "triggered": self.__global_threshold, 
                    },
                    {
                        "Action": "Adaptive",
                        "triggered": self.__adaptive_threshold, 
                    },
                ]},
            ]},
            # {"Frequency": [
            #     {
            #         "Action": "Basic Filter", "triggered": self.__basic_filter, 
            #         "toolTip": "".join((
            #             "Low-pass, High-Pass, Band-Pass, and Band-Reject",
            #             "filters."
            #         ))
            #     }, 
            # ]},
            {"Preferences": [
                {
                    "Action": "Stats.",
                    "triggered": self.__show_information_frame,
                    "shortcut": "F12"
                },  
            ]},
        ]

        def create_nested_menu(parent_menu, childs):
            for child in childs:
                if ("Action" in child and
                    isinstance(child["Action"], str)):
                    name = child["Action"].replace(" ", "")
                    act: QtGui.QAction = (
                        parent_menu["menu"].addAction(child["Action"]))
                    if "shortcut" in child:
                        act.setShortcut(child["shortcut"])
                    if "toolTip" in child:
                        act.setToolTip(child["toolTip"])
                    act.triggered.connect(child["triggered"])
                    parent_menu[name] = act
                else:
                    for key in child:
                        child_menu = parent_menu[key] = {}
                        child_menu["menu"] = parent_menu["menu"].addMenu(key)
                        child_menu["menu"].setToolTipsVisible(True)
                        create_nested_menu(parent_menu[key], child[key])
            return parent_menu
        
        self.__menu = {"menu": self.menuBar()}
        create_nested_menu(self.__menu, events)
    
    def __create_main_tab_bar(self):
        # QTabBar and ist attributes
        self.__tab_bar = QtWidgets.QTabBar(parent=self.__central_widget)
        self.__central_layout.addWidget(self.__tab_bar)
        self.__tab_bar.setTabsClosable(True)
        self.__tab_bar.setMovable(True)
        # -Shotcut for change tab
        shortcut = QtGui.QShortcut('Ctrl+tab', self.__tab_bar)
        shortcut.activated.connect(self.__next_tab)
        # -Connect signal
        self.__tab_bar.tabCloseRequested.connect(self.__delete_tab)
        self.__tab_bar.currentChanged.connect(self.select_tab)
        self.__tab_bar.tabMoved.connect(self.__move_tab)
        # QTabBar.currentIndex. -1 when no tab exists.
        self.__tab_index: int = -1
    
    def __create_main_image_view(self):
        # Containers of info of images
        self.__img_objects: List[Image] = []
        
        # image view
        # -Set row-major (row, column) data representation.
        # -The default of pyqtgraph is column-major (column, row).
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.__image_view = pg.ImageView(
            self.__central_widget,
            # view=CustumViewBox() 
        )
        self.__histogram = self.__image_view.ui.histogram # Histogram item
        self.__central_layout.addWidget(self.__image_view)
        # -Hide buttons below histogram.
        self.__image_view.ui.roiBtn.hide()
        self.__image_view.ui.menuBtn.hide()
        # -Disable zooming with mouse wheel.
        # self.__image_view.getView().setMouseEnabled(x=False, y=False)
        
        view = self.__image_view.ui.graphicsView # Image display region
        # -Remove 'export' options.
        del view.sceneObj.contextMenu[:]
        del self.__histogram.sceneObj.contextMenu[:]
        # -Remove other option
        self.__image_view.view.menu.removeAction( 
            self.__image_view.view.menu.actions()[3] # Mose Mode
        )
        for act in self.__histogram.vb.menu.actions()[1:]:
            # Remains 'View all'.
            self.__histogram.vb.menu.removeAction(act)
        # Mouse move event
        self.__image_view.scene.sigMouseMoved.connect(self.mouse_moved)
    
    def __create_informations_frame(self):
        # Main frame of informations
        widgets = self.__info_frame = {}
        widgets["parent"] = QtWidgets.QDockWidget("Informations",
                                                  self.__central_widget)
        widgets["parent"].setFloating(False)
        widgets["parent"].setMinimumWidth(170)
        
        central = QtWidgets.QWidget(widgets["parent"])
        widgets["parent"].setWidget(central)
        # Place QDockWidget at right side of Mainwindow
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, widgets["parent"])
        self.resizeDocks([widgets["parent"]], [230],
                         QtCore.Qt.Orientation.Horizontal)
        
        # Information display widgets
        tree_view = widgets["tree_view"] = QtWidgets.QTreeView(central)
        tree_view.move(5, 5)
        # -Its model
        model = self.__info_model = {}
        model["model"] = QtGui.QStandardItemModel(tree_view)
        model["model"].setHorizontalHeaderLabels(["Name", "Value"])
        
        # Image size
        model["size_label"] = QtGui.QStandardItem("Size")
        model["model"].appendRow(model["size_label"])
        model["size"] = QtGui.QStandardItem("0")
        model["model"].setItem(0, 1, model["size"])
        
        # Brightness statistics
        model["brightness"] = QtGui.QStandardItem("Brightness")
        model["model"].appendRow(model["brightness"])
        
        labels = BasicOperators.stats_labels
        for index, label in enumerate(labels):
            # Set label
            model[f"{label}_label"] = QtGui.QStandardItem(label)
            model["brightness"].appendRow(model[f"{label}_label"])
            # Set value
            model[f"{label}"] = QtGui.QStandardItem("0")
            model["brightness"].setChild(index, 1, model[f"{label}"])
            
        tree_view.setModel(model["model"])
        tree_view.expandAll()
        
        # Signals connect
        def information_frame_resize_event(e):
            # QDockWidget resize
            w = e.size().width() - 10
            h = e.size().height() - 10
            # self.informationFrame["histPlot"].resize(w, 130)
            tree_view.resize(w, h)
        central.resizeEvent = information_frame_resize_event
    
    # Basic function
    # -Image view
    def update_all(self, img: Optional[Union[IMG_8U, Image]] = None):
        if img is None:
            if not self.__img_objects:
                img = np.random.normal(127.5, 50, (480, 640)).astype(np.uint8)
            else:
                img = self.__img_objects[self.__tab_index]
        self.__update_imformation_tree(img)
        if isinstance(img, Image):
            img = img.image
        self.__update_view(img)
        
    def __update_view(self, img: IMG_8U):
        self.__image_view.setImage(img, levels=(0, 255))
        # setImage will update levelMode before updating image. Hence,
        # setImage(img, levels=(0, 255), levelMode="rgba") raise error when
        # substituting color image for grayscale image.
        # 'setLevelMode' should be called individually.
        if img.ndim == 2:
            self.__histogram.setLevelMode("mono")
        else:
            self.__histogram.setLevelMode("rgba")
        self.__image_view.autoRange()
        self.__histogram.vb.autoRange()

    def __update_imformation_tree(self, img: Union[IMG_8U, Image]):
        if isinstance(img, Image):
            # The Image object will compute statistics when Image.image be
            # updated.
            vals = img.brightness_statistics
            img = img.image
            self.__info_model["size"].setText(f"{img.shape}")
        else:
            self.__info_model["size"].setText(f"{img.shape}")
            if img.ndim == 3:
                img: ARR_8U2D = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            vals = Image.get_statistics(img)
        
        labels = BasicOperators.stats_labels
        # Standardize float value length.
        vals = [
            # Max, Min, and Median are uint8 numbers.
            f"{v:#.5g}" if index < 5 else str(v)
            for index, v in enumerate(vals)
        ]
        for label, val in zip(labels, vals):
            self.__info_model[label].setText(val)
    
    # Get and set current image
    @property
    def current_image(self) -> Optional[IMG_8U]:
        return self.__image_view.image
    
    @property
    def image(self) -> Optional[IMG_8U]:
        if self.__tab_index == -1:
            return None
        return self.__img_objects[self.__tab_index].image
    
    @image.setter
    def image(self, new_img: IMG_8U):
        self.__img_objects[self.__tab_index].image = new_img
        self.update_all(new_img)
    
    # -Tab bar event
    def __next_tab(self) -> None:
        """Use Ctrl+Tab to change to next tab."""
        if not self.__img_objects:
            return None
        self.select_tab((self.__tab_index+1) % len(self.__img_objects))
        
    def select_tab(self, index: int) -> None:
        self.__tab_index = index
        self.__tab_bar.setCurrentIndex(index)
        self.update_all()
    
    def __delete_tab(self, index: int) -> None:
        # Delete objects
        self.__tab_bar.removeTab(index)
        self.__img_objects.pop(index)
        self.__tab_index = self.__tab_bar.currentIndex()
        # Update view and tree
        self.update_all()
        
    def __move_tab(self, _from: int, _to: int) -> None:
        self.__img_objects[_from], self.__img_objects[_to] = (
            self.__img_objects[_to], self.__img_objects[_from]
        )
    
    def add_new_tab(self, img: Union[IMG_8U, Image],
                    name: str,
                    file_extension: str = ".jpeg",
                    path: Optional[str] = None
        ) -> None:
        # Create Image object and append to __img_objects.
        if isinstance(img, Image):
            self.__img_objects.append(img)
        elif isinstance(img, np.ndarray):
            self.__img_objects.append(
                Image(img, name, file_extension, path)
            )
        else:
            print("Error. type(img) must be Image or np.ndarray")
            return
        # Add a tab to QTabBar. Emit currentChanged signal if no tab exists.
        self.__tab_bar.addTab(name)
        
    def select_last_tab(self):
        self.select_tab(self.__tab_bar.count()-1)
    
    # others
    def mouse_moved(self, view_pos):
        """When mouse on image view."""
        item = self.__image_view.getImageItem()
        img = item.image
        scene_pos = item.mapFromScene(view_pos)
        
        size = img.shape
        row, col = int(scene_pos.y()), int(scene_pos.x())
        if (0 <= row < size[0]) and (0 <= col < size[1]):
            value = img[row, col]
            if img.ndim == 3:
                self.__status_bar.showMessage(
                    f"(x,y)=({col},{row}), (R,G,B)={value}"
                )
            else:
                self.__status_bar.showMessage(
                    f"(x,y)=({col},{row}), Gray={value}"
                )
    
    def filename_repeat(self, filename: str) -> str:
        """
        Check whether filename in __img_names and return a new filename with
        numbering. For example, if \"new file\" contains in __img_names but
        \"new file (1)\" does not, then return \"new file (1)\."
        """
        original = str(filename)
        index = 1
        img_names = [obj.name for obj in self.__img_objects]
        while filename in img_names:
            filename = " ".join([original, f"({index})"])
            index += 1
        return filename
    
    def lock_tab_bar(self, lock: bool = False):
        self.__tab_bar.setDisabled(lock)
    
    def closeEvent(self, e: QEvent):
        """Override of QMainWindow.closeEvent."""
        reply = QtWidgets.QMessageBox.question(
            self, "Close", "Close the Window?",
            QtWidgets.QMessageBox.StandardButton.Yes | \
                QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No)
        if reply == QtWidgets.QMessageBox.StandardButton.Yes.value:
            e.accept()
            super().closeEvent(e)
        else:
            e.ignore()
    
    # Menu bar actions
    # File
    def __create_new_image(self):
        from file_dialog import NewFileDialog
        NewFileDialog(self)
    
    @classmethod
    def get_paths(cls, parent=None, action: str = "open"):
        formats = " ".join(Image.support_file_extensions)
        filter = f"Images ({formats})" # 可讀取之格式
        if action == "open":
            paths = QtWidgets.QFileDialog.getOpenFileNames(
                parent, "Open Files", cls.open_directory, filter=filter)[0]
        elif action == "save":
            paths = QtWidgets.QFileDialog.getSaveFileName(
                parent, "Save File", cls.save_directory, filter=filter)[0]
        elif action == "path":
            paths = QtWidgets.QFileDialog.getExistingDirectory(
                parent, "Path", cls.save_directory)
        else:
            print(f"action must be \"open\" or \"save\", not {action}")
            return ()
        return paths
    
    def open_file(self) -> None:
        # Read path(s) (including filename).
        paths = Mainwindow.get_paths(self, "open")
        if not paths: # no file be selected.
            return
        # Update default directory.
        Mainwindow.open_directory = paths[0][:paths[0].rfind("/")+1]
        # Deal path
        for path in paths:
            last_slash = path.rfind("/") + 1
            
            file_name = self.filename_repeat(path[last_slash:])
            last_dot = file_name.rindex(".")
            file_extension = file_name[last_dot:]
            file_name = file_name[:last_dot]
            # Read Image
            img = Image.read_image(path)
            self.add_new_tab(img,
                             file_name, file_extension,
                             path=path[:last_slash])
        self.select_last_tab()
        
    def save_file(self):
        if not self.__img_objects:
            return
        from file_dialog import SaveFileDialog
        SaveFileDialog(self, self.__img_objects[self.__tab_index])
    
    def save_all_file(self):
        if not self.__img_objects:
            return
        path = Mainwindow.get_paths(self, "path")
        if not path:
            return
        Mainwindow.save_directory = path
        for i, img in enumerate(self.__img_objects):
            img.save_image(path)
    
    def __read_standard(self, name: str) -> None:
        from skimage import data
        img = eval(f"data.{name}()")
        if img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if img.dtype == np.uint16:
            # 2**16-1 / 2**8-1 = 257
            img = np.divide(img, 257, dtype=np.float32)
            img = np.around(img).astype(np.uint8)
        elif img.dtype != np.uint8:
            img *= 255
            img = np.around(img).astype(np.uint8)
        self.add_new_tab(img, name)
        self.select_last_tab()
    
    # Edit
    def __reset_image(self):
        if self.__tab_index > -1:
            self.__img_objects[self.__tab_index].reset()
            self.update_all()
        
    def __image_undo(self):
        if self.__tab_index > -1:
            self.__img_objects[self.__tab_index].undo()
            self.update_all()
        
    def __image_redo(self):
        if self.__tab_index > -1:
            self.__img_objects[self.__tab_index].redo()
            self.update_all()
        
    def __image_resize(self):
        if not self.__img_objects:
            return
        from editting_dialog import ResizeDialog
        ResizeDialog(self)
            
    # Color
    def __color_format(self, color):
        if not self.__img_objects:
            return
        if color == "RGB888":
            self.__img_objects[self.__tab_index].to_color()
        elif color == "Grayscale8":
            self.__img_objects[self.__tab_index].to_grayscale()
        self.update_all()
    
    # - Contract Adjust
    def __func_trans(self):
        """Intensity transformation by some common function."""
        if not self.__img_objects:
            return
        color_dialog.FuncTransDialog(self)
    def __PL_trans(self):
        """Piecewise linear transformation."""
        if not self.__img_objects:
            return
        color_dialog.PiecewiseLinearDialog(self)
    def __auto_gamma_correction(self):
        """Auto gamma_correction introduced by P. Babakhani1 and P. Zarei."""
        if not self.__img_objects:
            return
        self.lock_tab_bar(True)
        self.__img_objects[self.__tab_index].auto_gamma_correction_PB_PZ()
        self.update_all()
        self.lock_tab_bar(False)
        
    def __color_slicing(self):
        """Slice specific range of color."""
        if not self.__img_objects:
            return
        color_dialog.SlicingDialog(self)
    
    def __clahe(self):
        """Histogram Equalization."""
        if not self.__img_objects:
            return
        color_dialog.CLAHEDialog(self)
    
    def __equalize_hist(self):
        """Histogram Equalization."""
        if not self.__img_objects:
            return
        self.lock_tab_bar(True)
        self.__img_objects[self.__tab_index].equalize_hist()
        self.update_all()
        self.lock_tab_bar(False)

    def __k_means(self):
        """Slice specific range of color."""
        if not self.__img_objects:
            return
        color_dialog.kMeansDialog(self)

    def __match_hist(self):
        """Histogram matching."""
        if not self.__img_objects:
            return
        color_dialog.HistogramMatchingDialog(self)

    # Spatial
    def __blur(self):
        if not self.__img_objects:
            return
        spatial_dialog.BlurDialog(self)

    # -Edge
    def __canny(self):
        if not self.__img_objects:
            return
        spatial_dialog.CannyDialog(self)
    def __apply_grad(self):
        if not self.__img_objects:
            return
        spatial_dialog.GradientDialog(self)
    def __marr_hildreth(self):
        if not self.__img_objects:
            return
        spatial_dialog.MarrHildrethDialog(self)
    
    # -Morphology
    def __basic_morph(self):
        if not self.__img_objects:
            return
        spatial_dialog.BasicMorphologyDialog(self)
    
    def __add_noise(self):
        if not self.__img_objects:
            return
        spatial_dialog.NoiseDialog(self, "add")
    
    def __apply_sharpening(self):
        if not self.__img_objects:
            return
        spatial_dialog.GradientDialog(self, 1)
    
    def __unsharp_mask(self):
        if not self.__img_objects:
            return
        spatial_dialog.BlurDialog(self, 1)
    
    # -Thresholding
    def __global_threshold(self):
        if not self.__img_objects:
            return
        spatial_dialog.ThresholdDialog(self)
    def __adaptive_threshold(self):
        if not self.__img_objects:
            return
        spatial_dialog.AdaptiveThresholdDialog(self)
    
    # Frequency
    def __basic_filter(self):
        if not self.__img_objects:
            return
        from frequency_dialog import BasicFilterDialog
        BasicFilterDialog(self)
    
    # Preferences
    def __show_information_frame(self):
        self.__info_frame["parent"].setVisible(
            not self.__info_frame["parent"].isVisible()
        )
    