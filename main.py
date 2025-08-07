from sys import argv

from qdarkstyle import load_stylesheet_pyqt5
from PyQt6.QtWidgets import QApplication

from src.GUI.mainwindow import Mainwindow

if __name__ == '__main__':
    app = QApplication(argv)
    app.setStyleSheet(load_stylesheet_pyqt5())
    win = Mainwindow()
    win.activateWindow()  # 提醒（windows工具列圖示閃爍）
    win.raise_()
    exit(app.exec())
