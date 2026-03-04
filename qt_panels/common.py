"""Shared Qt widgets, matplotlib canvas, and constants for all panels."""

import numpy as np

# ---------------------------------------------------------------------------
# Qt imports — try PyQt5 first (Spyder / Anaconda), then PySide2
# ---------------------------------------------------------------------------
try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
        QHBoxLayout, QFormLayout, QGroupBox, QLabel, QDoubleSpinBox,
        QSpinBox, QComboBox, QPushButton, QTextEdit, QSplitter,
        QStatusBar, QAction, QMessageBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QCheckBox, QFrame, QSizePolicy, QScrollArea,
        QFileDialog, QStackedWidget, QDialog, QDialogButtonBox,
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
except ImportError:
    from PySide2.QtWidgets import (
        QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
        QHBoxLayout, QFormLayout, QGroupBox, QLabel, QDoubleSpinBox,
        QSpinBox, QComboBox, QPushButton, QTextEdit, QSplitter,
        QStatusBar, QAction, QMessageBox, QTableWidget, QTableWidgetItem,
        QHeaderView, QCheckBox, QFrame, QSizePolicy, QScrollArea,
        QFileDialog, QStackedWidget, QDialog, QDialogButtonBox,
    )
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QFont

# ---------------------------------------------------------------------------
# Matplotlib Qt backend
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT,
)
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# App-wide constants
# ---------------------------------------------------------------------------
APP_NAME = "GeotechStaffEngineer"
APP_VERSION = "3.2.0"

LAYER_COLORS = [
    "#e6d4a0", "#d4a574", "#a0c4a0", "#c8b4a0", "#b0d0e8",
    "#f0c0c0", "#c4c4e0", "#d0e8a0",
]
GWT_COLOR = "#2196F3"
SLIP_COLOR = "#D32F2F"
SURFACE_COLOR = "#2E7D32"

MESH_PRESETS = {
    "Very Coarse": (5, 5),
    "Coarse": (8, 8),
    "Medium": (12, 12),
    "Fine": (18, 18),
    "Very Fine": (25, 25),
}

STYLESHEET = """
QMainWindow {
    background-color: #f5f5f5;
}
QTabWidget::pane {
    border: 1px solid #ccc;
    background: white;
}
QTabBar::tab {
    background: #e0e0e0;
    padding: 8px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background: white;
    border-bottom: 2px solid #1976D2;
}
QGroupBox {
    font-weight: bold;
    border: 1px solid #ccc;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
}
QPushButton {
    background-color: #1976D2;
    color: white;
    border: none;
    padding: 8px 24px;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #1565C0;
}
QPushButton:pressed {
    background-color: #0D47A1;
}
QPushButton#secondary {
    background-color: #757575;
}
QPushButton#secondary:hover {
    background-color: #616161;
}
QTextEdit#results {
    background-color: #263238;
    color: #E0E0E0;
    font-family: "Consolas", "Courier New", monospace;
    font-size: 12px;
    border: 1px solid #37474F;
    border-radius: 4px;
    padding: 8px;
}
QStatusBar {
    background-color: #e0e0e0;
    font-size: 12px;
}
QTableWidget {
    gridline-color: #ddd;
    font-size: 12px;
}
QTableWidget::item {
    padding: 2px 6px;
}
"""


# ---------------------------------------------------------------------------
# Matplotlib canvas widget
# ---------------------------------------------------------------------------
class MplCanvas(FigureCanvasQTAgg):
    """Embeddable matplotlib canvas for Qt."""

    def __init__(self, parent=None, width=8, height=5):
        self.fig = Figure(figsize=(width, height), dpi=100,
                          facecolor="white", tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)


# ---------------------------------------------------------------------------
# Helper: styled results text box
# ---------------------------------------------------------------------------
def make_results_box():
    te = QTextEdit()
    te.setReadOnly(True)
    te.setObjectName("results")
    te.setMinimumHeight(120)
    return te
