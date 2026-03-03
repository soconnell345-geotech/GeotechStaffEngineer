"""
GeotechStaffEngineer — Desktop Qt Application

Portable geotechnical analysis GUI.  Copy this file + the project source
to any PC with Python + PyQt5/PySide2 + numpy + scipy + matplotlib.

Setup on work PC:
    1. Copy the GeotechStaffEngineer/ folder to your PC
    2. Edit SOURCE_PATH below if running from a different location
    3. Run:  python geotech_qt_gui.py
       Or:   open in Spyder and press F5

Requires: PyQt5 (or PySide2), numpy, scipy, matplotlib
          (all included in Anaconda)
"""

import sys
import os
import json
import datetime

# ---------------------------------------------------------------------------
# Source path — edit this if the project lives somewhere else on your PC
# ---------------------------------------------------------------------------
SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))
# SOURCE_PATH = r"C:\Users\YourName\Documents\GeotechStaffEngineer"

if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)
    refs_path = os.path.join(SOURCE_PATH, "geotech-references")
    if os.path.isdir(refs_path) and refs_path not in sys.path:
        sys.path.insert(0, refs_path)

# ---------------------------------------------------------------------------
# Qt panels package — all panels, constants, and widgets
# ---------------------------------------------------------------------------
from qt_panels import (
    APP_NAME, APP_VERSION, STYLESHEET,
    BearingCapacityPanel, SlopeStabilityPanel,
    SettlementPanel, FEM2DPanel,
)
from qt_panels.common import (
    QApplication, QMainWindow, QTabWidget, QStatusBar,
    QAction, QMessageBox, QFileDialog,
)


# ===========================================================================
# MAIN WINDOW
# ===========================================================================
class MainWindow(QMainWindow):
    """Main application window with tabbed analysis panels."""

    TAB_NAMES = ["Bearing Capacity", "Slope Stability", "Settlement", "FEM 2D"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        self.resize(1400, 900)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Menu bar
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        save_action = QAction("Save Results...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_project)
        file_menu.addAction(save_action)
        load_action = QAction("Load Results...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_project)
        file_menu.addAction(load_action)
        export_action = QAction("Export Plot as PNG...", self)
        export_action.triggered.connect(self._export_png)
        file_menu.addAction(export_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = menu.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(
            BearingCapacityPanel(self.status_bar), "Bearing Capacity")
        self.tabs.addTab(
            SlopeStabilityPanel(self.status_bar), "Slope Stability")
        self.tabs.addTab(
            SettlementPanel(self.status_bar), "Settlement")
        self.tabs.addTab(
            FEM2DPanel(self.status_bar), "FEM 2D")
        self.setCentralWidget(self.tabs)

    def _current_panel(self):
        return self.tabs.currentWidget()

    def _save_project(self):
        panel = self._current_panel()
        if not hasattr(panel, "get_state"):
            self.status_bar.showMessage("Save not supported for this tab")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "",
            "GeotechStaffEngineer Files (*.gse);;All Files (*)")
        if not path:
            return
        try:
            state = panel.get_state()
            tab_name = self.tabs.tabText(self.tabs.currentIndex())
            doc = {
                "app": APP_NAME,
                "version": APP_VERSION,
                "tab": tab_name,
                "inputs": state,
                "timestamp": datetime.datetime.now().isoformat(),
            }
            # Add analysis_type for FEM2D
            if isinstance(panel, FEM2DPanel):
                doc["analysis_type"] = panel.type_cb.currentText()
            with open(path, "w") as f:
                json.dump(doc, f, indent=2, default=str)
            self.status_bar.showMessage(f"Saved to {path}", 10000)
        except Exception as e:
            QMessageBox.warning(self, "Save Error", str(e))

    def _load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Results", "",
            "GeotechStaffEngineer Files (*.gse);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                doc = json.load(f)
            tab_name = doc.get("tab", "")
            # Switch to the correct tab
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == tab_name:
                    self.tabs.setCurrentIndex(i)
                    break
            panel = self._current_panel()
            if hasattr(panel, "set_state") and "inputs" in doc:
                panel.set_state(doc["inputs"])
            self.status_bar.showMessage(f"Loaded from {path}", 10000)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _export_png(self):
        panel = self._current_panel()
        canvas = getattr(panel, "canvas", None)
        if canvas is None:
            self.status_bar.showMessage("No plot to export")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "",
            "PNG Images (*.png);;All Files (*)")
        if not path:
            return
        try:
            canvas.fig.savefig(path, dpi=150, bbox_inches="tight")
            self.status_bar.showMessage(f"Plot exported to {path}", 10000)
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p>Version {APP_VERSION}</p>"
            f"<p>Python toolkit for geotechnical engineering analysis.</p>"
            f"<p>37 analysis modules covering foundations, piles, slopes, "
            f"seismic, FEM, and more.</p>"
            f"<p><b>Modules available:</b> bearing_capacity, settlement, "
            f"slope_stability, axial_pile, lateral_pile, sheet_pile, soe, "
            f"pile_group, drilled_shaft, retaining_walls, ground_improvement, "
            f"seismic_geotech, fem2d, wind_loads, ...</p>"
            f"<hr>"
            f"<p>&copy; 2026 Sean O'Connell</p>",
        )


# ===========================================================================
# Entry point
# ===========================================================================
def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    # If running inside Spyder or Jupyter, don't block with exec_()
    if "SPY_PYTHONPATH" in os.environ or "JPY_PARENT_PID" in os.environ:
        return window
    else:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
