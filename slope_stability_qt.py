"""
Slope Stability Analysis — Standalone Qt Application

Limit equilibrium slope stability with Bishop, Spencer, and Fellenius
methods. Supports circular and noncircular slip surfaces.

Setup:
    1. Copy the GeotechStaffEngineer/ folder to your PC
    2. Run:  python slope_stability_qt.py

Requires: PyQt5 (or PySide2), numpy, scipy, matplotlib
"""

import sys
import os
import json
import datetime

# ---------------------------------------------------------------------------
# Source path
# ---------------------------------------------------------------------------
SOURCE_PATH = os.path.dirname(os.path.abspath(__file__))

if SOURCE_PATH not in sys.path:
    sys.path.insert(0, SOURCE_PATH)
    refs_path = os.path.join(SOURCE_PATH, "geotech-references")
    if os.path.isdir(refs_path) and refs_path not in sys.path:
        sys.path.insert(0, refs_path)

# ---------------------------------------------------------------------------
from qt_panels import APP_NAME, APP_VERSION, STYLESHEET
from qt_panels.slope_stability_panel import SlopeStabilityPanel
from qt_panels.common import (
    QApplication, QMainWindow, QStatusBar,
    QAction, QMessageBox, QFileDialog,
)


class SlopeStabilityWindow(QMainWindow):
    """Standalone slope stability analysis window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Slope Stability  -  {APP_NAME}  v{APP_VERSION}")
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

        # Central panel
        self.panel = SlopeStabilityPanel(self.status_bar)
        self.setCentralWidget(self.panel)

    def _save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results", "",
            "GeotechStaffEngineer Files (*.gse);;All Files (*)")
        if not path:
            return
        try:
            state = self.panel.get_state()
            doc = {
                "app": APP_NAME,
                "version": APP_VERSION,
                "tab": "Slope Stability",
                "inputs": state,
                "timestamp": datetime.datetime.now().isoformat(),
            }
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
            if "inputs" in doc:
                self.panel.set_state(doc["inputs"])
            self.status_bar.showMessage(f"Loaded from {path}", 10000)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))

    def _export_png(self):
        canvas = getattr(self.panel, "canvas", None)
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
            "About Slope Stability",
            f"<h2>Slope Stability Analysis</h2>"
            f"<p>{APP_NAME} v{APP_VERSION}</p>"
            f"<p>Limit equilibrium analysis using:</p>"
            f"<ul>"
            f"<li>Fellenius (Ordinary Method of Slices)</li>"
            f"<li>Bishop's Simplified Method</li>"
            f"<li>Spencer's Method (circular + noncircular)</li>"
            f"</ul>"
            f"<p>Circular and noncircular slip surface search with "
            f"entry/exit range constraints.</p>"
            f"<hr>"
            f"<p>References: Duncan, Wright & Brandon (2014)</p>"
            f"<p>&copy; 2026 Sean O'Connell</p>",
        )


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} - Slope Stability")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = SlopeStabilityWindow()
    window.show()

    if "SPY_PYTHONPATH" in os.environ or "JPY_PARENT_PID" in os.environ:
        return window
    else:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
