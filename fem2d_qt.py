"""
FEM 2D Analysis — Standalone Qt Application

2D plane-strain finite element analysis with Mohr-Coulomb and Hardening Soil
constitutive models. Supports gravity, foundation, slope SRM, excavation,
seepage, and consolidation analyses.

Setup:
    1. Copy the GeotechStaffEngineer/ folder to your PC
    2. Run:  python fem2d_qt.py

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
from qt_panels.fem2d_panel import FEM2DPanel
from qt_panels.common import (
    QApplication, QMainWindow, QStatusBar,
    QAction, QMessageBox, QFileDialog,
)


class FEM2DWindow(QMainWindow):
    """Standalone FEM 2D analysis window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FEM 2D  -  {APP_NAME}  v{APP_VERSION}")
        self.resize(1500, 950)

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
        self.panel = FEM2DPanel(self.status_bar)
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
                "tab": "FEM 2D",
                "analysis_type": self.panel.type_cb.currentText(),
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
            "About FEM 2D",
            f"<h2>FEM 2D Analysis</h2>"
            f"<p>{APP_NAME} v{APP_VERSION}</p>"
            f"<p>2D plane-strain finite element analysis:</p>"
            f"<ul>"
            f"<li>CST and Q4 elements</li>"
            f"<li>Mohr-Coulomb and Hardening Soil models</li>"
            f"<li>Strength Reduction Method (SRM) for slopes</li>"
            f"<li>Braced excavation with beam walls and struts</li>"
            f"<li>Steady-state seepage (Laplace)</li>"
            f"<li>Coupled Biot consolidation</li>"
            f"</ul>"
            f"<p>Mesh preview with boundary condition visualization.</p>"
            f"<hr>"
            f"<p>References: Griffiths & Lane (1999), "
            f"Schanz, Vermeer & Bonnier (1999), "
            f"Smith & Griffiths (2004)</p>"
            f"<p>&copy; 2026 Sean O'Connell</p>",
        )


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(f"{APP_NAME} - FEM 2D")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = FEM2DWindow()
    window.show()

    if "SPY_PYTHONPATH" in os.environ or "JPY_PARENT_PID" in os.environ:
        return window
    else:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
