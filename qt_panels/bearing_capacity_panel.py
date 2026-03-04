"""Bearing capacity analysis panel."""

import math
import traceback
import numpy as np

from qt_panels.common import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QComboBox, QPushButton, QCheckBox,
    QScrollArea, QSplitter, Qt,
    MplCanvas, make_results_box, NavigationToolbar2QT,
    GWT_COLOR, SLIP_COLOR,
    mpatches,
)
from bearing_capacity import (
    Footing, SoilLayer, BearingSoilProfile, BearingCapacityAnalysis,
)


class BearingCapacityPanel(QWidget):
    """Shallow foundation bearing capacity analysis (Vesic/Meyerhof)."""

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: inputs (scrollable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        form_widget = QWidget()
        form = QVBoxLayout(form_widget)

        # Footing group
        ftg = QGroupBox("Footing")
        fl = QFormLayout()
        self.width_sb = QDoubleSpinBox(value=2.0, minimum=0.1, maximum=100,
                                       singleStep=0.5, decimals=2, suffix=" m")
        self.length_sb = QDoubleSpinBox(value=2.0, minimum=0.1, maximum=200,
                                        singleStep=0.5, decimals=2, suffix=" m")
        self.depth_sb = QDoubleSpinBox(value=1.5, minimum=0.0, maximum=50,
                                       singleStep=0.5, decimals=2, suffix=" m")
        self.shape_cb = QComboBox()
        self.shape_cb.addItems(["strip", "rectangular", "square", "circular"])
        self.shape_cb.setCurrentText("square")
        fl.addRow("Width B:", self.width_sb)
        fl.addRow("Length L:", self.length_sb)
        fl.addRow("Depth D:", self.depth_sb)
        fl.addRow("Shape:", self.shape_cb)
        ftg.setLayout(fl)
        form.addWidget(ftg)

        # Soil group
        sg = QGroupBox("Soil Properties")
        sl = QFormLayout()
        self.phi_sb = QDoubleSpinBox(value=30.0, minimum=0.0, maximum=50,
                                     singleStep=1.0, decimals=1, suffix=" deg")
        self.c_sb = QDoubleSpinBox(value=0.0, minimum=0.0, maximum=500,
                                   singleStep=5.0, decimals=1, suffix=" kPa")
        self.gamma_sb = QDoubleSpinBox(value=18.0, minimum=10.0, maximum=25,
                                       singleStep=0.5, decimals=1, suffix=" kN/m\u00b3")
        self.gwt_cb = QCheckBox("Groundwater present")
        self.gwt_depth_sb = QDoubleSpinBox(value=3.0, minimum=0.0, maximum=50,
                                           singleStep=0.5, decimals=1, suffix=" m")
        self.gwt_depth_sb.setEnabled(False)
        self.gwt_cb.toggled.connect(self.gwt_depth_sb.setEnabled)
        sl.addRow("Friction angle \u03c6:", self.phi_sb)
        sl.addRow("Cohesion c:", self.c_sb)
        sl.addRow("Unit weight \u03b3:", self.gamma_sb)
        sl.addRow("", self.gwt_cb)
        sl.addRow("GWT depth:", self.gwt_depth_sb)
        sg.setLayout(sl)
        form.addWidget(sg)

        # Analysis settings
        ag = QGroupBox("Analysis Settings")
        al = QFormLayout()
        self.fs_sb = QDoubleSpinBox(value=3.0, minimum=1.0, maximum=10,
                                    singleStep=0.5, decimals=1)
        self.method_cb = QComboBox()
        self.method_cb.addItems(["vesic", "meyerhof", "hansen"])
        al.addRow("Factor of Safety:", self.fs_sb)
        al.addRow("N\u03b3 Method:", self.method_cb)
        ag.setLayout(al)
        form.addWidget(ag)

        # Run button
        self.run_btn = QPushButton("Analyze")
        self.run_btn.clicked.connect(self._run_analysis)
        form.addWidget(self.run_btn)

        form.addStretch()
        scroll.setWidget(form_widget)
        outer.addWidget(scroll)

        # --- Right: plot + results ---
        right_splitter = QSplitter(Qt.Vertical)

        canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(canvas_widget)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(width=7, height=4)
        self.toolbar = NavigationToolbar2QT(self.canvas, canvas_widget)
        canvas_layout.addWidget(self.toolbar)
        canvas_layout.addWidget(self.canvas)
        right_splitter.addWidget(canvas_widget)

        self.results_text = make_results_box()
        right_splitter.addWidget(self.results_text)

        right_splitter.setSizes([400, 200])
        outer.addWidget(right_splitter, 1)

    def _run_analysis(self):
        try:
            shape = self.shape_cb.currentText()
            length = None
            if shape == "rectangular":
                length = self.length_sb.value()
            elif shape == "square":
                length = self.width_sb.value()
            elif shape == "circular":
                length = self.width_sb.value()

            footing = Footing(
                width=self.width_sb.value(),
                length=length,
                depth=self.depth_sb.value(),
                shape=shape,
            )

            gwt_depth = None
            if self.gwt_cb.isChecked():
                gwt_depth = self.gwt_depth_sb.value()

            soil = BearingSoilProfile(
                layer1=SoilLayer(
                    cohesion=self.c_sb.value(),
                    friction_angle=self.phi_sb.value(),
                    unit_weight=self.gamma_sb.value(),
                ),
                gwt_depth=gwt_depth,
            )

            analysis = BearingCapacityAnalysis(
                footing=footing,
                soil=soil,
                factor_of_safety=self.fs_sb.value(),
                ngamma_method=self.method_cb.currentText(),
            )
            result = analysis.compute()

            self.results_text.setText(result.summary())
            self._plot_result(result)
            self._status.showMessage(
                f"Bearing capacity: qult = {result.q_ultimate:,.0f} kPa, "
                f"qall = {result.q_allowable:,.0f} kPa", 10000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    def _plot_result(self, result):
        ax = self.canvas.axes
        ax.clear()

        B = self.width_sb.value()
        D = self.depth_sb.value()
        B_eff = result.B_eff

        # Draw soil
        ax.axhspan(-10, 0, color="#e6d4a0", alpha=0.4, label="Soil")
        ax.axhline(0, color="#8B4513", linewidth=2, label="Ground surface")

        # Groundwater
        if self.gwt_cb.isChecked():
            gwt = -self.gwt_depth_sb.value()
            ax.axhline(gwt, color=GWT_COLOR, linewidth=1.5, linestyle="--",
                        label=f"GWT ({self.gwt_depth_sb.value():.1f} m)")

        # Footing
        fx = -B / 2
        footing_rect = mpatches.FancyBboxPatch(
            (fx, -D), B, D, boxstyle="round,pad=0.02",
            facecolor="#757575", edgecolor="black", linewidth=2, zorder=5,
        )
        ax.add_patch(footing_rect)
        ax.text(0, -D / 2, f"B = {B:.1f} m\nD = {D:.1f} m",
                ha="center", va="center", fontweight="bold",
                fontsize=9, color="white", zorder=6)

        # Failure wedge (simplified Prandtl-type)
        depth_influence = B_eff * 1.5
        # Simplified log-spiral approximation
        theta = np.linspace(0, math.pi, 50)
        spiral_x = B_eff * np.cos(theta)
        spiral_y = -D - depth_influence * np.sin(theta)
        ax.fill(spiral_x, spiral_y, alpha=0.15, color=SLIP_COLOR,
                label="Failure zone")
        ax.plot(spiral_x, spiral_y, color=SLIP_COLOR, linewidth=1.5,
                linestyle="--")

        # Pressure arrows
        n_arrows = 5
        for i in range(n_arrows):
            xi = fx + B * (i + 0.5) / n_arrows
            arrow_len = 1.0
            ax.annotate("", xy=(xi, 0.05), xytext=(xi, 0.05 + arrow_len),
                         arrowprops=dict(arrowstyle="->", color="#D32F2F",
                                         lw=2))

        # Annotations
        ax.text(B / 2 + 0.5, -D / 2,
                f"q_ult = {result.q_ultimate:,.0f} kPa\n"
                f"q_all = {result.q_allowable:,.0f} kPa",
                fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#1976D2", alpha=0.9))

        # Term breakdown bar (small inset)
        terms = [result.term_cohesion, result.term_overburden,
                 result.term_selfweight]
        if sum(terms) > 0:
            left = -B / 2 - depth_influence
            total = sum(terms)
            bar_width = 2 * depth_influence + B
            y_bar = 2.5
            bar_h = 0.6
            colors = ["#4CAF50", "#FF9800", "#2196F3"]
            x_pos = left
            for t, c, _lb in zip(terms, colors,
                                  ["Cohesion", "Overburden", "Self-weight"]):
                w = bar_width * t / total
                ax.barh(y_bar, w, left=x_pos, height=bar_h,
                         color=c, edgecolor="white", linewidth=0.5)
                if t / total > 0.1:
                    ax.text(x_pos + w / 2, y_bar, f"{100 * t / total:.0f}%",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="white")
                x_pos += w
            ax.text(left - 0.3, y_bar, "Terms:", ha="right", va="center",
                    fontsize=8, fontweight="bold")

        ax.set_xlim(-B / 2 - depth_influence - 1, B / 2 + depth_influence + 3)
        ax.set_ylim(-D - depth_influence - 1, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Bearing Capacity Analysis", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def get_state(self):
        return {
            "width": self.width_sb.value(),
            "length": self.length_sb.value(),
            "depth": self.depth_sb.value(),
            "shape": self.shape_cb.currentText(),
            "phi": self.phi_sb.value(),
            "c": self.c_sb.value(),
            "gamma": self.gamma_sb.value(),
            "gwt_enabled": self.gwt_cb.isChecked(),
            "gwt_depth": self.gwt_depth_sb.value(),
            "fs": self.fs_sb.value(),
            "method": self.method_cb.currentText(),
            "results_text": self.results_text.toPlainText(),
        }

    def set_state(self, state):
        self.width_sb.setValue(state.get("width", 2.0))
        self.length_sb.setValue(state.get("length", 2.0))
        self.depth_sb.setValue(state.get("depth", 1.5))
        self.shape_cb.setCurrentText(state.get("shape", "square"))
        self.phi_sb.setValue(state.get("phi", 30.0))
        self.c_sb.setValue(state.get("c", 0.0))
        self.gamma_sb.setValue(state.get("gamma", 18.0))
        self.gwt_cb.setChecked(state.get("gwt_enabled", False))
        self.gwt_depth_sb.setValue(state.get("gwt_depth", 3.0))
        self.fs_sb.setValue(state.get("fs", 3.0))
        self.method_cb.setCurrentText(state.get("method", "vesic"))
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
