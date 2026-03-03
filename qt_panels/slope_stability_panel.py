"""Slope stability analysis panel."""

import math
import traceback
import numpy as np

from qt_panels.common import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
    QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, Qt,
    MplCanvas, make_results_box,
    LAYER_COLORS, GWT_COLOR, SLIP_COLOR, SURFACE_COLOR,
)
from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, analyze_slope, search_critical_surface,
)


class SlopeStabilityPanel(QWidget):
    """Slope stability analysis with Bishop/Spencer/Fellenius."""

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._last_geom = None
        self._last_result = None
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: inputs (scrollable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        form_widget = QWidget()
        form = QVBoxLayout(form_widget)

        # Surface points table
        sg = QGroupBox("Surface Profile (x, z)")
        sl = QVBoxLayout()
        self.surface_table = QTableWidget(4, 2)
        self.surface_table.setHorizontalHeaderLabels(["x (m)", "z (m)"])
        self.surface_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.surface_table.setMaximumHeight(160)
        default_surface = [(0, 10), (10, 10), (30, 0), (50, 0)]
        for i, (x, z) in enumerate(default_surface):
            self.surface_table.setItem(i, 0, QTableWidgetItem(str(x)))
            self.surface_table.setItem(i, 1, QTableWidgetItem(str(z)))
        btn_row = QHBoxLayout()
        add_s = QPushButton("+")
        add_s.setMaximumWidth(40)
        add_s.clicked.connect(lambda: self._add_table_row(self.surface_table))
        rem_s = QPushButton("\u2013")
        rem_s.setMaximumWidth(40)
        rem_s.setObjectName("secondary")
        rem_s.clicked.connect(
            lambda: self._remove_table_row(self.surface_table))
        btn_row.addWidget(add_s)
        btn_row.addWidget(rem_s)
        btn_row.addStretch()
        sl.addWidget(self.surface_table)
        sl.addLayout(btn_row)
        sg.setLayout(sl)
        form.addWidget(sg)

        # Soil layers table
        lg = QGroupBox("Soil Layers")
        ll = QVBoxLayout()
        self.layers_table = QTableWidget(1, 7)
        self.layers_table.setHorizontalHeaderLabels([
            "Name", "Top (m)", "Bot (m)", "\u03b3 (kN/m\u00b3)",
            "\u03c6 (deg)", "c' (kPa)", "Mode",
        ])
        self.layers_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.layers_table.setMaximumHeight(130)
        defaults = ["Fill", "10", "-5", "18", "25", "10", "drained"]
        for j, val in enumerate(defaults):
            self.layers_table.setItem(0, j, QTableWidgetItem(val))
        btn_row2 = QHBoxLayout()
        add_l = QPushButton("+")
        add_l.setMaximumWidth(40)
        add_l.clicked.connect(lambda: self._add_layer_row())
        rem_l = QPushButton("\u2013")
        rem_l.setMaximumWidth(40)
        rem_l.setObjectName("secondary")
        rem_l.clicked.connect(
            lambda: self._remove_table_row(self.layers_table))
        btn_row2.addWidget(add_l)
        btn_row2.addWidget(rem_l)
        btn_row2.addStretch()
        ll.addWidget(self.layers_table)
        ll.addLayout(btn_row2)
        lg.setLayout(ll)
        form.addWidget(lg)

        # GWT
        wg = QGroupBox("Groundwater Table")
        wl = QVBoxLayout()
        self.gwt_check = QCheckBox("Enable GWT")
        self.gwt_check.setChecked(True)
        self.gwt_table = QTableWidget(2, 2)
        self.gwt_table.setHorizontalHeaderLabels(["x (m)", "z (m)"])
        self.gwt_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.gwt_table.setMaximumHeight(90)
        for i, (x, z) in enumerate([(0, 5), (50, -1)]):
            self.gwt_table.setItem(i, 0, QTableWidgetItem(str(x)))
            self.gwt_table.setItem(i, 1, QTableWidgetItem(str(z)))
        self.gwt_check.toggled.connect(self.gwt_table.setVisible)
        wl.addWidget(self.gwt_check)
        wl.addWidget(self.gwt_table)
        wg.setLayout(wl)
        form.addWidget(wg)

        # Analysis settings
        ag = QGroupBox("Analysis Settings")
        al = QFormLayout()
        self.method_cb = QComboBox()
        self.method_cb.addItems(["bishop", "spencer", "fellenius"])
        self.nslices_sb = QSpinBox(value=30, minimum=5, maximum=100)
        self.fos_req_sb = QDoubleSpinBox(value=1.5, minimum=1.0, maximum=5.0,
                                         singleStep=0.1, decimals=2)
        self.compare_cb = QCheckBox("Compare all methods")
        self.compare_cb.setChecked(True)
        self.kh_sb = QDoubleSpinBox(value=0.0, minimum=0.0, maximum=0.5,
                                    singleStep=0.05, decimals=3)
        al.addRow("Method:", self.method_cb)
        al.addRow("Slices:", self.nslices_sb)
        al.addRow("Required FOS:", self.fos_req_sb)
        al.addRow("", self.compare_cb)
        al.addRow("Seismic kh:", self.kh_sb)
        ag.setLayout(al)
        form.addWidget(ag)

        # Grid search settings
        srch = QGroupBox("Grid Search")
        srchl = QFormLayout()
        self.nx_sb = QSpinBox(value=10, minimum=3, maximum=30)
        self.ny_sb = QSpinBox(value=10, minimum=3, maximum=30)
        srchl.addRow("Grid NX:", self.nx_sb)
        srchl.addRow("Grid NY:", self.ny_sb)
        srch.setLayout(srchl)
        form.addWidget(srch)

        # Buttons
        btn_layout = QHBoxLayout()
        self.search_btn = QPushButton("Search Critical Surface")
        self.search_btn.clicked.connect(self._run_search)
        btn_layout.addWidget(self.search_btn)
        form.addLayout(btn_layout)

        form.addStretch()
        scroll.setWidget(form_widget)
        outer.addWidget(scroll)

        # --- Right: plot + results ---
        right_splitter = QSplitter(Qt.Vertical)

        self.canvas = MplCanvas(width=8, height=5)
        right_splitter.addWidget(self.canvas)

        self.results_text = make_results_box()
        right_splitter.addWidget(self.results_text)

        right_splitter.setSizes([450, 200])
        outer.addWidget(right_splitter, 1)

        # Draw initial geometry
        self._plot_geometry_only()

    # --- Table helpers ---
    @staticmethod
    def _add_table_row(table):
        table.insertRow(table.rowCount())

    @staticmethod
    def _remove_table_row(table):
        row = table.currentRow()
        if row >= 0 and table.rowCount() > 1:
            table.removeRow(row)

    def _add_layer_row(self):
        r = self.layers_table.rowCount()
        self.layers_table.insertRow(r)
        defaults = [f"Layer {r + 1}", "0", "-10", "18", "30", "5", "drained"]
        for j, val in enumerate(defaults):
            self.layers_table.setItem(r, j, QTableWidgetItem(val))

    # --- Parse inputs ---
    def _read_surface(self):
        pts = []
        for i in range(self.surface_table.rowCount()):
            x_item = self.surface_table.item(i, 0)
            z_item = self.surface_table.item(i, 1)
            if x_item and z_item and x_item.text() and z_item.text():
                pts.append((float(x_item.text()), float(z_item.text())))
        return pts

    def _read_layers(self):
        layers = []
        for i in range(self.layers_table.rowCount()):
            vals = []
            for j in range(self.layers_table.columnCount()):
                item = self.layers_table.item(i, j)
                vals.append(item.text() if item else "")
            if not vals[0]:
                continue
            layers.append(SlopeSoilLayer(
                name=vals[0],
                top_elevation=float(vals[1]),
                bottom_elevation=float(vals[2]),
                gamma=float(vals[3]),
                phi=float(vals[4]),
                c_prime=float(vals[5]),
                analysis_mode=vals[6] if vals[6] else "drained",
            ))
        return layers

    def _read_gwt(self):
        if not self.gwt_check.isChecked():
            return None
        pts = []
        for i in range(self.gwt_table.rowCount()):
            x_item = self.gwt_table.item(i, 0)
            z_item = self.gwt_table.item(i, 1)
            if x_item and z_item and x_item.text() and z_item.text():
                pts.append((float(x_item.text()), float(z_item.text())))
        return pts if pts else None

    def _build_geometry(self):
        surface = self._read_surface()
        layers = self._read_layers()
        gwt = self._read_gwt()
        return SlopeGeometry(
            surface_points=surface,
            soil_layers=layers,
            gwt_points=gwt,
            kh=self.kh_sb.value(),
        )

    # --- Analysis ---
    def _run_search(self):
        try:
            self._status.showMessage("Searching for critical surface...")
            QApplication.processEvents()

            geom = self._build_geometry()
            self._last_geom = geom

            search_result = search_critical_surface(
                geom,
                nx=self.nx_sb.value(),
                ny=self.ny_sb.value(),
                method=self.method_cb.currentText(),
                n_slices=self.nslices_sb.value(),
                FOS_required=self.fos_req_sb.value(),
            )

            if search_result.critical is None:
                self.results_text.setText("No valid slip surfaces found.")
                self._status.showMessage("Search complete — no valid surfaces")
                return

            # Compare methods on the critical surface
            crit = search_result.critical
            compare_text = ""
            if self.compare_cb.isChecked():
                r_comp = analyze_slope(
                    geom, crit.xc, crit.yc, crit.radius,
                    method=self.method_cb.currentText(),
                    n_slices=self.nslices_sb.value(),
                    FOS_required=self.fos_req_sb.value(),
                    compare_methods=True,
                )
                compare_text = (
                    f"\n  Method Comparison (critical surface):\n"
                    f"    Fellenius: {r_comp.FOS_fellenius:.3f}\n"
                    f"    Bishop:   {r_comp.FOS_bishop:.3f}\n"
                )
                if r_comp.theta_spencer is not None:
                    compare_text += (
                        f"    Spencer:  {r_comp.FOS:.3f} "
                        f"(theta={r_comp.theta_spencer:.1f} deg)\n"
                    )

            self._last_result = search_result.critical
            self.results_text.setText(
                search_result.summary() + compare_text
            )
            self._plot_result(geom, search_result.critical)
            self._status.showMessage(
                f"Critical FOS = {crit.FOS:.3f} "
                f"({crit.method}, {search_result.n_surfaces_evaluated} surfaces)",
                15000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    # --- Plotting ---
    def _plot_geometry_only(self):
        try:
            geom = self._build_geometry()
            self._plot_cross_section(geom)
        except Exception:
            pass

    def _plot_cross_section(self, geom, result=None):
        ax = self.canvas.axes
        ax.clear()

        surface_x = [p[0] for p in geom.surface_points]
        surface_z = [p[1] for p in geom.surface_points]
        x_min, x_max = min(surface_x), max(surface_x)

        # Fill soil layers (bottom to top)
        sorted_layers = sorted(geom.soil_layers,
                                key=lambda la: la.top_elevation)
        for i, layer in enumerate(sorted_layers):
            color = LAYER_COLORS[i % len(LAYER_COLORS)]
            bot = layer.bottom_elevation
            top = layer.top_elevation
            x_fill = np.linspace(x_min, x_max, 200)
            z_top_fill = np.clip(
                np.interp(x_fill, surface_x, surface_z), bot, top)
            z_bot_fill = np.full_like(x_fill, bot)
            ax.fill_between(x_fill, z_bot_fill, z_top_fill,
                             color=color, alpha=0.6,
                             label=f"{layer.name} (\u03c6={layer.phi}\u00b0)")

        # Ground surface
        ax.plot(surface_x, surface_z, color=SURFACE_COLOR, linewidth=2.5,
                label="Surface", zorder=4)

        # GWT
        if geom.gwt_points:
            gwt_x = [p[0] for p in geom.gwt_points]
            gwt_z = [p[1] for p in geom.gwt_points]
            ax.plot(gwt_x, gwt_z, color=GWT_COLOR, linewidth=1.5,
                    linestyle="--", marker="v", markersize=6,
                    label="GWT", zorder=4)

        # Slip circle
        if result is not None:
            theta = np.linspace(0, 2 * math.pi, 200)
            cx = result.xc + result.radius * np.cos(theta)
            cy = result.yc + result.radius * np.sin(theta)
            ax.plot(cx, cy, color=SLIP_COLOR, linewidth=2,
                    linestyle="-", zorder=5, label="Slip circle")
            ax.plot(result.xc, result.yc, "x", color=SLIP_COLOR,
                    markersize=10, markeredgewidth=2, zorder=6)

            # Entry/exit markers
            ax.plot(result.x_entry, np.interp(result.x_entry, surface_x,
                    surface_z), "o", color=SLIP_COLOR, markersize=8, zorder=6)
            ax.plot(result.x_exit, np.interp(result.x_exit, surface_x,
                    surface_z), "o", color=SLIP_COLOR, markersize=8, zorder=6)

            # FOS label
            status = "STABLE" if result.is_stable else "UNSTABLE"
            fos_color = "#4CAF50" if result.is_stable else "#D32F2F"
            ax.text(
                0.98, 0.95,
                f"FOS = {result.FOS:.3f}\n{result.method} [{status}]",
                transform=ax.transAxes, fontsize=14, fontweight="bold",
                ha="right", va="top", color=fos_color,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor=fos_color, alpha=0.9),
                zorder=10,
            )

        # Formatting
        z_low = min(la.bottom_elevation for la in geom.soil_layers)
        margin = (x_max - x_min) * 0.05
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(z_low - 2, max(surface_z) + 5)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Slope Stability Analysis", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _plot_result(self, geom, result):
        self._plot_cross_section(geom, result)

    def get_state(self):
        surface = self._read_surface()
        layers = []
        for i in range(self.layers_table.rowCount()):
            row = []
            for j in range(self.layers_table.columnCount()):
                item = self.layers_table.item(i, j)
                row.append(item.text() if item else "")
            layers.append(row)
        gwt = []
        for i in range(self.gwt_table.rowCount()):
            x_item = self.gwt_table.item(i, 0)
            z_item = self.gwt_table.item(i, 1)
            if x_item and z_item:
                gwt.append([x_item.text(), z_item.text()])
        return {
            "surface_points": surface,
            "layers": layers,
            "gwt_enabled": self.gwt_check.isChecked(),
            "gwt_points": gwt,
            "method": self.method_cb.currentText(),
            "n_slices": self.nslices_sb.value(),
            "fos_required": self.fos_req_sb.value(),
            "compare": self.compare_cb.isChecked(),
            "kh": self.kh_sb.value(),
            "nx": self.nx_sb.value(),
            "ny": self.ny_sb.value(),
            "results_text": self.results_text.toPlainText(),
        }

    def set_state(self, state):
        pts = state.get("surface_points", [])
        self.surface_table.setRowCount(len(pts))
        for i, (x, z) in enumerate(pts):
            self.surface_table.setItem(i, 0, QTableWidgetItem(str(x)))
            self.surface_table.setItem(i, 1, QTableWidgetItem(str(z)))
        layers = state.get("layers", [])
        self.layers_table.setRowCount(len(layers))
        for i, row in enumerate(layers):
            for j, val in enumerate(row):
                self.layers_table.setItem(i, j, QTableWidgetItem(str(val)))
        self.gwt_check.setChecked(state.get("gwt_enabled", True))
        gwt_pts = state.get("gwt_points", [])
        self.gwt_table.setRowCount(max(len(gwt_pts), 1))
        for i, (x, z) in enumerate(gwt_pts):
            self.gwt_table.setItem(i, 0, QTableWidgetItem(str(x)))
            self.gwt_table.setItem(i, 1, QTableWidgetItem(str(z)))
        self.method_cb.setCurrentText(state.get("method", "bishop"))
        self.nslices_sb.setValue(state.get("n_slices", 30))
        self.fos_req_sb.setValue(state.get("fos_required", 1.5))
        self.compare_cb.setChecked(state.get("compare", True))
        self.kh_sb.setValue(state.get("kh", 0.0))
        self.nx_sb.setValue(state.get("nx", 10))
        self.ny_sb.setValue(state.get("ny", 10))
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
