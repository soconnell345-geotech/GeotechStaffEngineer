"""Slope stability analysis panel.

Updated for Phase 5 of slope stability overhaul:
- Removed FOS_required input
- Added convergence tolerance control
- Added surface type selector (Circular / Non-circular)
- Added entry/exit x-range limits
- Added slice visualization on cross-section
- Removed soil nails section
"""

import math
import traceback
import numpy as np

from qt_panels.common import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
    QScrollArea, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView, Qt,
    QFileDialog, QMessageBox,
    MplCanvas, make_results_box, NavigationToolbar2QT,
    LAYER_COLORS, GWT_COLOR, SLIP_COLOR, SURFACE_COLOR,
)
from slope_stability import (
    SlopeGeometry, SlopeSoilLayer, analyze_slope, search_critical_surface,
)
from slope_stability.slices import build_slices, compute_slice_forces
from slope_stability.slip_surface import CircularSlipSurface


class SlopeStabilityPanel(QWidget):
    """Slope stability analysis with Bishop/Spencer/Fellenius."""

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._last_geom = None
        self._last_result = None
        self._last_slices = None
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: inputs (scrollable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        form_widget = QWidget()
        form = QVBoxLayout(form_widget)

        # Import DXF button
        self.import_dxf_btn = QPushButton("Import DXF...")
        self.import_dxf_btn.setObjectName("secondary")
        self.import_dxf_btn.clicked.connect(self._import_dxf)
        form.addWidget(self.import_dxf_btn)

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
        self.surface_type_cb = QComboBox()
        self.surface_type_cb.addItems(["Circular", "Non-circular (Spencer)"])
        self.surface_type_cb.currentIndexChanged.connect(
            self._on_surface_type_changed)
        self.method_cb = QComboBox()
        self.method_cb.addItems(["bishop", "spencer", "fellenius"])
        self.nslices_sb = QSpinBox(value=30, minimum=5, maximum=100)
        self.tol_sb = QDoubleSpinBox(value=0.0001, minimum=0.000001,
                                      maximum=0.01, singleStep=0.0001,
                                      decimals=6)
        self.compare_cb = QCheckBox("Compare all methods")
        self.compare_cb.setChecked(True)
        self.show_slices_cb = QCheckBox("Show slices on plot")
        self.show_slices_cb.setChecked(True)
        self.kh_sb = QDoubleSpinBox(value=0.0, minimum=0.0, maximum=0.5,
                                    singleStep=0.05, decimals=3)
        al.addRow("Surface Type:", self.surface_type_cb)
        al.addRow("Method:", self.method_cb)
        al.addRow("Slices:", self.nslices_sb)
        al.addRow("Tolerance:", self.tol_sb)
        al.addRow("", self.compare_cb)
        al.addRow("", self.show_slices_cb)
        al.addRow("Seismic kh:", self.kh_sb)
        ag.setLayout(al)
        form.addWidget(ag)

        # Grid search settings (circular)
        self.srch_group = QGroupBox("Grid Search (Circular)")
        srchl = QFormLayout()
        self.nx_sb = QSpinBox(value=10, minimum=3, maximum=30)
        self.ny_sb = QSpinBox(value=10, minimum=3, maximum=30)
        srchl.addRow("Grid NX:", self.nx_sb)
        srchl.addRow("Grid NY:", self.ny_sb)
        self.srch_group.setLayout(srchl)
        form.addWidget(self.srch_group)

        # Noncircular search settings
        self.nc_group = QGroupBox("Noncircular Search")
        ncl = QFormLayout()
        self.n_trials_sb = QSpinBox(value=500, minimum=50, maximum=5000,
                                     singleStep=100)
        self.n_points_sb = QSpinBox(value=5, minimum=3, maximum=10)
        ncl.addRow("Trials:", self.n_trials_sb)
        ncl.addRow("Polyline pts:", self.n_points_sb)
        self.nc_group.setLayout(ncl)
        self.nc_group.setVisible(False)
        form.addWidget(self.nc_group)

        # Entry/exit limits
        self.limits_group = QGroupBox("Entry/Exit Limits")
        lim_layout = QVBoxLayout()
        self.limits_check = QCheckBox("Limit entry/exit x-range")
        self.limits_check.setChecked(False)
        lim_form = QFormLayout()
        self.entry_min_sb = QDoubleSpinBox(value=0.0, minimum=-1000,
                                            maximum=1000, decimals=1)
        self.entry_max_sb = QDoubleSpinBox(value=15.0, minimum=-1000,
                                            maximum=1000, decimals=1)
        self.exit_min_sb = QDoubleSpinBox(value=25.0, minimum=-1000,
                                           maximum=1000, decimals=1)
        self.exit_max_sb = QDoubleSpinBox(value=50.0, minimum=-1000,
                                           maximum=1000, decimals=1)
        lim_form.addRow("Entry x min:", self.entry_min_sb)
        lim_form.addRow("Entry x max:", self.entry_max_sb)
        lim_form.addRow("Exit x min:", self.exit_min_sb)
        lim_form.addRow("Exit x max:", self.exit_max_sb)
        self.limits_form_widget = QWidget()
        self.limits_form_widget.setLayout(lim_form)
        self.limits_form_widget.setVisible(False)
        self.limits_check.toggled.connect(self.limits_form_widget.setVisible)
        lim_layout.addWidget(self.limits_check)
        lim_layout.addWidget(self.limits_form_widget)
        self.limits_group.setLayout(lim_layout)
        form.addWidget(self.limits_group)

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

        self.plot_tabs = QTabWidget()

        # Tab 1: Cross-Section
        cs_widget = QWidget()
        cs_layout = QVBoxLayout(cs_widget)
        cs_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(width=8, height=5)
        self.toolbar = NavigationToolbar2QT(self.canvas, cs_widget)
        cs_layout.addWidget(self.toolbar)
        cs_layout.addWidget(self.canvas)
        self.plot_tabs.addTab(cs_widget, "Cross-Section")

        # Tab 2: Stress Distribution
        stress_widget = QWidget()
        stress_layout = QVBoxLayout(stress_widget)
        stress_layout.setContentsMargins(0, 0, 0, 0)
        self.stress_canvas = MplCanvas(width=8, height=5)
        self.stress_toolbar = NavigationToolbar2QT(
            self.stress_canvas, stress_widget)
        stress_layout.addWidget(self.stress_toolbar)
        stress_layout.addWidget(self.stress_canvas)
        self.plot_tabs.addTab(stress_widget, "Stress Distribution")

        right_splitter.addWidget(self.plot_tabs)

        self.results_text = make_results_box()
        right_splitter.addWidget(self.results_text)

        right_splitter.setSizes([450, 200])
        outer.addWidget(right_splitter, 1)

        # Draw initial geometry
        self._plot_geometry_only()

    # --- DXF Import ---
    def _import_dxf(self):
        """Open a DXF file, show mapping dialog, populate geometry."""
        try:
            from dxf_import import discover_layers, parse_dxf_geometry, build_slope_geometry
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependency",
                "DXF import requires ezdxf.\n"
                "Install with: pip install ezdxf")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open DXF File", "",
            "DXF Files (*.dxf);;All Files (*)")
        if not filepath:
            return

        try:
            discovery = discover_layers(filepath)
            if not discovery.layers:
                QMessageBox.warning(
                    self, "Empty DXF",
                    "No layers with entities found in this DXF file.")
                return

            from qt_panels.dxf_import_dialog import DxfLayerMappingDialog
            dlg = DxfLayerMappingDialog(self, discovery, mode="slope")
            if dlg.exec_() != dlg.Accepted:
                return

            mapping, soil_properties, nail_defaults, units, flip_y = (
                dlg.get_results())

            parse_result = parse_dxf_geometry(
                filepath, mapping, units=units, flip_y=flip_y)

            geom = build_slope_geometry(
                parse_result, soil_properties, nail_defaults)

            self._populate_from_geometry(geom)
            self._status.showMessage(
                f"DXF imported: {len(geom.surface_points)} surface pts, "
                f"{len(geom.soil_layers)} layers", 10000)

        except Exception as e:
            QMessageBox.critical(
                self, "DXF Import Error", str(e))

    def _populate_from_geometry(self, geom):
        """Fill surface, layers, and GWT tables from a SlopeGeometry."""
        # Surface points
        self.surface_table.setRowCount(len(geom.surface_points))
        for i, (x, z) in enumerate(geom.surface_points):
            self.surface_table.setItem(
                i, 0, QTableWidgetItem(f"{x:.2f}"))
            self.surface_table.setItem(
                i, 1, QTableWidgetItem(f"{z:.2f}"))

        # Soil layers
        self.layers_table.setRowCount(len(geom.soil_layers))
        for i, layer in enumerate(geom.soil_layers):
            vals = [
                layer.name,
                f"{layer.top_elevation:.2f}",
                f"{layer.bottom_elevation:.2f}",
                f"{layer.gamma:.1f}",
                f"{layer.phi:.1f}",
                f"{layer.c_prime:.1f}",
                layer.analysis_mode,
            ]
            for j, val in enumerate(vals):
                self.layers_table.setItem(i, j, QTableWidgetItem(val))

        # GWT
        if geom.gwt_points:
            self.gwt_check.setChecked(True)
            self.gwt_table.setRowCount(len(geom.gwt_points))
            for i, (x, z) in enumerate(geom.gwt_points):
                self.gwt_table.setItem(
                    i, 0, QTableWidgetItem(f"{x:.2f}"))
                self.gwt_table.setItem(
                    i, 1, QTableWidgetItem(f"{z:.2f}"))
        else:
            self.gwt_check.setChecked(False)

        # Redraw
        self._plot_geometry_only()

    def _on_surface_type_changed(self, idx):
        is_circular = (idx == 0)
        self.srch_group.setVisible(is_circular)
        self.nc_group.setVisible(not is_circular)
        # Method selector only for circular; noncircular always uses Spencer
        self.method_cb.setEnabled(is_circular)
        if not is_circular:
            self.method_cb.setCurrentText("spencer")

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

            is_noncircular = self.surface_type_cb.currentIndex() == 1
            surface_type = "noncircular" if is_noncircular else "circular"
            tol = self.tol_sb.value()

            # Entry/exit limits
            x_entry_range = None
            x_exit_range = None
            if self.limits_check.isChecked():
                x_entry_range = (self.entry_min_sb.value(),
                                 self.entry_max_sb.value())
                x_exit_range = (self.exit_min_sb.value(),
                                self.exit_max_sb.value())

            search_kwargs = dict(
                nx=self.nx_sb.value(),
                ny=self.ny_sb.value(),
                method=self.method_cb.currentText(),
                n_slices=self.nslices_sb.value(),
                tol=tol,
                surface_type=surface_type,
                x_entry_range=x_entry_range,
                x_exit_range=x_exit_range,
            )
            if is_noncircular:
                search_kwargs["n_trials"] = self.n_trials_sb.value()
                search_kwargs["n_points"] = self.n_points_sb.value()

            search_result = search_critical_surface(geom, **search_kwargs)

            if search_result.critical is None:
                self.results_text.setText("No valid slip surfaces found.")
                self._status.showMessage("Search complete - no valid surfaces")
                return

            # Build slices for visualization
            crit = search_result.critical
            self._last_slices = None
            if crit.radius > 0:
                try:
                    slip = CircularSlipSurface(crit.xc, crit.yc, crit.radius)
                    self._last_slices = build_slices(
                        geom, slip, self.nslices_sb.value())
                except (ValueError, ZeroDivisionError):
                    pass

            # Compare methods on the critical surface
            compare_text = ""
            if self.compare_cb.isChecked() and crit.radius > 0:
                r_comp = analyze_slope(
                    geom, crit.xc, crit.yc, crit.radius,
                    method=self.method_cb.currentText(),
                    n_slices=self.nslices_sb.value(),
                    tol=tol,
                    compare_methods=True,
                )
                compare_text = (
                    f"\n  Method Comparison (critical surface):\n"
                    f"    Fellenius: {r_comp.FOS_fellenius:.3f}\n"
                )
                if r_comp.FOS_bishop is not None:
                    compare_text += (
                        f"    Bishop:   {r_comp.FOS_bishop:.3f}\n"
                    )
                if r_comp.theta_spencer is not None:
                    compare_text += (
                        f"    Spencer:  {r_comp.FOS:.3f} "
                        f"(theta={r_comp.theta_spencer:.1f} deg)\n"
                    )

            self._last_result = crit
            self.results_text.setText(
                search_result.summary() + compare_text
            )
            self._plot_result(geom, crit)
            self._status.showMessage(
                f"Critical FOS = {crit.FOS:.3f} "
                f"({crit.method}, {search_result.n_surfaces_evaluated} "
                f"surfaces)", 15000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    # --- Plotting ---
    def _plot_geometry_only(self):
        try:
            geom = self._build_geometry()
            self._plot_cross_section(geom)
            self._plot_stress_distribution(geom, None)
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
            x_fill = np.linspace(x_min, x_max, 200)
            if layer.bottom_boundary_points is not None:
                bpts_x = [p[0] for p in layer.bottom_boundary_points]
                bpts_z = [p[1] for p in layer.bottom_boundary_points]
                z_bot_fill = np.interp(x_fill, bpts_x, bpts_z)
            else:
                z_bot_fill = np.full_like(x_fill, layer.bottom_elevation)
            z_top_fill = np.clip(
                np.interp(x_fill, surface_x, surface_z),
                z_bot_fill, layer.top_elevation)
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

        # Slip circle / surface
        if result is not None:
            if result.radius > 0:
                # Circular slip surface
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

            # Slice visualization (pickable for force polygon)
            self._slice_artists = {}
            if (self.show_slices_cb.isChecked() and
                    self._last_slices is not None):
                for idx, s in enumerate(self._last_slices):
                    line, = ax.plot(
                        [s.x_mid, s.x_mid], [s.z_base, s.z_top],
                        color="#666666", linewidth=1.0, alpha=0.6,
                        zorder=3, picker=5)
                    self._slice_artists[line] = idx

            # FOS label
            fos_color = "#4CAF50" if result.FOS >= 1.0 else "#D32F2F"
            ax.text(
                0.98, 0.95,
                f"FOS = {result.FOS:.3f}\n{result.method}",
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

        # Connect pick event for slice force polygon
        if hasattr(self, '_pick_cid') and self._pick_cid is not None:
            self.canvas.mpl_disconnect(self._pick_cid)
        self._pick_cid = self.canvas.mpl_connect(
            'pick_event', self._on_slice_pick)
        self._highlight_patch = None

        self.canvas.draw()

    def _plot_result(self, geom, result):
        self._plot_cross_section(geom, result)
        self._plot_stress_distribution(geom, result)

    def _plot_stress_distribution(self, geom, result):
        """Plot contact stresses along the slip surface on the stress tab."""
        ax = self.stress_canvas.axes
        ax.clear()

        if self._last_slices is None or result is None:
            ax.text(0.5, 0.5, "Run analysis to see stress distribution",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, color='#888888')
            self.stress_canvas.draw()
            return

        # Compute stresses from slices
        dist = []
        sigma_n = []
        tau_mob = []
        tau_avail = []
        cumulative = 0.0
        for s in self._last_slices:
            sf = compute_slice_forces(s)
            dl = s.base_length
            cumulative += dl
            dist.append(cumulative - dl / 2.0)
            sigma_n.append(sf.N_prime / dl if dl > 0 else 0.0)
            tau_mob.append(sf.S_mobilized / dl if dl > 0 else 0.0)
            tau_avail.append(sf.T_available / dl if dl > 0 else 0.0)

        ax.plot(dist, sigma_n, 'b-o', markersize=3, linewidth=1.5,
                label="$\\sigma'_n$ (effective normal)")
        ax.plot(dist, tau_mob, 'r--s', markersize=3, linewidth=1.5,
                label="$\\tau_{mob}$ (mobilized shear)")
        ax.plot(dist, tau_avail, 'g-^', markersize=3, linewidth=1.5,
                label="$\\tau_{avail}$ (shear resistance)")
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)

        ax.legend(fontsize=9, loc='upper right')
        ax.set_xlabel("Distance Along Slip Surface (m)")
        ax.set_ylabel("Stress (kPa)")
        ax.set_title(
            f"Contact Stresses (FOS={result.FOS:.3f}, {result.method})",
            fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.stress_canvas.draw()

    # --- Slice force polygon ---
    def _on_slice_pick(self, event):
        """Handle click on a slice line — show force polygon popup."""
        artist = event.artist
        if not hasattr(self, '_slice_artists'):
            return
        idx = self._slice_artists.get(artist)
        if idx is None or self._last_slices is None:
            return

        s = self._last_slices[idx]
        forces = compute_slice_forces(s)

        # Highlight the picked slice on main plot
        ax = self.canvas.axes
        if self._highlight_patch is not None:
            try:
                self._highlight_patch.remove()
            except ValueError:
                pass
        from matplotlib.patches import Polygon as MplPolygon
        verts = [
            (s.x_left, s.z_base), (s.x_left, s.z_top),
            (s.x_right, s.z_top), (s.x_right, s.z_base),
        ]
        self._highlight_patch = MplPolygon(
            verts, closed=True, facecolor="yellow", alpha=0.5,
            edgecolor="orange", linewidth=2, zorder=8)
        ax.add_patch(self._highlight_patch)
        self.canvas.draw()

        # Open force polygon popup
        self._plot_force_polygon(idx, s, forces)

    def _plot_force_polygon(self, idx, s, forces):
        """Open a popup matplotlib window with the slice free-body diagram."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(6, 7))
        fig.canvas.manager.set_window_title(
            f"Slice {idx + 1} Free Body Diagram")

        alpha = s.alpha
        cx = s.x_mid
        cz = s.z_centroid

        # Draw slice outline (tilted rectangle)
        hw = s.width / 2.0
        hh = s.height / 2.0
        corners = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh),
        ]
        # Shift to centroid position (no rotation for clarity)
        xs = [cx + c[0] for c in corners] + [cx + corners[0][0]]
        zs = [cz + c[1] for c in corners] + [cz + corners[0][1]]
        ax.fill(xs[:-1], zs[:-1], color="#e6d4a0", alpha=0.6,
                edgecolor="black", linewidth=1.5)
        ax.plot(xs, zs, color="black", linewidth=1.5)

        # Draw base line (tilted)
        base_cx = cx
        base_cz = s.z_base
        bl2 = s.base_length / 2.0
        bx1 = base_cx - bl2 * math.cos(alpha)
        bz1 = base_cz - bl2 * math.sin(alpha)
        bx2 = base_cx + bl2 * math.cos(alpha)
        bz2 = base_cz + bl2 * math.sin(alpha)
        ax.plot([bx1, bx2], [bz1, bz2], color="#8B4513", linewidth=3,
                zorder=5)

        # Helper: draw force arrow
        def _arrow(x0, z0, dx, dz, color, label, ha="left", va="bottom"):
            scale = s.height * 0.4 / max(abs(forces.W), 1e-6)
            adx = dx * scale
            adz = dz * scale
            ax.annotate(
                "", xy=(x0 + adx, z0 + adz), xytext=(x0, z0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5,
                                mutation_scale=15),
                zorder=10,
            )
            ax.text(
                x0 + adx * 1.15, z0 + adz * 1.15,
                f"{label}\n{abs(math.sqrt(dx**2 + dz**2)):.1f} kN/m",
                fontsize=9, color=color, fontweight="bold",
                ha=ha, va=va, zorder=11,
            )

        # W: weight downward from centroid
        _arrow(cx, cz, 0, -forces.W, "#1565C0", "W",
               ha="right", va="top")

        # N': normal to base, into soil
        n_dx = -math.sin(alpha)
        n_dz = -math.cos(alpha)
        _arrow(base_cx, base_cz, n_dx * forces.N_prime,
               n_dz * forces.N_prime, "#2E7D32", "N'",
               ha="right", va="top")

        # S: along base in driving direction
        s_dx = math.cos(alpha)
        s_dz = math.sin(alpha)
        if forces.S_mobilized < 0:
            s_dx, s_dz = -s_dx, -s_dz
        _arrow(base_cx, base_cz, s_dx * abs(forces.S_mobilized),
               s_dz * abs(forces.S_mobilized), "#D32F2F", "S",
               ha="left", va="bottom")

        # T: along base, resisting (opposite to S)
        t_dx = -math.cos(alpha)
        t_dz = -math.sin(alpha)
        if forces.S_mobilized < 0:
            t_dx, t_dz = -t_dx, -t_dz
        _arrow(base_cx, base_cz, t_dx * forces.T_available,
               t_dz * forces.T_available, "#1B5E20", "T",
               ha="right", va="top")

        # U: pore water force (normal to base, upward)
        if forces.U > 0.1:
            u_dx = math.sin(alpha)
            u_dz = math.cos(alpha)
            _arrow(base_cx, base_cz, u_dx * forces.U,
                   u_dz * forces.U, "#00BCD4", "U",
                   ha="left", va="bottom")

        # Seismic: horizontal
        if abs(forces.seismic) > 0.1:
            _arrow(cx, cz, forces.seismic, 0, "#FF9800", "kh*W",
                   ha="left", va="bottom")

        # Info text box
        info = (
            f"Slice {idx + 1}\n"
            f"x_mid = {s.x_mid:.2f} m\n"
            f"alpha = {forces.alpha_deg:.1f} deg\n"
            f"c = {s.c:.1f} kPa, phi = {s.phi:.1f} deg\n"
            f"u = {s.pore_pressure:.1f} kPa\n"
            f"dl = {s.base_length:.3f} m\n"
            f"\n"
            f"W = {forces.W:.1f} kN/m\n"
            f"N' = {forces.N_prime:.1f} kN/m\n"
            f"S = {forces.S_mobilized:.1f} kN/m\n"
            f"T = {forces.T_available:.1f} kN/m\n"
            f"U = {forces.U:.1f} kN/m"
        )
        ax.text(
            0.02, 0.98, info, transform=ax.transAxes,
            fontsize=9, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#666", alpha=0.95),
            zorder=12,
        )

        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title(f"Slice {idx + 1} Free Body Diagram", fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plt.show(block=False)

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
            "surface_type": self.surface_type_cb.currentIndex(),
            "method": self.method_cb.currentText(),
            "n_slices": self.nslices_sb.value(),
            "tol": self.tol_sb.value(),
            "compare": self.compare_cb.isChecked(),
            "show_slices": self.show_slices_cb.isChecked(),
            "kh": self.kh_sb.value(),
            "nx": self.nx_sb.value(),
            "ny": self.ny_sb.value(),
            "n_trials": self.n_trials_sb.value(),
            "n_points": self.n_points_sb.value(),
            "limits_enabled": self.limits_check.isChecked(),
            "entry_min": self.entry_min_sb.value(),
            "entry_max": self.entry_max_sb.value(),
            "exit_min": self.exit_min_sb.value(),
            "exit_max": self.exit_max_sb.value(),
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
        self.surface_type_cb.setCurrentIndex(
            state.get("surface_type", 0))
        self.method_cb.setCurrentText(state.get("method", "bishop"))
        self.nslices_sb.setValue(state.get("n_slices", 30))
        self.tol_sb.setValue(state.get("tol", 0.0001))
        self.compare_cb.setChecked(state.get("compare", True))
        self.show_slices_cb.setChecked(state.get("show_slices", True))
        self.kh_sb.setValue(state.get("kh", 0.0))
        self.nx_sb.setValue(state.get("nx", 10))
        self.ny_sb.setValue(state.get("ny", 10))
        self.n_trials_sb.setValue(state.get("n_trials", 500))
        self.n_points_sb.setValue(state.get("n_points", 5))
        self.limits_check.setChecked(state.get("limits_enabled", False))
        self.entry_min_sb.setValue(state.get("entry_min", 0.0))
        self.entry_max_sb.setValue(state.get("entry_max", 15.0))
        self.exit_min_sb.setValue(state.get("exit_min", 25.0))
        self.exit_max_sb.setValue(state.get("exit_max", 50.0))
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
