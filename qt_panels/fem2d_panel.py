"""FEM 2D analysis panel — 6 analysis types."""

import math
import traceback
import numpy as np

from qt_panels.common import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QLabel, QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
    QScrollArea, QSplitter, QStackedWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, Qt,
    MplCanvas, make_results_box,
    MESH_PRESETS,
)
from fem2d import (
    analyze_gravity, analyze_foundation, analyze_slope_srm,
    analyze_excavation, analyze_seepage, analyze_consolidation,
    generate_rect_mesh,
)


class FEM2DPanel(QWidget):
    """2D Finite Element analysis — 6 analysis types."""

    FIELDS_MECHANICAL = [
        ("Displacement Magnitude", "disp_mag"),
        ("Displacement X", "disp_x"),
        ("Displacement Y", "disp_y"),
        ("Stress \u03c3xx", "sigma_xx"),
        ("Stress \u03c3yy", "sigma_yy"),
        ("Shear \u03c4xy", "tau_xy"),
    ]
    FIELDS_SEEPAGE = [
        ("Total Head", "head"),
        ("Pore Pressure", "pore_pressure"),
        ("Velocity Magnitude", "vel_mag"),
    ]
    FIELDS_CONSOLIDATION = [
        ("Settlement vs Time", "settle_time"),
        ("Pore Pressure vs Time", "pp_time"),
        ("Final Displacement Y", "final_disp_y"),
        ("Final Pore Pressure", "final_pp"),
    ]

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._last_result = None
        self._last_result_type = None  # 'fem', 'seepage', 'consolidation'
        self._last_mesh = None  # (nodes, elements) for consolidation
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: inputs (scrollable) ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(420)
        form_widget = QWidget()
        form = QVBoxLayout(form_widget)

        # Analysis type selector
        type_grp = QGroupBox("Analysis Type")
        tl = QVBoxLayout()
        self.type_cb = QComboBox()
        self.type_cb.addItems([
            "Gravity", "Foundation", "Slope SRM",
            "Excavation", "Seepage", "Consolidation",
        ])
        self.type_cb.currentIndexChanged.connect(self._on_type_changed)
        tl.addWidget(self.type_cb)
        type_grp.setLayout(tl)
        form.addWidget(type_grp)

        # Domain & Mesh
        dom_grp = QGroupBox("Domain && Mesh")
        dl = QFormLayout()
        self.dom_w = QDoubleSpinBox()
        self.dom_w.setRange(1.0, 500.0)
        self.dom_w.setValue(20.0)
        self.dom_w.setSingleStep(5)
        self.dom_w.setDecimals(1)
        self.dom_w.setSuffix(" m")
        self.dom_d = QDoubleSpinBox()
        self.dom_d.setRange(1.0, 200.0)
        self.dom_d.setValue(10.0)
        self.dom_d.setSingleStep(2)
        self.dom_d.setDecimals(1)
        self.dom_d.setSuffix(" m")
        self.mesh_cb = QComboBox()
        self.mesh_cb.addItems(list(MESH_PRESETS.keys()))
        self.mesh_cb.setCurrentText("Medium")
        dl.addRow("Width:", self.dom_w)
        dl.addRow("Depth:", self.dom_d)
        dl.addRow("Mesh Density:", self.mesh_cb)
        dom_grp.setLayout(dl)
        form.addWidget(dom_grp)

        # Material model selector
        mat_grp = QGroupBox("Material Model")
        mat_lay = QVBoxLayout()
        self.mat_model_cb = QComboBox()
        self.mat_model_cb.addItems(["Mohr-Coulomb", "Hardening Soil"])
        self.mat_model_cb.currentIndexChanged.connect(
            self._on_material_model_changed)
        mat_lay.addWidget(self.mat_model_cb)

        # Hardening Soil parameters (hidden by default)
        self.hs_group = QGroupBox("Hardening Soil Parameters")
        hs_lay = QFormLayout()
        self.hs_E50 = QDoubleSpinBox()
        self.hs_E50.setRange(100, 1e8)
        self.hs_E50.setValue(25000)
        self.hs_E50.setSingleStep(5000)
        self.hs_E50.setDecimals(0)
        self.hs_E50.setSuffix(" kPa")
        self.hs_Eur = QDoubleSpinBox()
        self.hs_Eur.setRange(100, 1e8)
        self.hs_Eur.setValue(75000)
        self.hs_Eur.setSingleStep(5000)
        self.hs_Eur.setDecimals(0)
        self.hs_Eur.setSuffix(" kPa")
        self.hs_m = QDoubleSpinBox()
        self.hs_m.setRange(0.1, 1.0)
        self.hs_m.setValue(0.5)
        self.hs_m.setSingleStep(0.1)
        self.hs_m.setDecimals(2)
        self.hs_pref = QDoubleSpinBox()
        self.hs_pref.setRange(1, 10000)
        self.hs_pref.setValue(100)
        self.hs_pref.setSingleStep(10)
        self.hs_pref.setDecimals(0)
        self.hs_pref.setSuffix(" kPa")
        self.hs_Rf = QDoubleSpinBox()
        self.hs_Rf.setRange(0.5, 1.0)
        self.hs_Rf.setValue(0.9)
        self.hs_Rf.setSingleStep(0.05)
        self.hs_Rf.setDecimals(2)
        hs_lay.addRow("E50_ref:", self.hs_E50)
        hs_lay.addRow("Eur_ref:", self.hs_Eur)
        hs_lay.addRow("m:", self.hs_m)
        hs_lay.addRow("p_ref:", self.hs_pref)
        hs_lay.addRow("R_f:", self.hs_Rf)
        self.hs_group.setLayout(hs_lay)
        self.hs_group.setVisible(False)
        mat_lay.addWidget(self.hs_group)
        mat_grp.setLayout(mat_lay)
        form.addWidget(mat_grp)

        # Soil properties table
        soil_grp = QGroupBox("Soil Properties")
        soil_lay = QVBoxLayout()
        self.soil_table = QTableWidget(1, 8)
        self.soil_table.setHorizontalHeaderLabels([
            "Name", "Thick", "E(kPa)", "\u03bd",
            "\u03b3(kN/m\u00b3)", "c(kPa)", "\u03c6(\u00b0)",
            "\u03c8(\u00b0)",
        ])
        self.soil_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.soil_table.setMaximumHeight(130)
        defaults = ["Clay", "10", "30000", "0.3", "18", "25", "20", "0"]
        for j, v in enumerate(defaults):
            self.soil_table.setItem(0, j, QTableWidgetItem(v))
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+")
        add_btn.setMaximumWidth(40)
        add_btn.clicked.connect(self._add_soil_row)
        rem_btn = QPushButton("\u2013")
        rem_btn.setMaximumWidth(40)
        rem_btn.setObjectName("secondary")
        rem_btn.clicked.connect(self._remove_soil_row)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(rem_btn)
        btn_row.addStretch()
        soil_lay.addWidget(self.soil_table)
        soil_lay.addLayout(btn_row)
        soil_grp.setLayout(soil_lay)
        form.addWidget(soil_grp)

        # Stacked widget for type-specific inputs
        self.stack = QStackedWidget()

        # 0: Gravity (no extra inputs)
        self.stack.addWidget(QWidget())

        # 1: Foundation
        fnd_w = QWidget()
        fnd_l = QFormLayout(fnd_w)
        self.fnd_q = QDoubleSpinBox()
        self.fnd_q.setRange(0, 10000)
        self.fnd_q.setValue(100.0)
        self.fnd_q.setSingleStep(10)
        self.fnd_q.setDecimals(1)
        self.fnd_q.setSuffix(" kPa")
        self.fnd_B = QDoubleSpinBox()
        self.fnd_B.setRange(0.1, 100)
        self.fnd_B.setValue(2.0)
        self.fnd_B.setSingleStep(0.5)
        self.fnd_B.setDecimals(2)
        self.fnd_B.setSuffix(" m")
        fnd_l.addRow("Load q:", self.fnd_q)
        fnd_l.addRow("Foundation width B:", self.fnd_B)
        self.stack.addWidget(fnd_w)

        # 2: Slope SRM
        srm_w = QWidget()
        srm_l = QFormLayout(srm_w)
        self.srm_height = QDoubleSpinBox()
        self.srm_height.setRange(1, 100)
        self.srm_height.setValue(5.0)
        self.srm_height.setSingleStep(1)
        self.srm_height.setDecimals(1)
        self.srm_height.setSuffix(" m")
        self.srm_angle = QDoubleSpinBox()
        self.srm_angle.setRange(5, 80)
        self.srm_angle.setValue(30.0)
        self.srm_angle.setSingleStep(5)
        self.srm_angle.setDecimals(1)
        self.srm_angle.setSuffix("\u00b0")
        self.srm_crest = QDoubleSpinBox()
        self.srm_crest.setRange(0, 100)
        self.srm_crest.setValue(5.0)
        self.srm_crest.setSingleStep(1)
        self.srm_crest.setDecimals(1)
        self.srm_crest.setSuffix(" m")
        self.srm_gwt_cb = QCheckBox("Groundwater table")
        self.srm_gwt_elev = QDoubleSpinBox()
        self.srm_gwt_elev.setRange(-200, 200)
        self.srm_gwt_elev.setValue(-3.0)
        self.srm_gwt_elev.setSingleStep(1)
        self.srm_gwt_elev.setDecimals(1)
        self.srm_gwt_elev.setSuffix(" m")
        self.srm_gwt_elev.setEnabled(False)
        self.srm_gwt_cb.toggled.connect(self.srm_gwt_elev.setEnabled)
        srm_l.addRow("Slope height:", self.srm_height)
        srm_l.addRow("Slope angle:", self.srm_angle)
        srm_l.addRow("Crest offset:", self.srm_crest)
        srm_l.addRow("", self.srm_gwt_cb)
        srm_l.addRow("GWT elevation:", self.srm_gwt_elev)
        self.stack.addWidget(srm_w)

        # 3: Excavation
        exc_w = QWidget()
        exc_l = QVBoxLayout(exc_w)
        exc_form = QFormLayout()
        self.exc_depth = QDoubleSpinBox()
        self.exc_depth.setRange(0.5, 50)
        self.exc_depth.setValue(5.0)
        self.exc_depth.setSingleStep(1)
        self.exc_depth.setDecimals(1)
        self.exc_depth.setSuffix(" m")
        self.exc_width = QDoubleSpinBox()
        self.exc_width.setRange(1, 200)
        self.exc_width.setValue(10.0)
        self.exc_width.setSingleStep(2)
        self.exc_width.setDecimals(1)
        self.exc_width.setSuffix(" m")
        self.exc_wall_d = QDoubleSpinBox()
        self.exc_wall_d.setRange(1, 100)
        self.exc_wall_d.setValue(10.0)
        self.exc_wall_d.setSingleStep(1)
        self.exc_wall_d.setDecimals(1)
        self.exc_wall_d.setSuffix(" m")
        self.exc_EI = QDoubleSpinBox()
        self.exc_EI.setRange(100, 1e8)
        self.exc_EI.setValue(50000)
        self.exc_EI.setSingleStep(5000)
        self.exc_EI.setDecimals(0)
        self.exc_EI.setSuffix(" kNm\u00b2/m")
        self.exc_EA = QDoubleSpinBox()
        self.exc_EA.setRange(1e3, 1e10)
        self.exc_EA.setValue(5e6)
        self.exc_EA.setSingleStep(1e5)
        self.exc_EA.setDecimals(0)
        self.exc_EA.setSuffix(" kN/m")
        exc_form.addRow("Excavation depth:", self.exc_depth)
        exc_form.addRow("Excavation width:", self.exc_width)
        exc_form.addRow("Wall depth:", self.exc_wall_d)
        exc_form.addRow("Wall EI:", self.exc_EI)
        exc_form.addRow("Wall EA:", self.exc_EA)
        exc_l.addLayout(exc_form)

        # GWT for excavation
        self.exc_gwt_cb = QCheckBox("Groundwater table")
        self.exc_gwt_elev = QDoubleSpinBox()
        self.exc_gwt_elev.setRange(-200, 200)
        self.exc_gwt_elev.setValue(-3.0)
        self.exc_gwt_elev.setSingleStep(1)
        self.exc_gwt_elev.setDecimals(1)
        self.exc_gwt_elev.setSuffix(" m")
        self.exc_gwt_elev.setEnabled(False)
        self.exc_gwt_cb.toggled.connect(self.exc_gwt_elev.setEnabled)
        gwt_row = QHBoxLayout()
        gwt_row.addWidget(self.exc_gwt_cb)
        gwt_row.addWidget(QLabel("Elev:"))
        gwt_row.addWidget(self.exc_gwt_elev)
        gwt_row.addStretch()
        exc_l.addLayout(gwt_row)

        # Strut table
        exc_l.addWidget(QLabel("Struts:"))
        self.strut_table = QTableWidget(1, 2)
        self.strut_table.setHorizontalHeaderLabels(
            ["Depth (m)", "Stiffness (kN/m/m)"])
        self.strut_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.strut_table.setMaximumHeight(100)
        self.strut_table.setItem(0, 0, QTableWidgetItem("1.5"))
        self.strut_table.setItem(0, 1, QTableWidgetItem("50000"))
        strut_btns = QHBoxLayout()
        strut_add = QPushButton("+")
        strut_add.setMaximumWidth(40)
        strut_add.clicked.connect(self._add_strut_row)
        strut_rem = QPushButton("\u2013")
        strut_rem.setMaximumWidth(40)
        strut_rem.setObjectName("secondary")
        strut_rem.clicked.connect(self._remove_strut_row)
        strut_btns.addWidget(strut_add)
        strut_btns.addWidget(strut_rem)
        strut_btns.addStretch()
        exc_l.addWidget(self.strut_table)
        exc_l.addLayout(strut_btns)
        self.stack.addWidget(exc_w)

        # 4: Seepage
        seep_w = QWidget()
        seep_l = QFormLayout(seep_w)
        self.seep_k = QComboBox()
        self.seep_k.setEditable(True)
        self.seep_k.addItems(["1e-3", "1e-4", "1e-5", "1e-6", "1e-7", "1e-8"])
        self.seep_k.setCurrentText("1e-5")
        seep_l.addRow("Conductivity k (m/s):", self.seep_k)
        self.seep_left_cb = QCheckBox("Left edge")
        self.seep_left_cb.setChecked(True)
        self.seep_left_head = QDoubleSpinBox()
        self.seep_left_head.setRange(-100, 100)
        self.seep_left_head.setValue(10.0)
        self.seep_left_head.setSingleStep(1)
        self.seep_left_head.setDecimals(1)
        self.seep_left_head.setSuffix(" m")
        self.seep_right_cb = QCheckBox("Right edge")
        self.seep_right_cb.setChecked(True)
        self.seep_right_head = QDoubleSpinBox()
        self.seep_right_head.setRange(-100, 100)
        self.seep_right_head.setValue(0.0)
        self.seep_right_head.setSingleStep(1)
        self.seep_right_head.setDecimals(1)
        self.seep_right_head.setSuffix(" m")
        self.seep_top_cb = QCheckBox("Top edge")
        self.seep_top_head = QDoubleSpinBox()
        self.seep_top_head.setRange(-100, 100)
        self.seep_top_head.setValue(0.0)
        self.seep_top_head.setSingleStep(1)
        self.seep_top_head.setDecimals(1)
        self.seep_top_head.setSuffix(" m")
        self.seep_bot_cb = QCheckBox("Bottom edge")
        self.seep_bot_head = QDoubleSpinBox()
        self.seep_bot_head.setRange(-100, 100)
        self.seep_bot_head.setValue(0.0)
        self.seep_bot_head.setSingleStep(1)
        self.seep_bot_head.setDecimals(1)
        self.seep_bot_head.setSuffix(" m")
        seep_l.addRow(self.seep_left_cb, self.seep_left_head)
        seep_l.addRow(self.seep_right_cb, self.seep_right_head)
        seep_l.addRow(self.seep_top_cb, self.seep_top_head)
        seep_l.addRow(self.seep_bot_cb, self.seep_bot_head)
        self.stack.addWidget(seep_w)

        # 5: Consolidation
        con_w = QWidget()
        con_l = QFormLayout(con_w)
        self.con_k = QComboBox()
        self.con_k.setEditable(True)
        self.con_k.addItems(["1e-6", "1e-7", "1e-8", "1e-9", "1e-10"])
        self.con_k.setCurrentText("1e-8")
        self.con_q = QDoubleSpinBox()
        self.con_q.setRange(0, 10000)
        self.con_q.setValue(100.0)
        self.con_q.setSingleStep(10)
        self.con_q.setDecimals(1)
        self.con_q.setSuffix(" kPa")
        self.con_t_start = QDoubleSpinBox()
        self.con_t_start.setRange(0.001, 36500)
        self.con_t_start.setValue(0.01)
        self.con_t_start.setSingleStep(0.1)
        self.con_t_start.setDecimals(3)
        self.con_t_start.setSuffix(" days")
        self.con_t_end = QDoubleSpinBox()
        self.con_t_end.setRange(1, 36500)
        self.con_t_end.setValue(365.0)
        self.con_t_end.setSingleStep(30)
        self.con_t_end.setDecimals(0)
        self.con_t_end.setSuffix(" days")
        self.con_n_steps = QSpinBox()
        self.con_n_steps.setRange(5, 200)
        self.con_n_steps.setValue(20)
        self.con_log_cb = QCheckBox("Log-spaced time steps")
        self.con_log_cb.setChecked(True)
        con_l.addRow("Conductivity k (m/s):", self.con_k)
        con_l.addRow("Surface load q:", self.con_q)
        con_l.addRow("Time start:", self.con_t_start)
        con_l.addRow("Time end:", self.con_t_end)
        con_l.addRow("Time steps:", self.con_n_steps)
        con_l.addRow("", self.con_log_cb)
        self.stack.addWidget(con_w)

        form.addWidget(self.stack)

        # Solver settings
        solver_grp = QGroupBox("Solver Settings")
        solver_grp.setCheckable(True)
        solver_grp.setChecked(False)
        solver_lay = QFormLayout()
        self.solver_nsteps = QSpinBox()
        self.solver_nsteps.setRange(1, 100)
        self.solver_nsteps.setValue(10)
        self.solver_maxiter = QSpinBox()
        self.solver_maxiter.setRange(10, 1000)
        self.solver_maxiter.setValue(100)
        self.solver_tol = QComboBox()
        self.solver_tol.setEditable(True)
        self.solver_tol.addItems(["1e-4", "1e-5", "1e-6"])
        self.solver_tol.setCurrentText("1e-5")
        self.solver_srf_tol = QDoubleSpinBox()
        self.solver_srf_tol.setRange(0.001, 0.1)
        self.solver_srf_tol.setValue(0.02)
        self.solver_srf_tol.setSingleStep(0.005)
        self.solver_srf_tol.setDecimals(3)
        self.solver_srf_tol_label = QLabel("SRF tolerance:")
        solver_lay.addRow("Load steps:", self.solver_nsteps)
        solver_lay.addRow("Max NR iterations:", self.solver_maxiter)
        solver_lay.addRow("Convergence tol:", self.solver_tol)
        solver_lay.addRow(self.solver_srf_tol_label, self.solver_srf_tol)
        solver_grp.setLayout(solver_lay)
        self.solver_grp = solver_grp
        self._update_solver_visibility()
        form.addWidget(solver_grp)

        # Run button
        self.run_btn = QPushButton("Analyze")
        self.run_btn.clicked.connect(self._run_analysis)
        form.addWidget(self.run_btn)

        form.addStretch()
        scroll.setWidget(form_widget)
        outer.addWidget(scroll)

        # --- Right: field selector + plot + results ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        # Field selector row
        field_row = QHBoxLayout()
        field_row.addWidget(QLabel("Field:"))
        self.field_cb = QComboBox()
        self._update_field_options()
        self.field_cb.currentIndexChanged.connect(self._replot)
        field_row.addWidget(self.field_cb, 1)
        self.deformed_cb = QCheckBox("Deformed")
        self.deformed_cb.toggled.connect(self._replot)
        field_row.addWidget(self.deformed_cb)
        field_row.addWidget(QLabel("Scale:"))
        self.scale_sb = QDoubleSpinBox()
        self.scale_sb.setRange(0.1, 10000)
        self.scale_sb.setValue(1.0)
        self.scale_sb.setSingleStep(1)
        self.scale_sb.setDecimals(1)
        self.scale_sb.setMaximumWidth(80)
        self.scale_sb.valueChanged.connect(self._replot)
        field_row.addWidget(self.scale_sb)
        right_layout.addLayout(field_row)

        right_splitter = QSplitter(Qt.Vertical)
        self.canvas = MplCanvas(width=8, height=6)
        right_splitter.addWidget(self.canvas)
        self.results_text = make_results_box()
        right_splitter.addWidget(self.results_text)
        right_splitter.setSizes([500, 200])
        right_layout.addWidget(right_splitter)

        outer.addWidget(right_widget, 1)

    # --- Table helpers ---
    def _add_soil_row(self):
        r = self.soil_table.rowCount()
        self.soil_table.insertRow(r)
        defaults = [f"Layer {r + 1}", "5", "30000", "0.3", "18", "10", "25",
                    "0"]
        for j, v in enumerate(defaults):
            self.soil_table.setItem(r, j, QTableWidgetItem(v))

    def _remove_soil_row(self):
        row = self.soil_table.currentRow()
        if row >= 0 and self.soil_table.rowCount() > 1:
            self.soil_table.removeRow(row)

    def _add_strut_row(self):
        r = self.strut_table.rowCount()
        self.strut_table.insertRow(r)
        self.strut_table.setItem(r, 0, QTableWidgetItem(f"{r * 2 + 1.5:.1f}"))
        self.strut_table.setItem(r, 1, QTableWidgetItem("50000"))

    def _remove_strut_row(self):
        row = self.strut_table.currentRow()
        if row >= 0 and self.strut_table.rowCount() > 1:
            self.strut_table.removeRow(row)

    def _read_struts(self):
        """Return list of strut dicts from strut table."""
        struts = []
        for i in range(self.strut_table.rowCount()):
            d_item = self.strut_table.item(i, 0)
            k_item = self.strut_table.item(i, 1)
            if d_item and k_item:
                try:
                    depth = float(d_item.text())
                    stiffness = float(k_item.text())
                    if depth > 0 and stiffness > 0:
                        struts.append({
                            "depth": depth, "stiffness": stiffness})
                except ValueError:
                    pass
        return struts

    def _on_material_model_changed(self, idx):
        self.hs_group.setVisible(idx == 1)  # Show HS params for "Hardening Soil"

    # --- Type-change handler ---
    def _on_type_changed(self, idx):
        self.stack.setCurrentIndex(idx)
        self._update_field_options()
        self._update_solver_visibility()

    def _update_solver_visibility(self):
        """Show SRF tolerance only for Slope SRM analysis."""
        is_srm = self.type_cb.currentText() == "Slope SRM"
        self.solver_srf_tol.setVisible(is_srm)
        self.solver_srf_tol_label.setVisible(is_srm)

    def _update_field_options(self):
        self.field_cb.blockSignals(True)
        self.field_cb.clear()
        atype = self.type_cb.currentText()
        if atype == "Seepage":
            fields = self.FIELDS_SEEPAGE
        elif atype == "Consolidation":
            fields = self.FIELDS_CONSOLIDATION
        else:
            fields = self.FIELDS_MECHANICAL
        for label, key in fields:
            self.field_cb.addItem(label, key)
        self.field_cb.blockSignals(False)

    # --- Read soil table ---
    def _read_soil_layers(self):
        """Return list of dicts for fem2d functions."""
        layers = []
        current_top = 0.0
        is_hs = self.mat_model_cb.currentText() == "Hardening Soil"
        for i in range(self.soil_table.rowCount()):
            vals = []
            for j in range(self.soil_table.columnCount()):
                item = self.soil_table.item(i, j)
                vals.append(item.text() if item else "")
            if not vals[0]:
                continue
            thickness = float(vals[1]) if vals[1] else 10.0
            bottom = current_top - thickness
            layer = {
                "name": vals[0],
                "bottom_elevation": bottom,
                "E": float(vals[2]) if vals[2] else 30000,
                "nu": float(vals[3]) if vals[3] else 0.3,
                "gamma": float(vals[4]) if vals[4] else 18,
                "c": float(vals[5]) if vals[5] else 0,
                "phi": float(vals[6]) if vals[6] else 30,
                "psi": float(vals[7]) if (len(vals) > 7 and vals[7]) else 0,
            }
            if is_hs:
                layer["model"] = "hs"
                layer["E50_ref"] = self.hs_E50.value()
                layer["Eur_ref"] = self.hs_Eur.value()
                layer["m"] = self.hs_m.value()
                layer["p_ref"] = self.hs_pref.value()
                layer["R_f"] = self.hs_Rf.value()
            layers.append(layer)
            current_top = bottom
        if not layers:
            layers = [{"bottom_elevation": -10, "E": 30000, "nu": 0.3,
                        "gamma": 18, "c": 25, "phi": 20, "psi": 0}]
        return layers

    def _read_solver_params(self):
        """Return dict of solver params if the group is checked, else defaults."""
        if self.solver_grp.isChecked():
            return {
                "n_steps": self.solver_nsteps.value(),
                "max_iter": self.solver_maxiter.value(),
                "tol": float(self.solver_tol.currentText()),
                "srf_tol": self.solver_srf_tol.value(),
            }
        return {
            "n_steps": 10, "max_iter": 100, "tol": 1e-5, "srf_tol": 0.02,
        }

    # --- Run analysis ---
    def _run_analysis(self):
        try:
            self._status.showMessage("Running FEM analysis...")
            QApplication.processEvents()

            atype = self.type_cb.currentText()
            nx, ny = MESH_PRESETS[self.mesh_cb.currentText()]
            soil_layers = self._read_soil_layers()
            sl0 = soil_layers[0]
            sp = self._read_solver_params()

            if atype == "Gravity":
                result = analyze_gravity(
                    width=self.dom_w.value(),
                    depth=self.dom_d.value(),
                    gamma=sl0["gamma"], E=sl0["E"], nu=sl0["nu"],
                    nx=nx, ny=ny,
                )
                self._last_result = result
                self._last_result_type = "fem"
                self._last_mesh = None

            elif atype == "Foundation":
                result = analyze_foundation(
                    B=self.fnd_B.value(), q=self.fnd_q.value(),
                    depth=self.dom_d.value(),
                    E=sl0["E"], nu=sl0["nu"], gamma=sl0["gamma"],
                    nx=nx, ny=ny,
                )
                self._last_result = result
                self._last_result_type = "fem"
                self._last_mesh = None

            elif atype == "Slope SRM":
                H = self.srm_height.value()
                angle = self.srm_angle.value()
                crest = self.srm_crest.value()
                dx = H / math.tan(math.radians(angle))
                surface_points = [
                    (0, H), (crest, H),
                    (crest + dx, 0),
                    (crest + dx + crest, 0),
                ]
                gwt = None
                if self.srm_gwt_cb.isChecked():
                    gwt = self.srm_gwt_elev.value()
                result = analyze_slope_srm(
                    surface_points=surface_points,
                    soil_layers=soil_layers,
                    nx=nx, ny=ny, gwt=gwt,
                    srf_tol=sp["srf_tol"],
                    n_load_steps=sp["n_steps"],
                    max_iter=sp["max_iter"],
                    tol=sp["tol"],
                )
                self._last_result = result
                self._last_result_type = "fem"
                self._last_mesh = None

            elif atype == "Excavation":
                struts = self._read_struts()
                exc_gwt = None
                if self.exc_gwt_cb.isChecked():
                    exc_gwt = self.exc_gwt_elev.value()
                result = analyze_excavation(
                    width=self.exc_width.value(),
                    depth=self.exc_depth.value(),
                    wall_depth=self.exc_wall_d.value(),
                    soil_layers=soil_layers,
                    wall_EI=self.exc_EI.value(),
                    wall_EA=self.exc_EA.value(),
                    nx=nx, ny=ny,
                    n_steps=sp["n_steps"],
                    max_iter=sp["max_iter"],
                    tol=sp["tol"],
                    struts=struts if struts else None,
                    gwt=exc_gwt,
                )
                self._last_result = result
                self._last_result_type = "fem"
                self._last_mesh = None

            elif atype == "Seepage":
                width = self.dom_w.value()
                depth = self.dom_d.value()
                nodes, elements = generate_rect_mesh(
                    0, width, -depth, 0, nx, ny)
                k = float(self.seep_k.currentText())
                head_bcs = self._build_head_bcs(nodes, width, depth)
                result = analyze_seepage(nodes, elements, k, head_bcs)
                self._last_result = result
                self._last_result_type = "seepage"
                self._last_mesh = None

            elif atype == "Consolidation":
                width = self.dom_w.value()
                depth = self.dom_d.value()
                k = float(self.con_k.currentText())
                t_start = self.con_t_start.value() * 86400  # days -> s
                t_end = self.con_t_end.value() * 86400
                n_steps = self.con_n_steps.value()
                if self.con_log_cb.isChecked():
                    time_pts = np.logspace(
                        np.log10(max(t_start, 1)),
                        np.log10(t_end), n_steps)
                else:
                    time_pts = np.linspace(t_start, t_end, n_steps)
                # Store mesh for contour plots
                nodes, elements = generate_rect_mesh(
                    0, width, -depth, 0, nx, ny)
                self._last_mesh = (nodes, elements)
                result = analyze_consolidation(
                    width=width, depth=depth,
                    soil_layers=soil_layers, k=k,
                    load_q=self.con_q.value(),
                    time_points=time_pts, nx=nx, ny=ny,
                )
                self._last_result = result
                self._last_result_type = "consolidation"

            # Display results
            self.results_text.setText(result.summary())
            self._replot()
            fos_msg = ""
            if hasattr(result, "FOS") and result.FOS is not None:
                fos_msg = f" | FOS = {result.FOS:.3f}"
            self._status.showMessage(
                f"FEM analysis complete ({atype}){fos_msg}", 15000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    def _build_head_bcs(self, nodes, width, depth):
        """Build Dirichlet head BCs from edge checkboxes."""
        tol = 0.01
        x = nodes[:, 0]
        y = nodes[:, 1]
        head_bcs = []
        if self.seep_left_cb.isChecked():
            h = self.seep_left_head.value()
            for n in np.where(np.abs(x - 0) < tol)[0]:
                head_bcs.append((int(n), h))
        if self.seep_right_cb.isChecked():
            h = self.seep_right_head.value()
            for n in np.where(np.abs(x - width) < tol)[0]:
                head_bcs.append((int(n), h))
        if self.seep_top_cb.isChecked():
            h = self.seep_top_head.value()
            for n in np.where(np.abs(y - 0) < tol)[0]:
                head_bcs.append((int(n), h))
        if self.seep_bot_cb.isChecked():
            h = self.seep_bot_head.value()
            for n in np.where(np.abs(y - (-depth)) < tol)[0]:
                head_bcs.append((int(n), h))
        return head_bcs

    # --- Plotting ---
    def _replot(self):
        """Redraw the plot with the current field selection."""
        if self._last_result is None:
            return
        field_key = self.field_cb.currentData()
        if field_key is None:
            return

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self.canvas.axes = ax

        if self._last_result_type == "fem":
            self._plot_fem(ax, self._last_result, field_key)
        elif self._last_result_type == "seepage":
            self._plot_seepage(ax, self._last_result, field_key)
        elif self._last_result_type == "consolidation":
            self._plot_consolidation(ax, self._last_result, field_key)

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def _plot_fem(self, ax, result, field):
        """Contour plot for mechanical FEM results."""
        nodes = result.nodes
        elements = result.elements
        x = nodes[:, 0]
        y = nodes[:, 1]

        ux = result.displacements[0::2]
        uy = result.displacements[1::2]

        if field == "disp_mag":
            values = np.sqrt(ux**2 + uy**2)
            label = "Displacement Magnitude (m)"
            per_node = True
        elif field == "disp_x":
            values = ux
            label = "Displacement X (m)"
            per_node = True
        elif field == "disp_y":
            values = uy
            label = "Displacement Y (m)"
            per_node = True
        elif field == "sigma_xx":
            values = result.stresses[:, 0]
            label = "Stress \u03c3xx (kPa)"
            per_node = False
        elif field == "sigma_yy":
            values = result.stresses[:, 1]
            label = "Stress \u03c3yy (kPa)"
            per_node = False
        elif field == "tau_xy":
            values = result.stresses[:, 2]
            label = "Shear \u03c4xy (kPa)"
            per_node = False
        else:
            return

        if per_node:
            tc = ax.tripcolor(x, y, elements, values,
                              cmap="RdYlBu_r", shading="gouraud")
        else:
            tc = ax.tripcolor(x, y, elements, facecolors=values,
                              cmap="RdYlBu_r")
        self.canvas.fig.colorbar(tc, ax=ax, label=label, shrink=0.8)

        # Deformed mesh overlay
        if self.deformed_cb.isChecked():
            scale = self.scale_sb.value()
            x_def = x + ux * scale
            y_def = y + uy * scale
            ax.triplot(x_def, y_def, elements,
                       color="k", linewidth=0.3, alpha=0.5)

        # FOS badge for SRM
        if result.FOS is not None:
            fos_color = "#4CAF50" if result.FOS >= 1.5 else "#D32F2F"
            ax.text(0.02, 0.98, f"FOS = {result.FOS:.3f}",
                    transform=ax.transAxes, fontsize=14, fontweight="bold",
                    va="top", color=fos_color,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                              edgecolor=fos_color, alpha=0.9))

        # Beam forces annotation for excavation
        if result.beam_forces:
            max_M = result.max_beam_moment_kNm_per_m
            max_V = result.max_beam_shear_kN_per_m
            ax.text(0.98, 0.98,
                    f"Max M = {max_M:.1f} kNm/m\nMax V = {max_V:.1f} kN/m",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                              edgecolor="gray", alpha=0.9))

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"FEM 2D \u2014 {self.type_cb.currentText()}",
                     fontweight="bold")
        ax.grid(True, alpha=0.2)

    def _plot_seepage(self, ax, result, field):
        """Contour plot for seepage results."""
        nodes = result.nodes
        elements = result.elements
        x = nodes[:, 0]
        y = nodes[:, 1]

        if field == "head":
            values = result.head
            label = "Total Head (m)"
            per_node = True
        elif field == "pore_pressure":
            values = result.pore_pressures
            label = "Pore Pressure (kPa)"
            per_node = True
        elif field == "vel_mag":
            values = np.sqrt(result.velocity[:, 0]**2 +
                             result.velocity[:, 1]**2)
            label = "Velocity Magnitude (m/s)"
            per_node = False
        else:
            return

        if per_node:
            tc = ax.tripcolor(x, y, elements, values,
                              cmap="Blues", shading="gouraud")
        else:
            tc = ax.tripcolor(x, y, elements, facecolors=values,
                              cmap="Blues")
        self.canvas.fig.colorbar(tc, ax=ax, label=label, shrink=0.8)

        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("FEM 2D \u2014 Seepage", fontweight="bold")
        ax.grid(True, alpha=0.2)

    def _plot_consolidation(self, ax, result, field):
        """Time-history or final-state plot for consolidation."""
        if field == "settle_time":
            if result.times is not None and result.displacements is not None:
                max_settle = []
                for t_idx in range(len(result.times)):
                    uy = result.displacements[t_idx][1::2]
                    max_settle.append(float(np.min(uy)))
                times_days = result.times / 86400.0
                ax.semilogx(times_days, np.array(max_settle) * 1000,
                            "b-o", markersize=3)
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Max Settlement (mm)")
                ax.set_title("Settlement vs Time", fontweight="bold")
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.02,
                        f"U = {result.degree_of_consolidation:.1%}",
                        transform=ax.transAxes, fontsize=12,
                        fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor="lightyellow",
                                  edgecolor="gray", alpha=0.9))

        elif field == "pp_time":
            if result.times is not None and result.pore_pressures is not None:
                max_pp = [float(np.max(result.pore_pressures[t]))
                          for t in range(len(result.times))]
                times_days = result.times / 86400.0
                ax.semilogx(times_days, max_pp, "r-o", markersize=3)
                ax.set_xlabel("Time (days)")
                ax.set_ylabel("Max Excess Pore Pressure (kPa)")
                ax.set_title("Pore Pressure Dissipation", fontweight="bold")
                ax.grid(True, alpha=0.3)

        elif field in ("final_disp_y", "final_pp"):
            if self._last_mesh is None:
                ax.text(0.5, 0.5, "Mesh data not available",
                        transform=ax.transAxes, ha="center")
                return
            nodes, elements = self._last_mesh
            x = nodes[:, 0]
            y = nodes[:, 1]
            if field == "final_disp_y":
                uy = result.displacements[-1][1::2]
                tc = ax.tripcolor(x, y, elements, uy,
                                  cmap="RdYlBu_r", shading="gouraud")
                self.canvas.fig.colorbar(
                    tc, ax=ax, label="Displacement Y (m)", shrink=0.8)
            else:
                pp = result.pore_pressures[-1]
                tc = ax.tripcolor(x, y, elements, pp,
                                  cmap="Reds", shading="gouraud")
                self.canvas.fig.colorbar(
                    tc, ax=ax, label="Excess Pore Pressure (kPa)",
                    shrink=0.8)
            ax.set_aspect("equal")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title("FEM 2D \u2014 Consolidation (final state)",
                         fontweight="bold")
            ax.grid(True, alpha=0.2)

    # --- State save/load ---
    def get_state(self):
        state = {
            "analysis_type": self.type_cb.currentText(),
            "domain_width": self.dom_w.value(),
            "domain_depth": self.dom_d.value(),
            "mesh_density": self.mesh_cb.currentText(),
            "results_text": self.results_text.toPlainText(),
        }
        # Material model + HS params
        state["material_model"] = self.mat_model_cb.currentText()
        state["hs_E50"] = self.hs_E50.value()
        state["hs_Eur"] = self.hs_Eur.value()
        state["hs_m"] = self.hs_m.value()
        state["hs_pref"] = self.hs_pref.value()
        state["hs_Rf"] = self.hs_Rf.value()
        # Soil table
        soil_rows = []
        for i in range(self.soil_table.rowCount()):
            row = []
            for j in range(self.soil_table.columnCount()):
                item = self.soil_table.item(i, j)
                row.append(item.text() if item else "")
            soil_rows.append(row)
        state["soil_rows"] = soil_rows
        # Solver settings
        state["solver_enabled"] = self.solver_grp.isChecked()
        state["solver_nsteps"] = self.solver_nsteps.value()
        state["solver_maxiter"] = self.solver_maxiter.value()
        state["solver_tol"] = self.solver_tol.currentText()
        state["solver_srf_tol"] = self.solver_srf_tol.value()
        # Type-specific
        state["fnd_q"] = self.fnd_q.value()
        state["fnd_B"] = self.fnd_B.value()
        state["srm_height"] = self.srm_height.value()
        state["srm_angle"] = self.srm_angle.value()
        state["srm_crest"] = self.srm_crest.value()
        state["srm_gwt_enabled"] = self.srm_gwt_cb.isChecked()
        state["srm_gwt_elev"] = self.srm_gwt_elev.value()
        state["exc_depth"] = self.exc_depth.value()
        state["exc_width"] = self.exc_width.value()
        state["exc_wall_d"] = self.exc_wall_d.value()
        state["exc_EI"] = self.exc_EI.value()
        state["exc_EA"] = self.exc_EA.value()
        state["exc_gwt_enabled"] = self.exc_gwt_cb.isChecked()
        state["exc_gwt_elev"] = self.exc_gwt_elev.value()
        # Strut table
        strut_rows = []
        for i in range(self.strut_table.rowCount()):
            row = []
            for j in range(self.strut_table.columnCount()):
                item = self.strut_table.item(i, j)
                row.append(item.text() if item else "")
            strut_rows.append(row)
        state["strut_rows"] = strut_rows
        # Seepage
        state["seep_k"] = self.seep_k.currentText()
        state["seep_left"] = self.seep_left_cb.isChecked()
        state["seep_left_head"] = self.seep_left_head.value()
        state["seep_right"] = self.seep_right_cb.isChecked()
        state["seep_right_head"] = self.seep_right_head.value()
        state["seep_top"] = self.seep_top_cb.isChecked()
        state["seep_top_head"] = self.seep_top_head.value()
        state["seep_bot"] = self.seep_bot_cb.isChecked()
        state["seep_bot_head"] = self.seep_bot_head.value()
        state["con_k"] = self.con_k.currentText()
        state["con_q"] = self.con_q.value()
        state["con_t_start"] = self.con_t_start.value()
        state["con_t_end"] = self.con_t_end.value()
        state["con_n_steps"] = self.con_n_steps.value()
        state["con_log"] = self.con_log_cb.isChecked()
        return state

    def set_state(self, state):
        self.type_cb.setCurrentText(state.get("analysis_type", "Gravity"))
        self.dom_w.setValue(state.get("domain_width", 20.0))
        self.dom_d.setValue(state.get("domain_depth", 10.0))
        self.mesh_cb.setCurrentText(state.get("mesh_density", "Medium"))
        # Material model + HS params
        self.mat_model_cb.setCurrentText(
            state.get("material_model", "Mohr-Coulomb"))
        self.hs_E50.setValue(state.get("hs_E50", 25000))
        self.hs_Eur.setValue(state.get("hs_Eur", 75000))
        self.hs_m.setValue(state.get("hs_m", 0.5))
        self.hs_pref.setValue(state.get("hs_pref", 100))
        self.hs_Rf.setValue(state.get("hs_Rf", 0.9))
        # Soil table
        soil_rows = state.get("soil_rows", [])
        if soil_rows:
            self.soil_table.setRowCount(len(soil_rows))
            for i, row in enumerate(soil_rows):
                for j, val in enumerate(row):
                    if j < self.soil_table.columnCount():
                        self.soil_table.setItem(
                            i, j, QTableWidgetItem(str(val)))
        # Solver settings
        self.solver_grp.setChecked(state.get("solver_enabled", False))
        self.solver_nsteps.setValue(state.get("solver_nsteps", 10))
        self.solver_maxiter.setValue(state.get("solver_maxiter", 100))
        self.solver_tol.setCurrentText(state.get("solver_tol", "1e-5"))
        self.solver_srf_tol.setValue(state.get("solver_srf_tol", 0.02))
        # Type-specific
        self.fnd_q.setValue(state.get("fnd_q", 100.0))
        self.fnd_B.setValue(state.get("fnd_B", 2.0))
        self.srm_height.setValue(state.get("srm_height", 5.0))
        self.srm_angle.setValue(state.get("srm_angle", 30.0))
        self.srm_crest.setValue(state.get("srm_crest", 5.0))
        self.srm_gwt_cb.setChecked(state.get("srm_gwt_enabled", False))
        self.srm_gwt_elev.setValue(state.get("srm_gwt_elev", -3.0))
        self.exc_depth.setValue(state.get("exc_depth", 5.0))
        self.exc_width.setValue(state.get("exc_width", 10.0))
        self.exc_wall_d.setValue(state.get("exc_wall_d", 10.0))
        self.exc_EI.setValue(state.get("exc_EI", 50000))
        self.exc_EA.setValue(state.get("exc_EA", 5e6))
        self.exc_gwt_cb.setChecked(state.get("exc_gwt_enabled", False))
        self.exc_gwt_elev.setValue(state.get("exc_gwt_elev", -3.0))
        # Strut table
        strut_rows = state.get("strut_rows", [])
        if strut_rows:
            self.strut_table.setRowCount(len(strut_rows))
            for i, row in enumerate(strut_rows):
                for j, val in enumerate(row):
                    if j < self.strut_table.columnCount():
                        self.strut_table.setItem(
                            i, j, QTableWidgetItem(str(val)))
        # Seepage
        self.seep_k.setCurrentText(state.get("seep_k", "1e-5"))
        self.seep_left_cb.setChecked(state.get("seep_left", True))
        self.seep_left_head.setValue(state.get("seep_left_head", 10.0))
        self.seep_right_cb.setChecked(state.get("seep_right", True))
        self.seep_right_head.setValue(state.get("seep_right_head", 0.0))
        self.seep_top_cb.setChecked(state.get("seep_top", False))
        self.seep_top_head.setValue(state.get("seep_top_head", 0.0))
        self.seep_bot_cb.setChecked(state.get("seep_bot", False))
        self.seep_bot_head.setValue(state.get("seep_bot_head", 0.0))
        self.con_k.setCurrentText(state.get("con_k", "1e-8"))
        self.con_q.setValue(state.get("con_q", 100.0))
        self.con_t_start.setValue(state.get("con_t_start", 0.01))
        self.con_t_end.setValue(state.get("con_t_end", 365.0))
        self.con_n_steps.setValue(state.get("con_n_steps", 20))
        self.con_log_cb.setChecked(state.get("con_log", True))
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
