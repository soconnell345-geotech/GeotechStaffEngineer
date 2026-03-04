"""Footing design panel — bearing capacity (all methods) + settlement."""

import math
import traceback
import numpy as np

from qt_panels.common import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QPushButton, QCheckBox,
    QScrollArea, QSplitter, Qt,
    MplCanvas, make_results_box, NavigationToolbar2QT,
    GWT_COLOR, SLIP_COLOR,
    mpatches,
)
from bearing_capacity import (
    Footing, SoilLayer, BearingSoilProfile, BearingCapacityAnalysis,
)


class FootingDesignPanel(QWidget):
    """Footing design: bearing capacity (multi-method) + elastic settlement."""

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._last_results = {}  # store for plotting
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: scrollable inputs ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(400)
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
                                       singleStep=0.5, decimals=1,
                                       suffix=" kN/m\u00b3")
        self.gwt_cb = QCheckBox("Groundwater present")
        self.gwt_depth_sb = QDoubleSpinBox(value=3.0, minimum=0.0, maximum=50,
                                           singleStep=0.5, decimals=1,
                                           suffix=" m")
        self.gwt_depth_sb.setEnabled(False)
        self.gwt_cb.toggled.connect(self.gwt_depth_sb.setEnabled)
        sl.addRow("Friction angle \u03c6:", self.phi_sb)
        sl.addRow("Cohesion c:", self.c_sb)
        sl.addRow("Unit weight \u03b3:", self.gamma_sb)
        sl.addRow("", self.gwt_cb)
        sl.addRow("GWT depth:", self.gwt_depth_sb)
        sg.setLayout(sl)
        form.addWidget(sg)

        # Settlement parameters
        stg = QGroupBox("Settlement Parameters")
        stl = QFormLayout()
        self.q_applied_sb = QDoubleSpinBox(
            value=150.0, minimum=0.0, maximum=5000,
            singleStep=10, decimals=1, suffix=" kPa")
        self.Es_sb = QDoubleSpinBox(
            value=25000, minimum=100, maximum=1e6,
            singleStep=1000, decimals=0, suffix=" kPa")
        self.nu_sb = QDoubleSpinBox(
            value=0.3, minimum=0.0, maximum=0.5,
            singleStep=0.05, decimals=2)
        self.H_sb = QDoubleSpinBox(
            value=10.0, minimum=0.1, maximum=200,
            singleStep=1.0, decimals=1, suffix=" m")
        # Optional SPT N-value
        self.spt_cb = QCheckBox("SPT N-value available")
        self.spt_n_sb = QSpinBox(value=20, minimum=1, maximum=100)
        self.spt_n_sb.setEnabled(False)
        self.spt_cb.toggled.connect(self.spt_n_sb.setEnabled)
        stl.addRow("Applied stress q:", self.q_applied_sb)
        stl.addRow("Elastic modulus Es:", self.Es_sb)
        stl.addRow("Poisson's ratio \u03bd:", self.nu_sb)
        stl.addRow("Layer thickness H:", self.H_sb)
        stl.addRow("", self.spt_cb)
        stl.addRow("SPT N:", self.spt_n_sb)
        stg.setLayout(stl)
        form.addWidget(stg)

        # Analysis settings
        ag = QGroupBox("Analysis Settings")
        al = QFormLayout()
        self.fs_sb = QDoubleSpinBox(value=3.0, minimum=1.0, maximum=10,
                                    singleStep=0.5, decimals=1)
        al.addRow("Factor of Safety:", self.fs_sb)
        ag.setLayout(al)
        form.addWidget(ag)

        # Concrete design group (ACI 318-19)
        cg = QGroupBox("Concrete Design (ACI 318)")
        cl = QVBoxLayout()
        self.concrete_cb = QCheckBox("Enable concrete design")
        self.concrete_cb.setChecked(False)
        self.concrete_form_widget = QWidget()
        cf = QFormLayout(self.concrete_form_widget)

        self.col_load_sb = QDoubleSpinBox(
            value=1000, minimum=1, maximum=100000,
            singleStep=100, decimals=0, suffix=" kN")
        self.col_width_sb = QDoubleSpinBox(
            value=0.4, minimum=0.1, maximum=5.0,
            singleStep=0.05, decimals=2, suffix=" m")
        self.col_depth_sb = QDoubleSpinBox(
            value=0.4, minimum=0.1, maximum=5.0,
            singleStep=0.05, decimals=2, suffix=" m")
        self.ftg_thick_sb = QDoubleSpinBox(
            value=0.5, minimum=0.15, maximum=3.0,
            singleStep=0.05, decimals=2, suffix=" m")
        self.fc_sb = QDoubleSpinBox(
            value=28.0, minimum=17.0, maximum=70.0,
            singleStep=1.0, decimals=1, suffix=" MPa")
        self.fy_sb = QDoubleSpinBox(
            value=420.0, minimum=250.0, maximum=700.0,
            singleStep=10.0, decimals=0, suffix=" MPa")
        self.cover_sb = QDoubleSpinBox(
            value=75.0, minimum=25.0, maximum=150.0,
            singleStep=5.0, decimals=0, suffix=" mm")
        self.bar_size_cb = QComboBox()
        self.bar_size_cb.addItems([
            "#4", "#5", "#6", "#7", "#8", "#9", "#10", "#11"])
        self.bar_size_cb.setCurrentText("#8")

        cf.addRow("Column Pu:", self.col_load_sb)
        cf.addRow("Column width:", self.col_width_sb)
        cf.addRow("Column depth:", self.col_depth_sb)
        cf.addRow("Footing thickness h:", self.ftg_thick_sb)
        cf.addRow("f'c:", self.fc_sb)
        cf.addRow("fy:", self.fy_sb)
        cf.addRow("Cover:", self.cover_sb)
        cf.addRow("Bar size:", self.bar_size_cb)

        self.concrete_form_widget.setVisible(False)
        self.concrete_cb.toggled.connect(self.concrete_form_widget.setVisible)
        cl.addWidget(self.concrete_cb)
        cl.addWidget(self.concrete_form_widget)
        cg.setLayout(cl)
        form.addWidget(cg)

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

        right_splitter.setSizes([400, 250])
        outer.addWidget(right_splitter, 1)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
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

            B = self.width_sb.value()
            D = self.depth_sb.value()
            footing = Footing(
                width=B,
                length=length,
                depth=D,
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

            fs = self.fs_sb.value()

            # --- Run all 3 bearing capacity methods ---
            bc_results = {}
            for method in ["vesic", "meyerhof", "hansen"]:
                analysis = BearingCapacityAnalysis(
                    footing=footing,
                    soil=soil,
                    factor_of_safety=fs,
                    ngamma_method=method,
                )
                bc_results[method] = analysis.compute()

            # --- Settlement ---
            from settlement import elastic_settlement, stress_at_depth

            q_applied = self.q_applied_sb.value()
            Es = self.Es_sb.value()
            nu = self.nu_sb.value()
            H = self.H_sb.value()
            L_val = length if length else B

            s_elastic = elastic_settlement(q=q_applied, B=B, Es=Es, nu=nu)

            # Stress distribution for plot
            depths = np.linspace(0.01, H, 50)
            stresses = [stress_at_depth(q_applied, B, L_val, z) for z in depths]

            # Schmertmann (approximate using single Es layer)
            s_schmertmann = None
            schmertmann_note = ""
            try:
                from settlement import schmertmann_settlement, SchmertmannLayer
                q_overburden = soil.overburden_pressure(D)
                q_net = q_applied - q_overburden
                if q_net > 0:
                    # Create sublayers spanning 0 to 2B below footing
                    influence_depth = 2.0 * B
                    n_sublayers = 4
                    dz = influence_depth / n_sublayers
                    layers = []
                    for i in range(n_sublayers):
                        layers.append(SchmertmannLayer(
                            depth_top=i * dz,
                            depth_bottom=(i + 1) * dz,
                            Es=Es,
                        ))
                    s_schmertmann = schmertmann_settlement(
                        q_net=q_net,
                        q0=q_overburden,
                        B=B,
                        layers=layers,
                        footing_shape=shape if shape != "rectangular" else "square",
                    )
                else:
                    schmertmann_note = "q_net <= 0 (applied < overburden)"
            except Exception as e:
                schmertmann_note = str(e)

            # Meyerhof SPT-based settlement
            s_meyerhof_spt = None
            meyerhof_note = ""
            if self.spt_cb.isChecked():
                N = self.spt_n_sb.value()
                # Meyerhof (1965): q_all for 25mm settlement
                Kd = 1 + 0.33 * min(D / B, 1.0)
                if B <= 1.22:
                    q_all_25mm = 12.0 * N * Kd  # kPa for 25mm
                else:
                    q_all_25mm = 8.0 * N * ((B + 0.3) / B) ** 2 * Kd
                # Scale: settlement = 25mm * (q_applied / q_all_25mm)
                if q_all_25mm > 0:
                    s_meyerhof_spt = 0.025 * (q_applied / q_all_25mm)
            else:
                meyerhof_note = "Requires SPT N-value"

            # Store for plotting
            self._last_results = {
                "bc": bc_results,
                "s_elastic": s_elastic,
                "s_schmertmann": s_schmertmann,
                "s_meyerhof_spt": s_meyerhof_spt,
                "depths": depths,
                "stresses": stresses,
                "q_applied": q_applied,
                "B": B,
                "D": D,
            }

            # --- Concrete design (if enabled) ---
            concrete_result = None
            if self.concrete_cb.isChecked():
                from bearing_capacity.concrete_design import (
                    design_concrete_footing,
                )
                concrete_result = design_concrete_footing(
                    P_kN=self.col_load_sb.value(),
                    B_m=B,
                    L_m=L_val,
                    h_m=self.ftg_thick_sb.value(),
                    fc_MPa=self.fc_sb.value(),
                    fy_MPa=self.fy_sb.value(),
                    col_b_m=self.col_width_sb.value(),
                    col_d_m=self.col_depth_sb.value(),
                    cover_mm=self.cover_sb.value(),
                    bar_size=self.bar_size_cb.currentText(),
                )
                self._last_results["concrete"] = concrete_result

            # --- Build results text ---
            vesic = bc_results["vesic"]
            lines = self._format_results(
                bc_results, fs, B, D, q_applied, Es, nu, H,
                s_elastic, s_schmertmann, schmertmann_note,
                s_meyerhof_spt, meyerhof_note,
            )
            if concrete_result is not None:
                lines.append("")
                lines.append(concrete_result.summary())
            self.results_text.setText("\n".join(lines))

            # Plot bearing capacity diagram (use Vesic result for visualization)
            self._plot_result(vesic, bc_results, concrete_result)

            self._status.showMessage(
                f"q_ult: {vesic.q_ultimate:,.0f} kPa (Vesic), "
                f"Settlement: {s_elastic * 1000:.1f} mm (elastic)", 10000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    def _format_results(self, bc_results, fs, B, D, q_applied, Es, nu, H,
                        s_elastic, s_schmertmann, schmertmann_note,
                        s_meyerhof_spt, meyerhof_note):
        """Build multi-method results summary."""
        sep = "=" * 62
        dash = "-" * 62
        lines = [
            sep,
            "  FOOTING DESIGN RESULTS",
            sep,
            "",
        ]

        # Bearing capacity table
        lines.append("  BEARING CAPACITY")
        lines.append("  " + dash)
        lines.append(f"  {'Method':<14} {'q_ult (kPa)':>14} {'q_all (kPa)':>14}"
                     f" {'Nc':>8} {'Nq':>8} {'Ng':>8}")
        lines.append("  " + dash)
        for method in ["vesic", "meyerhof", "hansen"]:
            r = bc_results[method]
            lines.append(
                f"  {method.capitalize():<14} {r.q_ultimate:>14,.0f}"
                f" {r.q_allowable:>14,.0f}"
                f" {r.Nc:>8.2f} {r.Nq:>8.2f} {r.Ngamma:>8.2f}"
            )
        lines.append("  " + dash)

        # Range summary
        q_ults = [bc_results[m].q_ultimate for m in ["vesic", "meyerhof", "hansen"]]
        q_alls = [bc_results[m].q_allowable for m in ["vesic", "meyerhof", "hansen"]]
        lines.append(
            f"  q_ult range:  {min(q_ults):,.0f} - {max(q_ults):,.0f} kPa"
        )
        lines.append(
            f"  q_all range:  {min(q_alls):,.0f} - {max(q_alls):,.0f} kPa"
            f"  (FS = {fs:.1f})"
        )
        lines.append("")

        # Term breakdown (from Vesic)
        v = bc_results["vesic"]
        total = v.term_cohesion + v.term_overburden + v.term_selfweight
        if total > 0:
            lines.append("  Term Breakdown (Vesic):")
            lines.append(
                f"    Cohesion:     {v.term_cohesion:>10,.0f} kPa"
                f"  ({100 * v.term_cohesion / total:5.1f}%)"
            )
            lines.append(
                f"    Overburden:   {v.term_overburden:>10,.0f} kPa"
                f"  ({100 * v.term_overburden / total:5.1f}%)"
            )
            lines.append(
                f"    Self-weight:  {v.term_selfweight:>10,.0f} kPa"
                f"  ({100 * v.term_selfweight / total:5.1f}%)"
            )
            lines.append("")

        # Settlement table
        lines.append("  SETTLEMENT")
        lines.append("  " + dash)
        lines.append(f"  Applied stress:  q = {q_applied:.1f} kPa")
        lines.append(f"  Footing width:   B = {B:.2f} m,  D = {D:.2f} m")
        lines.append(f"  Es = {Es:,.0f} kPa,  nu = {nu:.2f},  H = {H:.1f} m")
        lines.append("")
        lines.append(f"  {'Method':<22} {'Settlement':>12} {'Status'}")
        lines.append("  " + dash)

        # Elastic
        lines.append(
            f"  {'Elastic':<22} {s_elastic * 1000:>10.1f} mm  Available"
        )

        # Schmertmann
        if s_schmertmann is not None:
            lines.append(
                f"  {'Schmertmann (1978)':<22}"
                f" {s_schmertmann * 1000:>10.1f} mm  Available"
            )
        else:
            note = schmertmann_note or "Computation error"
            lines.append(
                f"  {'Schmertmann (1978)':<22} {'N/A':>12}  {note}"
            )

        # Meyerhof SPT
        if s_meyerhof_spt is not None:
            lines.append(
                f"  {'Meyerhof SPT (1965)':<22}"
                f" {s_meyerhof_spt * 1000:>10.1f} mm  Available"
            )
        else:
            lines.append(
                f"  {'Meyerhof SPT (1965)':<22} {'N/A':>12}  {meyerhof_note}"
            )

        # Consolidation
        lines.append(
            f"  {'Consolidation':<22} {'N/A':>12}"
            f"  Requires Cc, e0, sigma_v0"
        )

        # Burland & Burbidge
        lines.append(
            f"  {'Burland & Burbidge':<22} {'N/A':>12}"
            f"  Not yet implemented"
        )

        lines.append("  " + dash)

        # Settlement range
        available = [s_elastic * 1000]
        if s_schmertmann is not None:
            available.append(s_schmertmann * 1000)
        if s_meyerhof_spt is not None:
            available.append(s_meyerhof_spt * 1000)
        if len(available) > 1:
            lines.append(
                f"  Settlement range: {min(available):.1f}"
                f" - {max(available):.1f} mm"
            )
        else:
            lines.append(f"  Settlement (elastic): {available[0]:.1f} mm")
        lines.append("")
        lines.append(sep)

        return lines

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def _plot_result(self, vesic_result, bc_results, concrete_result=None):
        """Plot bearing capacity diagram + optional concrete plan view."""
        fig = self.canvas.fig
        fig.clear()

        if concrete_result is not None:
            ax = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax2 = None

        B = self.width_sb.value()
        D = self.depth_sb.value()
        B_eff = vesic_result.B_eff

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

        # Failure wedge
        depth_influence = B_eff * 1.5
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

        # Multi-method annotation box
        text_lines = []
        for method in ["vesic", "meyerhof", "hansen"]:
            r = bc_results[method]
            text_lines.append(
                f"{method.capitalize():8s}: q_ult={r.q_ultimate:,.0f}"
                f"  q_all={r.q_allowable:,.0f} kPa"
            )
        ax.text(B / 2 + 0.5, -D / 2,
                "\n".join(text_lines),
                fontsize=9, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="#1976D2", alpha=0.9))

        # Term breakdown bar
        terms = [vesic_result.term_cohesion, vesic_result.term_overburden,
                 vesic_result.term_selfweight]
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
                    ax.text(x_pos + w / 2, y_bar,
                            f"{100 * t / total:.0f}%",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="white")
                x_pos += w
            ax.text(left - 0.3, y_bar, "Terms:", ha="right", va="center",
                    fontsize=8, fontweight="bold")

        ax.set_xlim(-B / 2 - depth_influence - 1,
                    B / 2 + depth_influence + 3)
        ax.set_ylim(-D - depth_influence - 1, 4)
        ax.set_aspect("equal")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Elevation (m)")
        ax.set_title("Bearing Capacity", fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Concrete plan view
        if ax2 is not None and concrete_result is not None:
            self._plot_concrete_plan(ax2, concrete_result)

        fig.tight_layout()
        self.canvas.draw()

    def _plot_concrete_plan(self, ax, cr):
        """Draw plan-view concrete design check on the given axes."""
        B = cr.B_m
        L = cr.L_m
        d = cr.d_m
        col_b = cr.col_b_m
        col_d = cr.col_d_m

        # Footing outline
        ax.add_patch(mpatches.Rectangle(
            (-B / 2, -L / 2), B, L,
            facecolor="#e0e0e0", edgecolor="black", linewidth=2))

        # Column outline (filled)
        ax.add_patch(mpatches.Rectangle(
            (-col_b / 2, -col_d / 2), col_b, col_d,
            facecolor="#757575", edgecolor="black", linewidth=2, zorder=4))
        ax.text(0, 0, "COL", ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=5)

        # One-way shear critical section (at d from column face, both sides)
        ow_color = "#4CAF50" if cr.oneway_ok else "#D32F2F"
        ow_x = col_b / 2 + d
        ax.plot([ow_x, ow_x], [-L / 2, L / 2],
                color=ow_color, linewidth=2, linestyle="--",
                label=f"1-way ({('OK' if cr.oneway_ok else 'FAIL')})")
        ax.plot([-ow_x, -ow_x], [-L / 2, L / 2],
                color=ow_color, linewidth=2, linestyle="--")

        # Two-way shear critical perimeter (at d/2 from column face)
        tw_color = "#1976D2" if cr.twoway_ok else "#D32F2F"
        tw_bx = col_b / 2 + d / 2
        tw_by = col_d / 2 + d / 2
        ax.add_patch(mpatches.Rectangle(
            (-tw_bx, -tw_by), 2 * tw_bx, 2 * tw_by,
            facecolor="none", edgecolor=tw_color, linewidth=2,
            linestyle=":",
            label=f"2-way ({('OK' if cr.twoway_ok else 'FAIL')})"))

        # Rebar lines (along B direction)
        n = cr.n_bars
        cover_m = self.cover_sb.value() / 1000.0
        if n > 1:
            y_start = -L / 2 + cover_m
            y_end = L / 2 - cover_m
            spacing_m = cr.spacing_mm / 1000.0
            for i in range(n):
                x = -B / 2 + cover_m + i * spacing_m
                if x > B / 2 - cover_m:
                    break
                ax.plot([x, x], [y_start, y_end],
                        color="#FF5722", linewidth=0.8, alpha=0.7)

        # Labels
        ax.set_xlim(-B / 2 - 0.2, B / 2 + 0.2)
        ax.set_ylim(-L / 2 - 0.2, L / 2 + 0.2)
        ax.set_aspect("equal")
        ax.set_xlabel("B direction (m)")
        ax.set_ylabel("L direction (m)")
        ax.set_title("Concrete Design — Plan View", fontweight="bold")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

        # Rebar info text
        ax.text(
            0.02, 0.02,
            f"{cr.n_bars} {cr.bar_size} @ {cr.spacing_mm:.0f} mm\n"
            f"As = {cr.As_provided_mm2:.0f} mm\u00b2",
            transform=ax.transAxes, fontsize=8,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.9))

    # ------------------------------------------------------------------
    # State save / load
    # ------------------------------------------------------------------
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
            "q_applied": self.q_applied_sb.value(),
            "Es": self.Es_sb.value(),
            "nu": self.nu_sb.value(),
            "H": self.H_sb.value(),
            "spt_enabled": self.spt_cb.isChecked(),
            "spt_n": self.spt_n_sb.value(),
            "fs": self.fs_sb.value(),
            "concrete_enabled": self.concrete_cb.isChecked(),
            "col_load": self.col_load_sb.value(),
            "col_width": self.col_width_sb.value(),
            "col_depth": self.col_depth_sb.value(),
            "ftg_thick": self.ftg_thick_sb.value(),
            "fc": self.fc_sb.value(),
            "fy": self.fy_sb.value(),
            "cover": self.cover_sb.value(),
            "bar_size": self.bar_size_cb.currentText(),
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
        self.q_applied_sb.setValue(state.get("q_applied", 150.0))
        self.Es_sb.setValue(state.get("Es", 25000))
        self.nu_sb.setValue(state.get("nu", 0.3))
        self.H_sb.setValue(state.get("H", 10.0))
        self.spt_cb.setChecked(state.get("spt_enabled", False))
        self.spt_n_sb.setValue(state.get("spt_n", 20))
        self.fs_sb.setValue(state.get("fs", 3.0))
        self.concrete_cb.setChecked(state.get("concrete_enabled", False))
        self.col_load_sb.setValue(state.get("col_load", 1000))
        self.col_width_sb.setValue(state.get("col_width", 0.4))
        self.col_depth_sb.setValue(state.get("col_depth", 0.4))
        self.ftg_thick_sb.setValue(state.get("ftg_thick", 0.5))
        self.fc_sb.setValue(state.get("fc", 28.0))
        self.fy_sb.setValue(state.get("fy", 420.0))
        self.cover_sb.setValue(state.get("cover", 75.0))
        if state.get("bar_size"):
            self.bar_size_cb.setCurrentText(state["bar_size"])
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
