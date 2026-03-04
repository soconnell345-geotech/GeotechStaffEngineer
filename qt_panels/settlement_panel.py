"""Settlement analysis panel."""

import traceback
import numpy as np

from qt_panels.common import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout, QGroupBox,
    QDoubleSpinBox, QPushButton,
    QScrollArea, QSplitter, Qt,
    MplCanvas, make_results_box, NavigationToolbar2QT,
)


class SettlementPanel(QWidget):
    """Elastic + consolidation settlement analysis."""

    def __init__(self, status_bar):
        super().__init__()
        self._status = status_bar
        self._build_ui()

    def _build_ui(self):
        outer = QHBoxLayout(self)

        # --- Left: inputs ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumWidth(380)
        form_widget = QWidget()
        form = QVBoxLayout(form_widget)

        # Loading
        lg = QGroupBox("Loading")
        ll = QFormLayout()
        self.q_sb = QDoubleSpinBox(value=150.0, minimum=0.0, maximum=5000,
                                   singleStep=10, decimals=1, suffix=" kPa")
        self.B_sb = QDoubleSpinBox(value=3.0, minimum=0.1, maximum=100,
                                   singleStep=0.5, decimals=2, suffix=" m")
        self.L_sb = QDoubleSpinBox(value=3.0, minimum=0.1, maximum=200,
                                   singleStep=0.5, decimals=2, suffix=" m")
        self.D_sb = QDoubleSpinBox(value=1.5, minimum=0.0, maximum=50,
                                   singleStep=0.5, decimals=2, suffix=" m")
        ll.addRow("Applied stress q:", self.q_sb)
        ll.addRow("Footing width B:", self.B_sb)
        ll.addRow("Footing length L:", self.L_sb)
        ll.addRow("Embedment depth D:", self.D_sb)
        lg.setLayout(ll)
        form.addWidget(lg)

        # Elastic settlement
        eg = QGroupBox("Elastic Settlement")
        el = QFormLayout()
        self.Es_sb = QDoubleSpinBox(value=25000, minimum=100, maximum=1e6,
                                    singleStep=1000, decimals=0, suffix=" kPa")
        self.nu_sb = QDoubleSpinBox(value=0.3, minimum=0.0, maximum=0.5,
                                    singleStep=0.05, decimals=2)
        self.H_sb = QDoubleSpinBox(value=10.0, minimum=0.1, maximum=200,
                                   singleStep=1.0, decimals=1, suffix=" m")
        el.addRow("Elastic modulus Es:", self.Es_sb)
        el.addRow("Poisson's ratio \u03bd:", self.nu_sb)
        el.addRow("Layer thickness H:", self.H_sb)
        eg.setLayout(el)
        form.addWidget(eg)

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
        right_splitter.setSizes([350, 200])
        outer.addWidget(right_splitter, 1)

    def _run_analysis(self):
        try:
            from settlement import elastic_settlement, stress_at_depth

            q = self.q_sb.value()
            B = self.B_sb.value()
            L = self.L_sb.value()
            Es = self.Es_sb.value()
            nu = self.nu_sb.value()
            H = self.H_sb.value()

            s_elastic = elastic_settlement(q=q, B=B, Es=Es, nu=nu)

            # Stress distribution for plot
            depths = np.linspace(0.01, H, 50)
            stresses = [stress_at_depth(q, B, L, z) for z in depths]

            lines = [
                "=" * 60,
                "  SETTLEMENT ANALYSIS RESULTS",
                "=" * 60,
                "",
                f"  Applied stress:      q  = {q:.1f} kPa",
                f"  Footing:             B  = {B:.1f} m,  L = {L:.1f} m",
                f"  Elastic modulus:     Es = {Es:,.0f} kPa",
                f"  Poisson's ratio:     nu = {nu:.2f}",
                f"  Layer thickness:     H  = {H:.1f} m",
                "",
                f"  Elastic Settlement:  {s_elastic * 1000:.1f} mm"
                f"  ({s_elastic:.4f} m)",
                "",
                "=" * 60,
            ]
            self.results_text.setText("\n".join(lines))
            self._plot(depths, stresses, s_elastic, q, B)
            self._status.showMessage(
                f"Elastic settlement = {s_elastic * 1000:.1f} mm", 10000)

        except Exception as e:
            self.results_text.setText(f"ERROR:\n{traceback.format_exc()}")
            self._status.showMessage(f"Error: {e}", 10000)

    def _plot(self, depths, stresses, s_elastic, q, B):
        ax = self.canvas.axes
        ax.clear()
        ax.plot(stresses, depths, color="#1976D2", linewidth=2)
        ax.axhline(0, color="#8B4513", linewidth=2)
        ax.fill_betweenx(depths, 0, stresses, alpha=0.15, color="#1976D2")
        ax.invert_yaxis()
        ax.set_xlabel("Vertical Stress Increase (kPa)")
        ax.set_ylabel("Depth below footing (m)")
        ax.set_title(f"Stress Distribution (2:1 Method)\n"
                     f"Settlement = {s_elastic * 1000:.1f} mm",
                     fontweight="bold")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def get_state(self):
        return {
            "q": self.q_sb.value(),
            "B": self.B_sb.value(),
            "L": self.L_sb.value(),
            "D": self.D_sb.value(),
            "Es": self.Es_sb.value(),
            "nu": self.nu_sb.value(),
            "H": self.H_sb.value(),
            "results_text": self.results_text.toPlainText(),
        }

    def set_state(self, state):
        self.q_sb.setValue(state.get("q", 150.0))
        self.B_sb.setValue(state.get("B", 3.0))
        self.L_sb.setValue(state.get("L", 3.0))
        self.D_sb.setValue(state.get("D", 1.5))
        self.Es_sb.setValue(state.get("Es", 25000))
        self.nu_sb.setValue(state.get("nu", 0.3))
        self.H_sb.setValue(state.get("H", 10.0))
        if state.get("results_text"):
            self.results_text.setText(state["results_text"])
