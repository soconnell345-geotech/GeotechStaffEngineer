"""DXF layer mapping dialog — shared by slope stability and FEM panels.

Presents DXF layer discovery results and lets the user assign roles
(Surface, Boundary, Water Table, Nails, Skip) and soil properties.
"""

from qt_panels.common import (
    QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QComboBox, QCheckBox, QTableWidget,
    QTableWidgetItem, QHeaderView, Qt,
)

# Roles available for each DXF layer
ROLE_CHOICES = ["Skip", "Surface", "Boundary", "Water Table", "Nails"]


class DxfLayerMappingDialog(QDialog):
    """Modal dialog for mapping DXF layers to slope/FEM geometry roles.

    Parameters
    ----------
    parent : QWidget
    discovery_result : DxfDiscoveryResult
    mode : str
        "slope" — show slope-specific soil property columns.
        "fem" — show FEM-specific soil property columns.
    """

    def __init__(self, parent, discovery_result, mode="slope"):
        super().__init__(parent)
        self._discovery = discovery_result
        self._mode = mode
        self.setWindowTitle("Import DXF — Layer Mapping")
        self.setMinimumSize(700, 500)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Header ---
        hdr = QLabel(
            f"File: {self._discovery.filepath}\n"
            f"Entities: {self._discovery.n_total_entities}   "
            f"Layers: {self._discovery.n_layers}"
        )
        hdr.setWordWrap(True)
        layout.addWidget(hdr)

        # --- Units + Flip Y row ---
        opts_row = QHBoxLayout()
        opts_row.addWidget(QLabel("Units:"))
        self.units_cb = QComboBox()
        self.units_cb.addItems(["m", "mm", "cm", "ft", "in"])
        if self._discovery.units_hint:
            self.units_cb.setCurrentText(self._discovery.units_hint)
        opts_row.addWidget(self.units_cb)
        self.flip_y_cb = QCheckBox("Flip Y axis")
        opts_row.addWidget(self.flip_y_cb)
        opts_row.addStretch()
        layout.addLayout(opts_row)

        # --- Layer role table ---
        grp1 = QGroupBox("Layer Roles")
        g1l = QVBoxLayout()
        n_layers = len(self._discovery.layers)
        self.layer_table = QTableWidget(n_layers, 5)
        self.layer_table.setHorizontalHeaderLabels([
            "Layer Name", "Entities", "Types", "Role", "Soil Name",
        ])
        self.layer_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)

        self._role_combos = []
        for i, lyr in enumerate(self._discovery.layers):
            # Name (read-only)
            name_item = QTableWidgetItem(lyr.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.layer_table.setItem(i, 0, name_item)

            # Entity count
            cnt_item = QTableWidgetItem(str(lyr.n_entities))
            cnt_item.setFlags(cnt_item.flags() & ~Qt.ItemIsEditable)
            self.layer_table.setItem(i, 1, cnt_item)

            # Entity types
            types_str = ", ".join(
                f"{k}:{v}" for k, v in sorted(lyr.entity_types.items()))
            types_item = QTableWidgetItem(types_str)
            types_item.setFlags(types_item.flags() & ~Qt.ItemIsEditable)
            self.layer_table.setItem(i, 2, types_item)

            # Role combo
            role_cb = QComboBox()
            role_cb.addItems(ROLE_CHOICES)
            # Auto-guess role from layer name
            name_lower = lyr.name.lower()
            if "surface" in name_lower or "ground" in name_lower:
                role_cb.setCurrentText("Surface")
            elif "water" in name_lower or "gwt" in name_lower:
                role_cb.setCurrentText("Water Table")
            elif "nail" in name_lower:
                role_cb.setCurrentText("Nails")
            elif "bound" in name_lower or "soil" in name_lower or "layer" in name_lower:
                role_cb.setCurrentText("Boundary")
            self._role_combos.append(role_cb)
            self.layer_table.setCellWidget(i, 3, role_cb)

            # Soil name (editable, for Boundary role)
            soil_item = QTableWidgetItem(lyr.name)
            self.layer_table.setItem(i, 4, soil_item)

        g1l.addWidget(self.layer_table)
        grp1.setLayout(g1l)
        layout.addWidget(grp1)

        # --- Soil properties table ---
        grp2 = QGroupBox("Soil Properties")
        g2l = QVBoxLayout()
        info = QLabel(
            "Assign properties to each soil layer. The top layer (above "
            "first boundary) is auto-named 'Surface'."
        )
        info.setWordWrap(True)
        g2l.addWidget(info)

        if self._mode == "slope":
            cols = ["Name", "γ (kN/m³)", "φ (deg)", "c' (kPa)",
                    "cu (kPa)", "Mode"]
            n_cols = 6
        else:
            cols = ["Name", "γ (kN/m³)", "E (kPa)", "ν",
                    "c (kPa)", "φ (deg)", "ψ (deg)", "Model"]
            n_cols = 8

        self.soil_table = QTableWidget(0, n_cols)
        self.soil_table.setHorizontalHeaderLabels(cols)
        self.soil_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.soil_table.setMaximumHeight(200)

        # Pre-populate: Surface row + one row per layer with Boundary role
        self._refresh_soil_table()

        # Connect role changes to refresh soil table
        for cb in self._role_combos:
            cb.currentIndexChanged.connect(self._refresh_soil_table)

        g2l.addWidget(self.soil_table)
        grp2.setLayout(g2l)
        layout.addWidget(grp2)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_soil_table(self):
        """Rebuild soil properties table rows from current role assignments."""
        # Collect names: "Surface" + all Boundary soil names
        names = ["Surface"]
        for i, cb in enumerate(self._role_combos):
            if cb.currentText() == "Boundary":
                soil_item = self.layer_table.item(i, 4)
                name = soil_item.text() if soil_item and soil_item.text() else f"Layer_{i}"
                if name not in names:
                    names.append(name)

        # Preserve existing values
        existing = {}
        for r in range(self.soil_table.rowCount()):
            name_item = self.soil_table.item(r, 0)
            if name_item:
                row_vals = []
                for c in range(self.soil_table.columnCount()):
                    it = self.soil_table.item(r, c)
                    row_vals.append(it.text() if it else "")
                existing[name_item.text()] = row_vals

        self.soil_table.setRowCount(len(names))
        for r, name in enumerate(names):
            if name in existing:
                for c, val in enumerate(existing[name]):
                    self.soil_table.setItem(r, c, QTableWidgetItem(val))
            else:
                # Defaults
                if self._mode == "slope":
                    defaults = [name, "18", "30", "5", "0", "drained"]
                else:
                    defaults = [name, "18", "30000", "0.3", "10", "25", "0", "mc"]
                for c, val in enumerate(defaults):
                    self.soil_table.setItem(r, c, QTableWidgetItem(val))

    def _on_accept(self):
        """Validate and accept."""
        # Check at least one Surface role
        has_surface = any(
            cb.currentText() == "Surface" for cb in self._role_combos)
        if not has_surface:
            from qt_panels.common import QMessageBox
            QMessageBox.warning(
                self, "Validation Error",
                "At least one layer must be assigned the 'Surface' role.")
            return

        # Check gamma > 0 for all soil rows
        for r in range(self.soil_table.rowCount()):
            gamma_item = self.soil_table.item(r, 1)
            try:
                gamma = float(gamma_item.text()) if gamma_item else 0
                if gamma <= 0:
                    from qt_panels.common import QMessageBox
                    name = self.soil_table.item(r, 0).text()
                    QMessageBox.warning(
                        self, "Validation Error",
                        f"Layer '{name}': γ must be positive.")
                    return
            except ValueError:
                from qt_panels.common import QMessageBox
                QMessageBox.warning(
                    self, "Validation Error",
                    f"Row {r + 1}: invalid γ value.")
                return

        self.accept()

    def get_results(self):
        """Return (LayerMapping, soil_properties, nail_defaults, units, flip_y).

        Returns
        -------
        layer_mapping : LayerMapping
        soil_properties : list of SoilPropertyAssignment or FEMSoilPropertyAssignment
        nail_defaults : dict or None
        units : str
        flip_y : bool
        """
        from dxf_import import LayerMapping

        surface_layer = ""
        soil_boundaries = {}
        water_table = None
        nails_layer = None

        for i, cb in enumerate(self._role_combos):
            role = cb.currentText()
            dxf_name = self._discovery.layers[i].name
            if role == "Surface":
                surface_layer = dxf_name
            elif role == "Boundary":
                soil_item = self.layer_table.item(i, 4)
                soil_name = soil_item.text() if soil_item else dxf_name
                soil_boundaries[dxf_name] = soil_name
            elif role == "Water Table":
                water_table = dxf_name
            elif role == "Nails":
                nails_layer = dxf_name

        mapping = LayerMapping(
            surface=surface_layer,
            soil_boundaries=soil_boundaries,
            water_table=water_table,
            nails=nails_layer,
        )

        # Build soil properties
        soil_properties = []
        if self._mode == "slope":
            from dxf_import import SoilPropertyAssignment
            for r in range(self.soil_table.rowCount()):
                vals = [
                    (self.soil_table.item(r, c).text()
                     if self.soil_table.item(r, c) else "")
                    for c in range(self.soil_table.columnCount())
                ]
                soil_properties.append(SoilPropertyAssignment(
                    name=vals[0],
                    gamma=float(vals[1]) if vals[1] else 18.0,
                    phi=float(vals[2]) if vals[2] else 0.0,
                    c_prime=float(vals[3]) if vals[3] else 0.0,
                    cu=float(vals[4]) if vals[4] else 0.0,
                    analysis_mode=vals[5] if vals[5] else "drained",
                ))
        else:
            from dxf_import import FEMSoilPropertyAssignment
            for r in range(self.soil_table.rowCount()):
                vals = [
                    (self.soil_table.item(r, c).text()
                     if self.soil_table.item(r, c) else "")
                    for c in range(self.soil_table.columnCount())
                ]
                soil_properties.append(FEMSoilPropertyAssignment(
                    name=vals[0],
                    gamma=float(vals[1]) if vals[1] else 18.0,
                    E=float(vals[2]) if vals[2] else 30000.0,
                    nu=float(vals[3]) if vals[3] else 0.3,
                    c=float(vals[4]) if vals[4] else 0.0,
                    phi=float(vals[5]) if vals[5] else 0.0,
                    psi=float(vals[6]) if vals[6] else 0.0,
                    model=vals[7] if vals[7] else "mc",
                ))

        units = self.units_cb.currentText()
        flip_y = self.flip_y_cb.isChecked()

        return mapping, soil_properties, None, units, flip_y
