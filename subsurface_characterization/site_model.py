"""
Site investigation data model.

Internal data model for subsurface characterization. All input formats
(DIGGS XML, CSV, dicts, pygef CPTParseResult) populate this model.
Visualizations work from the model, not tied to any input format.

Classes
-------
PointMeasurement : Discrete field/lab measurement at a specific depth
LithologyInterval : Interpreted soil/rock layer from field log
Investigation : Single subsurface investigation location
SiteModel : Top-level container for entire site investigation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np


# ---------------------------------------------------------------------------
# Standard parameter names
# ---------------------------------------------------------------------------

STANDARD_PARAMETERS = {
    "N_spt", "N60", "N160",
    "qc_kPa", "fs_kPa", "u2_kPa", "Rf_pct",
    "cu_kPa", "phi_deg", "qu_kPa", "RQD_pct",
    "gamma_kNm3", "wn_pct", "LL_pct", "PL_pct", "PI_pct",
    "e0", "Cc", "Cr", "sigma_p_kPa",
    "Su_vane_kPa", "pocket_pen_kPa",
}


# ---------------------------------------------------------------------------
# PointMeasurement
# ---------------------------------------------------------------------------

@dataclass
class PointMeasurement:
    """Discrete field/lab measurement at a specific depth.

    NOT layer-averaged — represents a single test result at a point.

    Parameters
    ----------
    depth_m : float
        Depth below ground surface (m).
    parameter : str
        Standard parameter name (e.g., 'N_spt', 'qc_kPa', 'cu_kPa').
    value : float
        Measured value in standard units.
    source : str
        'field' or 'lab'.
    test_type : str
        Test method (e.g., 'SPT', 'CPTu', 'vane_shear').
    sample_id : str
        Sample identifier.
    notes : str
        Additional notes.
    """

    depth_m: float = 0.0
    parameter: str = ""
    value: float = 0.0
    source: str = "field"
    test_type: str = ""
    sample_id: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "depth_m": round(self.depth_m, 3),
            "parameter": self.parameter,
            "value": round(self.value, 4) if isinstance(self.value, float) else self.value,
            "source": self.source,
            "test_type": self.test_type,
            "sample_id": self.sample_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PointMeasurement:
        """Create from dict."""
        return cls(
            depth_m=float(data.get("depth_m", 0)),
            parameter=str(data.get("parameter", "")),
            value=float(data.get("value", 0)),
            source=str(data.get("source", "field")),
            test_type=str(data.get("test_type", "")),
            sample_id=str(data.get("sample_id", "")),
            notes=str(data.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# LithologyInterval
# ---------------------------------------------------------------------------

@dataclass
class LithologyInterval:
    """Interpreted soil/rock layer from field log.

    Parameters
    ----------
    top_depth_m : float
        Depth to top of interval (m).
    bottom_depth_m : float
        Depth to bottom of interval (m).
    description : str
        Soil/rock description.
    uscs : str
        USCS classification symbol (e.g., 'CL', 'SP', 'CH').
    color : str
        Soil color description.
    moisture : str
        Moisture condition (e.g., 'dry', 'moist', 'wet').
    consistency_density : str
        Consistency (cohesive) or density (granular) description.
    """

    top_depth_m: float = 0.0
    bottom_depth_m: float = 0.0
    description: str = ""
    uscs: str = ""
    color: str = ""
    moisture: str = ""
    consistency_density: str = ""

    @property
    def thickness_m(self) -> float:
        """Layer thickness (m)."""
        return self.bottom_depth_m - self.top_depth_m

    @property
    def mid_depth_m(self) -> float:
        """Depth to center of layer (m)."""
        return (self.top_depth_m + self.bottom_depth_m) / 2.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "top_depth_m": round(self.top_depth_m, 3),
            "bottom_depth_m": round(self.bottom_depth_m, 3),
            "thickness_m": round(self.thickness_m, 3),
            "description": self.description,
            "uscs": self.uscs,
            "color": self.color,
            "moisture": self.moisture,
            "consistency_density": self.consistency_density,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LithologyInterval:
        """Create from dict."""
        return cls(
            top_depth_m=float(data.get("top_depth_m", 0)),
            bottom_depth_m=float(data.get("bottom_depth_m", 0)),
            description=str(data.get("description", "")),
            uscs=str(data.get("uscs", "")),
            color=str(data.get("color", "")),
            moisture=str(data.get("moisture", "")),
            consistency_density=str(data.get("consistency_density", "")),
        )


# ---------------------------------------------------------------------------
# Investigation
# ---------------------------------------------------------------------------

@dataclass
class Investigation:
    """Single subsurface investigation location (boring, CPT, test pit, etc.).

    Parameters
    ----------
    investigation_id : str
        Unique identifier (e.g., 'B-1', 'CPT-3', 'TP-2').
    investigation_type : str
        'boring', 'cpt', 'test_pit', 'monitoring_well', or custom.
    x : float
        X coordinate (easting or longitude).
    y : float
        Y coordinate (northing or latitude).
    elevation_m : float
        Ground surface elevation (m).
    total_depth_m : float
        Total depth of investigation (m).
    gwl_depth_m : float or None
        Groundwater level depth below ground surface (m).
    coordinate_system : str
        Coordinate reference system (e.g., 'EPSG:4326').
    date_started : str
        Date investigation started (ISO 8601 or free text).
    drilling_method : str
        Drilling or investigation method.
    lithology : list of LithologyInterval
        Interpreted soil/rock layers.
    measurements : list of PointMeasurement
        Discrete measurements at depth.
    notes : str
        Additional notes.
    """

    investigation_id: str = ""
    investigation_type: str = "boring"
    x: float = 0.0
    y: float = 0.0
    elevation_m: float = 0.0
    total_depth_m: float = 0.0
    gwl_depth_m: Optional[float] = None
    coordinate_system: str = ""
    date_started: str = ""
    drilling_method: str = ""
    lithology: List[LithologyInterval] = field(default_factory=list)
    measurements: List[PointMeasurement] = field(default_factory=list)
    notes: str = ""

    def get_measurements(self, parameter: str) -> List[PointMeasurement]:
        """Get all measurements for a parameter, sorted by depth."""
        return sorted(
            [m for m in self.measurements if m.parameter == parameter],
            key=lambda m: m.depth_m,
        )

    def get_parameter_arrays(self, parameter: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get depth and value arrays for a parameter.

        Returns (depth_array, value_array) as numpy arrays, sorted by depth.
        """
        meas = self.get_measurements(parameter)
        if not meas:
            return np.array([]), np.array([])
        depths = np.array([m.depth_m for m in meas])
        values = np.array([m.value for m in meas])
        return depths, values

    def available_parameters(self) -> List[str]:
        """Sorted list of unique parameter names in this investigation."""
        return sorted(set(m.parameter for m in self.measurements))

    def depth_to_rock_m(self) -> Optional[float]:
        """Depth to first rock layer (m), or None if no rock found."""
        for layer in sorted(self.lithology, key=lambda l: l.top_depth_m):
            desc_lower = layer.description.lower()
            uscs_upper = layer.uscs.upper()
            if "rock" in desc_lower or "bedrock" in desc_lower or uscs_upper == "R":
                return layer.top_depth_m
        return None

    def fill_thickness_m(self) -> Optional[float]:
        """Thickness of fill material (m), or None if no fill found."""
        for layer in sorted(self.lithology, key=lambda l: l.top_depth_m):
            desc_lower = layer.description.lower()
            if "fill" in desc_lower:
                return layer.thickness_m
        return None

    def uscs_at_depth(self, depth_m: float) -> str:
        """USCS classification at given depth, or '' if not found."""
        for layer in self.lithology:
            if layer.top_depth_m <= depth_m < layer.bottom_depth_m:
                return layer.uscs
        return ""

    def to_soil_profile(self):
        """Convert to geotech_common.SoilProfile for use with analysis modules.

        Requires lithology with USCS and at least some measurements.
        """
        from geotech_common.soil_profile import SoilProfile, SoilLayer, GroundwaterCondition

        layers = []
        for lith in sorted(self.lithology, key=lambda l: l.top_depth_m):
            # Try to find representative measurements within this layer
            layer_cu = None
            layer_phi = None
            layer_N = None
            layer_gamma = None

            for m in self.measurements:
                if lith.top_depth_m <= m.depth_m < lith.bottom_depth_m:
                    if m.parameter == "cu_kPa":
                        layer_cu = m.value
                    elif m.parameter == "phi_deg":
                        layer_phi = m.value
                    elif m.parameter == "N_spt":
                        layer_N = m.value
                    elif m.parameter == "gamma_kNm3":
                        layer_gamma = m.value

            # Determine cohesive from USCS
            cohesive_uscs = {"CH", "CL", "MH", "ML", "OH", "OL", "PT"}
            is_cohesive = lith.uscs.upper() in cohesive_uscs if lith.uscs else None

            is_rock = (
                "rock" in lith.description.lower()
                or "bedrock" in lith.description.lower()
                or lith.uscs.upper() == "R"
            )

            sl = SoilLayer(
                top_depth=lith.top_depth_m,
                bottom_depth=lith.bottom_depth_m,
                description=lith.description,
                uscs=lith.uscs,
                is_cohesive=is_cohesive,
                is_rock=is_rock,
                cu=layer_cu,
                phi=layer_phi,
                N_spt=layer_N,
                gamma=layer_gamma,
            )
            layers.append(sl)

        gwl_depth = self.gwl_depth_m if self.gwl_depth_m is not None else 30.0
        gw = GroundwaterCondition(depth=gwl_depth)

        return SoilProfile(
            layers=layers,
            groundwater=gw,
            location_name=self.investigation_id,
            boring_id=self.investigation_id,
        )

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "investigation_id": self.investigation_id,
            "investigation_type": self.investigation_type,
            "x": self.x,
            "y": self.y,
            "elevation_m": round(self.elevation_m, 3),
            "total_depth_m": round(self.total_depth_m, 3),
            "gwl_depth_m": round(self.gwl_depth_m, 3) if self.gwl_depth_m is not None else None,
            "coordinate_system": self.coordinate_system,
            "date_started": self.date_started,
            "drilling_method": self.drilling_method,
            "n_lithology": len(self.lithology),
            "n_measurements": len(self.measurements),
            "available_parameters": self.available_parameters(),
            "lithology": [l.to_dict() for l in self.lithology],
            "measurements": [m.to_dict() for m in self.measurements],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Investigation:
        """Create from dict."""
        lith = [LithologyInterval.from_dict(l) for l in data.get("lithology", [])]
        meas = [PointMeasurement.from_dict(m) for m in data.get("measurements", [])]
        return cls(
            investigation_id=str(data.get("investigation_id", "")),
            investigation_type=str(data.get("investigation_type", "boring")),
            x=float(data.get("x", 0)),
            y=float(data.get("y", 0)),
            elevation_m=float(data.get("elevation_m", 0)),
            total_depth_m=float(data.get("total_depth_m", 0)),
            gwl_depth_m=float(data["gwl_depth_m"]) if data.get("gwl_depth_m") is not None else None,
            coordinate_system=str(data.get("coordinate_system", "")),
            date_started=str(data.get("date_started", "")),
            drilling_method=str(data.get("drilling_method", "")),
            lithology=lith,
            measurements=meas,
            notes=str(data.get("notes", "")),
        )

    def summary(self) -> str:
        """One-line text summary."""
        gwl_str = f", GWL={self.gwl_depth_m:.1f}m" if self.gwl_depth_m is not None else ""
        params = ", ".join(self.available_parameters()) or "none"
        return (
            f"{self.investigation_id} ({self.investigation_type}): "
            f"depth={self.total_depth_m:.1f}m, "
            f"{len(self.measurements)} measurements ({params}), "
            f"{len(self.lithology)} layers{gwl_str}"
        )


# ---------------------------------------------------------------------------
# SiteModel
# ---------------------------------------------------------------------------

@dataclass
class SiteModel:
    """Top-level container for entire site investigation.

    Parameters
    ----------
    project_name : str
        Project name or identifier.
    investigations : list of Investigation
        All investigation locations.
    coordinate_system : str
        Site-wide coordinate reference system.
    datum : str
        Vertical datum.
    notes : str
        Additional notes.
    """

    project_name: str = ""
    investigations: List[Investigation] = field(default_factory=list)
    coordinate_system: str = ""
    datum: str = ""
    notes: str = ""

    def get_investigation(self, investigation_id: str) -> Investigation:
        """Get investigation by ID. Raises KeyError if not found."""
        for inv in self.investigations:
            if inv.investigation_id == investigation_id:
                return inv
        available = ", ".join(self.investigation_ids())
        raise KeyError(
            f"Investigation '{investigation_id}' not found. Available: {available}"
        )

    def investigation_ids(self) -> List[str]:
        """Sorted list of investigation IDs."""
        return sorted(inv.investigation_id for inv in self.investigations)

    def borings(self) -> List[Investigation]:
        """All boring-type investigations."""
        return [inv for inv in self.investigations if inv.investigation_type == "boring"]

    def cpts(self) -> List[Investigation]:
        """All CPT-type investigations."""
        return [inv for inv in self.investigations if inv.investigation_type == "cpt"]

    def all_measurements(self, parameter: str) -> List[Tuple[str, PointMeasurement]]:
        """Get all measurements for a parameter across all investigations.

        Returns list of (investigation_id, PointMeasurement) tuples.
        """
        results = []
        for inv in self.investigations:
            for m in inv.get_measurements(parameter):
                results.append((inv.investigation_id, m))
        return results

    def available_parameters(self) -> List[str]:
        """Union of all parameters across all investigations."""
        params = set()
        for inv in self.investigations:
            params.update(inv.available_parameters())
        return sorted(params)

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """(x_min, y_min, x_max, y_max) of all investigation locations."""
        if not self.investigations:
            return (0.0, 0.0, 0.0, 0.0)
        xs = [inv.x for inv in self.investigations]
        ys = [inv.y for inv in self.investigations]
        return (min(xs), min(ys), max(xs), max(ys))

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "project_name": self.project_name,
            "n_investigations": len(self.investigations),
            "coordinate_system": self.coordinate_system,
            "datum": self.datum,
            "available_parameters": self.available_parameters(),
            "investigations": [inv.to_dict() for inv in self.investigations],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SiteModel:
        """Create from dict."""
        invs = [Investigation.from_dict(i) for i in data.get("investigations", [])]
        return cls(
            project_name=str(data.get("project_name", "")),
            investigations=invs,
            coordinate_system=str(data.get("coordinate_system", "")),
            datum=str(data.get("datum", "")),
            notes=str(data.get("notes", "")),
        )

    def summary(self) -> str:
        """Multi-line text summary."""
        lines = [
            "=" * 60,
            f"  SITE MODEL: {self.project_name}",
            "=" * 60,
            f"  Investigations: {len(self.investigations)}",
            f"  Borings: {len(self.borings())}",
            f"  CPTs: {len(self.cpts())}",
            f"  Parameters: {', '.join(self.available_parameters()) or 'none'}",
        ]
        if self.coordinate_system:
            lines.append(f"  Coordinate system: {self.coordinate_system}")
        lines.append("")
        for inv in self.investigations:
            lines.append(f"  {inv.summary()}")
        lines.append("")
        return "\n".join(lines)
