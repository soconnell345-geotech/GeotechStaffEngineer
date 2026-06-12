"""Canonical, versioned Project document for 2D geotech model setup.

The Project is the single source of truth the MODEL-SETUP agent edits and the
human confirms. It carries everything needed to build a slope_stability
SlopeGeometry and/or fem2d analyze_slope_srm inputs:

* geometry        — ground surface + named layer-boundary polylines, with a
                    PROVENANCE tag ('user'|'dxf'|'pdf_vector'|'template'|
                    'vision_draft'). Vision-derived geometry is quarantined:
                    validate() blocks analysis until a human confirms it.
* stratigraphy    — layers top-to-bottom, each with a Material (strength model
                    mohr_coulomb | undrained | shansep | hoek_brown, unit
                    weights, optional FEM stiffness, optional per-parameter
                    probabilistic spec with a SOURCE citation).
* water           — GWT polyline, ru, declared ponding.
* loads           — surcharges + horizontal seismic coefficient kh.
* reinforcement   — nails / anchors / geosynthetics (slope_stability layouts).
* analyses        — requested LE and/or FEM-SRM runs with their settings.
* confirmations   — the staged human gates (geometry / materials /
                    water_loads). Only the confirmation tool sets these.
* assumptions     — the explicit assumption ledger (every default taken, with
                    its source).

Field names mirror the downstream module APIs (gamma, phi, c_prime, cu,
gwt_points, E, nu, psi, kh ...) so the builders are near-mechanical.

JSON round-trip: ``project.to_json()`` / ``Project.from_json(s)``. Documents
carry ``schema_version``; unknown keys are TOLERATED on load (forward
compatibility) — they are simply dropped.

All units SI: meters, kPa, kN/m3, degrees, kN/m of slope run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

#: Where a geometry came from. 'vision_draft' is the quarantined one — see
#: validate.check_vision_draft.
PROVENANCES = ("user", "dxf", "pdf_vector", "template", "vision_draft")

#: Material strength models. 'undrained' maps to SlopeSoilLayer
#: analysis_mode='undrained' (phi=0, cu); the other three map to the layer's
#: strength_model field directly.
STRENGTH_MODELS = ("mohr_coulomb", "undrained", "shansep", "hoek_brown")

#: The staged human-confirmation gates, in protocol order.
CONFIRMATION_STAGES = ("geometry", "materials", "water_loads")


# ---------------------------------------------------------------------------
# Load helpers (unknown-key tolerance)
# ---------------------------------------------------------------------------

def _known_kwargs(cls, data: Optional[dict]) -> dict:
    """Filter ``data`` to the dataclass fields of ``cls`` (drop unknowns)."""
    if not isinstance(data, dict):
        return {}
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _points(seq) -> List[Tuple[float, float]]:
    """Normalize a JSON point list ([[x, z], ...]) to [(x, z), ...]."""
    if seq is None:
        return []
    return [(float(p[0]), float(p[1])) for p in seq]


def _opt_points(seq) -> Optional[List[Tuple[float, float]]]:
    if seq is None:
        return None
    return _points(seq)


# ---------------------------------------------------------------------------
# Meta / geometry
# ---------------------------------------------------------------------------

@dataclass
class ProjectMeta:
    """Document metadata.

    Attributes
    ----------
    name, description : str
    units : str
        Always 'SI' (m, kPa, kN/m3, degrees). Stored for explicitness.
    schema_version : int
        Version of this document format.
    """
    name: str = "Untitled project"
    description: str = ""
    units: str = "SI"
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "ProjectMeta":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Geometry:
    """Section geometry: ground surface + named layer-boundary polylines.

    Attributes
    ----------
    surface_points : list of (x, z)
        Ground surface left-to-right (x strictly increasing).
    layer_boundaries : dict name -> list of (x, z)
        Named boundary polylines. A Layer references one by name as its
        BOTTOM (``Layer.bottom_boundary``). Polylines sorted by x.
    provenance : str
        One of PROVENANCES. 'vision_draft' blocks analysis until the
        geometry stage is human-confirmed (the design inversion: the agent
        never claims to have read geometry correctly).
    """
    surface_points: List[Tuple[float, float]] = field(default_factory=list)
    layer_boundaries: Dict[str, List[Tuple[float, float]]] = field(
        default_factory=dict)
    provenance: str = "user"

    def __post_init__(self):
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"provenance must be one of {PROVENANCES}, "
                f"got '{self.provenance}'")

    @property
    def x_range(self) -> Tuple[float, float]:
        """(x_min, x_max) of the ground surface (0, 0 when empty)."""
        if not self.surface_points:
            return (0.0, 0.0)
        return (self.surface_points[0][0], self.surface_points[-1][0])

    @property
    def z_surface_range(self) -> Tuple[float, float]:
        """(z_min, z_max) of the ground surface (0, 0 when empty)."""
        if not self.surface_points:
            return (0.0, 0.0)
        zs = [z for _, z in self.surface_points]
        return (min(zs), max(zs))

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Geometry":
        kw = _known_kwargs(cls, data)
        kw["surface_points"] = _points(kw.get("surface_points"))
        kw["layer_boundaries"] = {
            str(name): _points(pts)
            for name, pts in (kw.get("layer_boundaries") or {}).items()
        }
        return cls(**kw)


# ---------------------------------------------------------------------------
# Materials / stratigraphy
# ---------------------------------------------------------------------------

@dataclass
class Material:
    """Soil/rock material for one layer.

    Strength models (``strength_model``):

    * ``mohr_coulomb`` — drained c'-phi': needs ``phi`` AND ``c_prime``
      (either may be 0, but both must be set deliberately).
    * ``undrained``    — total-stress phi=0: needs ``cu`` > 0.
    * ``shansep``      — su = S * OCR^m * sigma'_v: needs ``shansep_S``,
      ``shansep_m``, ``ocr`` (the OCR needs a source — see validate).
    * ``hoek_brown``   — GHB rock mass: needs ``hb_sigci`` (kPa), ``hb_gsi``,
      ``hb_mi``, ``hb_D``.

    ``gamma`` is required for ANY analysis; ``gamma_sat`` defaults to gamma.
    FEM (SRM) additionally needs ``E`` (kPa) and ``nu``.

    ``probabilistic`` is an optional per-parameter uncertainty spec used by
    the LE probabilistic runs::

        {"phi": {"cov": 0.10, "dist": "lognormal",
                 "source": "Duncan (2000) Table 1"}}

    Keys follow slope_stability.probabilistic: 'phi', 'c_prime', 'cu',
    'gamma'. COV is a FRACTION here (cov_lookup returns percent — divide by
    100); each entry should carry its published ``source``.

    Optional fields are ``None`` when not yet set — validate() reports what
    is missing FOR THE REQUESTED ANALYSES instead of guessing defaults.
    """
    strength_model: str = "mohr_coulomb"
    gamma: Optional[float] = None
    gamma_sat: Optional[float] = None
    # mohr_coulomb
    phi: Optional[float] = None
    c_prime: Optional[float] = None
    # undrained
    cu: Optional[float] = None
    # shansep
    shansep_S: Optional[float] = None
    shansep_m: Optional[float] = None
    ocr: Optional[float] = None
    su_min: float = 0.0
    # hoek_brown
    hb_sigci: Optional[float] = None
    hb_gsi: Optional[float] = None
    hb_mi: Optional[float] = None
    hb_D: float = 0.0
    # FEM stiffness (optional)
    E: Optional[float] = None
    nu: Optional[float] = None
    psi: float = 0.0
    # probabilistic spec: {param: {"cov": frac, "dist": ..., "source": ...}}
    probabilistic: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        if self.strength_model not in STRENGTH_MODELS:
            raise ValueError(
                f"strength_model must be one of {STRENGTH_MODELS}, "
                f"got '{self.strength_model}'")

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Material":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Layer:
    """One stratigraphic layer (ordered top to bottom in the project).

    The TOP of a layer is the ground surface (first layer) or the bottom of
    the layer above. The BOTTOM is either a flat ``bottom_elevation`` or a
    named polyline in ``geometry.layer_boundaries`` (``bottom_boundary``);
    when both are given the polyline wins and ``bottom_elevation`` is the
    flat fallback used for elevation bookkeeping.
    """
    name: str = ""
    material: Material = field(default_factory=Material)
    top_elevation: Optional[float] = None
    bottom_elevation: Optional[float] = None
    bottom_boundary: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Layer":
        kw = _known_kwargs(cls, data)
        kw["material"] = Material.from_dict(kw.get("material"))
        return cls(**kw)


# ---------------------------------------------------------------------------
# Water / loads / reinforcement
# ---------------------------------------------------------------------------

@dataclass
class Water:
    """Groundwater: GWT polyline, ru coefficient, declared ponding.

    ``ponded=True`` declares that GWT above the ground surface is
    INTENTIONAL (e.g. submerged toe / impoundment); validate() then reports
    ponding as info instead of a warning.
    """
    gwt_points: Optional[List[Tuple[float, float]]] = None
    ru: float = 0.0
    ponded: bool = False

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Water":
        kw = _known_kwargs(cls, data)
        kw["gwt_points"] = _opt_points(kw.get("gwt_points"))
        return cls(**kw)


@dataclass
class Surcharge:
    """A uniform vertical surcharge band on the ground surface (kPa)."""
    q: float = 0.0
    x_start: Optional[float] = None
    x_end: Optional[float] = None
    label: str = ""

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Surcharge":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Loads:
    """External loads: surcharges + horizontal seismic coefficient.

    NOTE: slope_stability's SlopeGeometry carries ONE uniform surcharge band;
    the LE builder maps the FIRST surcharge and validate() warns on extras.
    """
    surcharges: List[Surcharge] = field(default_factory=list)
    kh: float = 0.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Loads":
        kw = _known_kwargs(cls, data)
        kw["surcharges"] = [Surcharge.from_dict(s)
                            for s in (kw.get("surcharges") or [])]
        return cls(**kw)


@dataclass
class Nail:
    """Soil nail layout (mirrors slope_stability.nails.SoilNail)."""
    x_head: float = 0.0
    z_head: float = 0.0
    length: float = 1.0
    inclination: float = 15.0
    bar_diameter: float = 25.0
    drill_hole_diameter: float = 150.0
    fy: float = 420.0
    bond_stress: float = 100.0
    spacing_h: float = 1.5

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Nail":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Anchor:
    """Tieback/anchor (mirrors slope_stability.reinforcement.Anchor)."""
    x_head: float = 0.0
    z_head: float = 0.0
    length: float = 1.0
    T_allow: float = 100.0
    inclination: float = 15.0

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Anchor":
        return cls(**_known_kwargs(cls, data))


@dataclass
class GeosyntheticLayer:
    """Horizontal geosynthetic (mirrors reinforcement.Geosynthetic)."""
    elevation: float = 0.0
    T_allow: float = 10.0
    x_start: Optional[float] = None
    x_end: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "GeosyntheticLayer":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Reinforcement:
    """All reinforcement elements."""
    nails: List[Nail] = field(default_factory=list)
    anchors: List[Anchor] = field(default_factory=list)
    geosynthetics: List[GeosyntheticLayer] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Reinforcement":
        kw = _known_kwargs(cls, data)
        kw["nails"] = [Nail.from_dict(n) for n in (kw.get("nails") or [])]
        kw["anchors"] = [Anchor.from_dict(a)
                         for a in (kw.get("anchors") or [])]
        kw["geosynthetics"] = [GeosyntheticLayer.from_dict(g)
                               for g in (kw.get("geosynthetics") or [])]
        return cls(**kw)

    def __bool__(self) -> bool:
        return bool(self.nails or self.anchors or self.geosynthetics)


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

@dataclass
class LESearch:
    """Critical-surface search settings (slope_stability.search_critical_surface)."""
    surface_type: str = "circular"
    nx: int = 10
    ny: int = 10
    n_trials: int = 500
    n_points: int = 5
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "LESearch":
        return cls(**_known_kwargs(cls, data))


@dataclass
class LEProbabilistic:
    """Probabilistic LE settings (FOSM or Monte Carlo on the critical surface).

    ``variables`` follows slope_stability.probabilistic's spec
    (``{"phi:LayerName": {"mean":..., "cov":..., "dist":...}}``). When empty,
    the builder assembles it from each layer Material's ``probabilistic``
    dict (the cov_lookup-cited values).
    """
    kind: str = "fosm"  # 'fosm' | 'monte_carlo'
    variables: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    n: int = 1000
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "LEProbabilistic":
        return cls(**_known_kwargs(cls, data))


@dataclass
class LEAnalysis:
    """A limit-equilibrium run: method + search + optional probabilistic."""
    type: str = "le"
    name: str = "LE"
    method: str = "bishop"   # fellenius|bishop|janbu|spencer|morgenstern_price|gle
    n_slices: int = 30
    search: LESearch = field(default_factory=LESearch)
    probabilistic: Optional[LEProbabilistic] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "LEAnalysis":
        kw = _known_kwargs(cls, data)
        kw["search"] = LESearch.from_dict(kw.get("search"))
        prob = kw.get("probabilistic")
        kw["probabilistic"] = (LEProbabilistic.from_dict(prob)
                               if prob is not None else None)
        return cls(**kw)


@dataclass
class FEMAnalysis:
    """A fem2d Strength-Reduction-Method run (analyze_slope_srm settings)."""
    type: str = "fem_srm"
    name: str = "FEM-SRM"
    nx: int = 30
    ny: int = 15
    depth: Optional[float] = None
    x_extend: Optional[float] = None
    srf_tol: float = 0.02
    element_type: str = "t6"
    srf_range: Tuple[float, float] = (0.5, 3.0)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "FEMAnalysis":
        kw = _known_kwargs(cls, data)
        rng = kw.get("srf_range")
        if rng is not None:
            kw["srf_range"] = (float(rng[0]), float(rng[1]))
        return cls(**kw)


def _analysis_from_dict(data: dict):
    """Dispatch an analysis dict on its 'type' key."""
    atype = (data or {}).get("type", "le")
    if atype == "fem_srm":
        return FEMAnalysis.from_dict(data)
    return LEAnalysis.from_dict(data)


# ---------------------------------------------------------------------------
# Confirmations / assumptions
# ---------------------------------------------------------------------------

@dataclass
class Confirmations:
    """The staged human gates. ONLY the confirmation tool sets these.

    ``project_run`` refuses while any gate is False; patching a stage's data
    resets that stage's gate (a confirmed model that is then edited is no
    longer confirmed).
    """
    geometry: bool = False
    materials: bool = False
    water_loads: bool = False

    def all_confirmed(self) -> bool:
        return self.geometry and self.materials and self.water_loads

    def missing(self) -> List[str]:
        return [s for s in CONFIRMATION_STAGES if not getattr(self, s)]

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Confirmations":
        return cls(**_known_kwargs(cls, data))


@dataclass
class Assumption:
    """One assumption-ledger entry: a default taken, with its source."""
    field: str = ""
    value: Any = None
    source: str = ""
    note: str = ""

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Assumption":
        return cls(**_known_kwargs(cls, data))


# ---------------------------------------------------------------------------
# Project
# ---------------------------------------------------------------------------

@dataclass
class Project:
    """The canonical model-setup document. See module docstring."""
    meta: ProjectMeta = field(default_factory=ProjectMeta)
    geometry: Geometry = field(default_factory=Geometry)
    stratigraphy: List[Layer] = field(default_factory=list)
    water: Water = field(default_factory=Water)
    loads: Loads = field(default_factory=Loads)
    reinforcement: Reinforcement = field(default_factory=Reinforcement)
    analyses: List[Any] = field(default_factory=list)
    confirmations: Confirmations = field(default_factory=Confirmations)
    assumptions: List[Assumption] = field(default_factory=list)

    # -- elevation bookkeeping (shared by validate / builders / render) ----

    def layer_top(self, i: int) -> Optional[float]:
        """Representative top elevation of stratigraphy[i].

        Explicit ``top_elevation`` wins; else the surface z-max for the first
        layer, else the bottom of the layer above. ``None`` when undetermined.
        """
        layer = self.stratigraphy[i]
        if layer.top_elevation is not None:
            return float(layer.top_elevation)
        if i == 0:
            if not self.geometry.surface_points:
                return None
            return self.geometry.z_surface_range[1]
        return self.layer_bottom(i - 1)

    def layer_bottom(self, i: int) -> Optional[float]:
        """Representative (flat) bottom elevation of stratigraphy[i].

        A named ``bottom_boundary`` polyline yields its MAX z (the same
        convention dxf_import.converter uses, so a boundary's high point is
        the next layer's top). Else the explicit ``bottom_elevation``.
        """
        layer = self.stratigraphy[i]
        if layer.bottom_boundary:
            pts = self.geometry.layer_boundaries.get(layer.bottom_boundary)
            if pts:
                return max(z for _, z in pts)
            return None
        if layer.bottom_elevation is not None:
            return float(layer.bottom_elevation)
        return None

    def boundary_points(self, i: int) -> Optional[List[Tuple[float, float]]]:
        """The bottom polyline of stratigraphy[i] (sorted by x), or None."""
        layer = self.stratigraphy[i]
        if layer.bottom_boundary:
            pts = self.geometry.layer_boundaries.get(layer.bottom_boundary)
            if pts:
                return sorted(pts, key=lambda p: p[0])
        return None

    def section_bottom(self) -> Optional[float]:
        """Lowest layer bottom (the bottom of the modeled section)."""
        bots = [self.layer_bottom(i) for i in range(len(self.stratigraphy))]
        bots = [b for b in bots if b is not None]
        return min(bots) if bots else None

    def add_assumption(self, field_name: str, value: Any, source: str = "",
                       note: str = "") -> None:
        """Append an assumption-ledger entry (deduped on field+value)."""
        for a in self.assumptions:
            if a.field == field_name and a.value == value:
                return
        self.assumptions.append(
            Assumption(field=field_name, value=value, source=source,
                       note=note))

    # -- (de)serialization ---------------------------------------------------

    def to_dict(self) -> dict:
        """Plain-dict form (tuples become lists; JSON-safe)."""
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Load from a plain dict. Unknown keys anywhere are dropped."""
        if not isinstance(data, dict):
            raise ValueError("Project.from_dict expects a dict")
        return cls(
            meta=ProjectMeta.from_dict(data.get("meta")),
            geometry=Geometry.from_dict(data.get("geometry")),
            stratigraphy=[Layer.from_dict(d)
                          for d in (data.get("stratigraphy") or [])],
            water=Water.from_dict(data.get("water")),
            loads=Loads.from_dict(data.get("loads")),
            reinforcement=Reinforcement.from_dict(data.get("reinforcement")),
            analyses=[_analysis_from_dict(d)
                      for d in (data.get("analyses") or [])],
            confirmations=Confirmations.from_dict(data.get("confirmations")),
            assumptions=[Assumption.from_dict(d)
                         for d in (data.get("assumptions") or [])],
        )

    @classmethod
    def from_json(cls, text: str) -> "Project":
        return cls.from_dict(json.loads(text))


__all__ = [
    "SCHEMA_VERSION", "PROVENANCES", "STRENGTH_MODELS",
    "CONFIRMATION_STAGES",
    "ProjectMeta", "Geometry", "Material", "Layer", "Water", "Surcharge",
    "Loads", "Nail", "Anchor", "GeosyntheticLayer", "Reinforcement",
    "LESearch", "LEProbabilistic", "LEAnalysis", "FEMAnalysis",
    "Confirmations", "Assumption", "Project",
]
