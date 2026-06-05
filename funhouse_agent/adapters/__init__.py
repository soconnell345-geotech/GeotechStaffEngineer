"""Adapter modules for funhouse_agent dispatch.

Each adapter bridges flat JSON parameters to structured analysis module APIs.
Modules are lazy-loaded: imports only happen when call_agent() is invoked.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Shared JSON-serialization helpers
# ---------------------------------------------------------------------------

def clean_value(v):
    """Convert numpy types and NaN to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def clean_result(result: dict) -> dict:
    """Recursively clean a result dict for JSON serialization."""
    cleaned = {}
    for k, v in result.items():
        if isinstance(v, list):
            cleaned[k] = [
                clean_result(item) if isinstance(item, dict)
                else clean_value(item)
                for item in v
            ]
        elif isinstance(v, dict):
            cleaned[k] = clean_result(v)
        else:
            cleaned[k] = clean_value(v)
    return cleaned


# ---------------------------------------------------------------------------
# Module registry — defines all available modules and their adapter paths
# ---------------------------------------------------------------------------

MODULE_REGISTRY = {
    "bearing_capacity": {
        "adapter": "funhouse_agent.adapters.bearing_capacity",
        "brief": "Shallow foundation bearing capacity (Vesic/Meyerhof/Hansen)",
    },
    "settlement": {
        "adapter": "funhouse_agent.adapters.settlement",
        "brief": "Foundation settlement (elastic, Schmertmann, consolidation)",
    },
    "slope_stability": {
        "adapter": "funhouse_agent.adapters.slope_stability",
        "brief": "Slope stability (Fellenius/Bishop/Spencer, circular+noncircular, grid search)",
    },
    "seismic_geotech": {
        "adapter": "funhouse_agent.adapters.seismic_geotech",
        "brief": "Seismic evaluation (site class, M-O pressure, liquefaction)",
    },
    "retaining_walls": {
        "adapter": "funhouse_agent.adapters.retaining_walls",
        "brief": "Retaining walls (cantilever + MSE, GEC-11)",
    },
    "axial_pile": {
        "adapter": "funhouse_agent.adapters.axial_pile",
        "brief": "Driven pile axial capacity (Nordlund/Tomlinson/Beta)",
    },
    "drilled_shaft": {
        "adapter": "funhouse_agent.adapters.drilled_shaft",
        "brief": "Drilled shaft capacity (GEC-10 alpha/beta/rock socket, LRFD)",
    },
    "sheet_pile": {
        "adapter": "funhouse_agent.adapters.sheet_pile",
        "brief": "Sheet pile walls (cantilever and anchored)",
    },
    "lateral_pile": {
        "adapter": "funhouse_agent.adapters.lateral_pile",
        "brief": "Lateral pile analysis (COM624P, 8 p-y models, FD solver)",
    },
    "pile_group": {
        "adapter": "funhouse_agent.adapters.pile_group",
        "brief": "Pile group analysis (rigid cap, 6-DOF, Converse-Labarre)",
    },
    "ground_improvement": {
        "adapter": "funhouse_agent.adapters.ground_improvement",
        "brief": "Ground improvement (aggregate piers, wick drains, vibro, GEC-13)",
    },
    "wave_equation": {
        "adapter": "funhouse_agent.adapters.wave_equation",
        "brief": "Smith 1-D wave equation (bearing graph, drivability)",
    },
    "downdrag": {
        "adapter": "funhouse_agent.adapters.downdrag",
        "brief": "Pile downdrag (Fellenius neutral plane, UFC 3-220-20)",
    },
    "wind_loads": {
        "adapter": "funhouse_agent.adapters.wind_loads",
        "brief": "ASCE 7-22 wind loads on freestanding walls and fences (Ch 29.3)",
    },
    "soe": {
        "adapter": "funhouse_agent.adapters.soe",
        "brief": "Support of excavation (braced/cantilever walls, stability, anchors)",
    },
    "geolysis": {
        "adapter": "funhouse_agent.adapters.geolysis",
        "brief": "Soil classification (USCS/AASHTO) + SPT corrections + bearing capacity",
    },
    "dxf_export": {
        "adapter": "funhouse_agent.adapters.dxf_export",
        "brief": "Export cross-section geometry to DXF file format",
    },
    "calc_package": {
        "adapter": "funhouse_agent.adapters.calc_package",
        "brief": "Generate Mathcad-style calc packages (HTML/LaTeX/PDF) for 13 analysis modules",
    },
    "pyseismosoil": {
        "adapter": "funhouse_agent.adapters.pyseismosoil_adapter",
        "brief": "Nonlinear soil curve generation (MKZ/HH) and Vs profile site characterization",
    },
    "pystra": {
        "adapter": "funhouse_agent.adapters.pystra_adapter",
        "brief": "Structural reliability analysis (FORM/SORM/Monte Carlo)",
    },
    "salib": {
        "adapter": "funhouse_agent.adapters.salib_adapter",
        "brief": "Sensitivity analysis (Sobol variance-based and Morris screening)",
    },
    "pygef": {
        "adapter": "funhouse_agent.adapters.pygef_adapter",
        "brief": "CPT and borehole file parser (GEF/BRO-XML via pygef)",
    },
    "dxf_import": {
        "adapter": "funhouse_agent.adapters.dxf_import_adapter",
        "brief": "DXF CAD import for slope stability + FEM (discover, parse, build geometry)",
    },
    "pdf_import": {
        "adapter": "funhouse_agent.adapters.pdf_import_adapter",
        "brief": "PDF cross-section import (PyMuPDF vector extraction, content discovery)",
    },
    "ags4": {
        "adapter": "funhouse_agent.adapters.ags4_adapter",
        "brief": "AGS4 geotechnical data format reader and validator",
    },
    "pydiggs": {
        "adapter": "funhouse_agent.adapters.pydiggs_adapter",
        "brief": "DIGGS 2.6 XML schema and dictionary validation",
    },
    "opensees": {
        "adapter": "funhouse_agent.adapters.opensees_adapter",
        "brief": "OpenSees FE analyses (PM4Sand DSS, BNWF lateral pile, 1D site response)",
    },
    "pystrata": {
        "adapter": "funhouse_agent.adapters.pystrata_adapter",
        "brief": "1D site response (equivalent-linear and linear elastic, SHAKE-type)",
    },
    "liquepy": {
        "adapter": "funhouse_agent.adapters.liquepy_adapter",
        "brief": "CPT-based liquefaction triggering and field correlations (Boulanger & Idriss 2014)",
    },
    "seismic_signals": {
        "adapter": "funhouse_agent.adapters.seismic_signals_adapter",
        "brief": "Earthquake signal processing (response spectra, intensity measures, RotD, filtering)",
    },
    "fem2d": {
        "adapter": "funhouse_agent.adapters.fem2d_adapter",
        "brief": "2D plane-strain FEM (gravity, foundation, slope SRM, excavation, seepage, consolidation, staged)",
    },
    "fdm2d": {
        "adapter": "funhouse_agent.adapters.fdm2d_adapter",
        "brief": "2D explicit Lagrangian FDM, FLAC-style (gravity, foundation)",
    },
    "gstools": {
        "adapter": "funhouse_agent.adapters.gstools_adapter",
        "brief": "Geostatistical kriging, variogram fitting, and random field generation",
    },
    "hvsrpy": {
        "adapter": "funhouse_agent.adapters.hvsrpy_adapter",
        "brief": "HVSR site characterization from ambient noise (resonant frequency, amplification)",
    },
    "swprocess": {
        "adapter": "funhouse_agent.adapters.swprocess_adapter",
        "brief": "MASW surface wave dispersion analysis",
    },
    "subsurface": {
        "adapter": "funhouse_agent.adapters.subsurface_adapter",
        "brief": "Subsurface data visualization — DIGGS XML ingestion, parameter vs depth, Atterberg limits, plan view, cross-section, trends",
    },
    # --- geotech-references agents (reference modules + cross-reference DB) ---
    "reference_db": {
        "adapter": "funhouse_agent.adapters.reference_db_adapter",
        "brief": "Cross-reference FTS5 search over all structured chapter text (DM7 + GEC + micropile). Use first to scope, then drill in via reference_get.",
    },
    "figure_db": {
        "adapter": "funhouse_agent.adapters.figure_db_adapter",
        "brief": "FTS5 search over the digitized figure catalogs (DM7 charts/figures). Find a chart by meaning, then read a value off it with the read_reference_figure vision tool.",
    },
    "dm7": {
        "adapter": "funhouse_agent.adapters.dm7_adapter",
        "brief": "NAVFAC DM7 equations (340+): soil classification, stresses, settlement, seepage, foundations",
    },
    "gec4": {
        "adapter": "funhouse_agent.adapters.gec4_adapter",
        "brief": "GEC-4 ground anchors reference (FHWA-IF-99-015, 1999): bond stresses, anchor load transfer, corrosion protection, anchored wall design (ASD)",
    },
    "gec5": {
        "adapter": "funhouse_agent.adapters.gec5_adapter",
        "brief": "GEC-5 geotechnical site characterization (FHWA NHI-16-072, 2017): investigation planning, soil/rock classification, strength, consolidation, stiffness, hazards",
    },
    "gec6": {
        "adapter": "funhouse_agent.adapters.gec6_adapter",
        "brief": "GEC-6 shallow foundations reference (FHWA-SA-02-054, tables/figures/text)",
    },
    "gec7": {
        "adapter": "funhouse_agent.adapters.gec7_adapter",
        "brief": "GEC-7 soil nail walls reference (FHWA-NHI-14-007, tables/figures/text)",
    },
    "gec8": {
        "adapter": "funhouse_agent.adapters.gec8_adapter",
        "brief": "GEC-8 CFA pile design (FHWA-HIF-07-03, 2007): DD pile capacity (NeSmith), ASD design, grout volume factor, p-multipliers, group efficiency",
    },
    "gec9": {
        "adapter": "funhouse_agent.adapters.gec9_adapter",
        "brief": "GEC-9 laterally loaded piles (FHWA-HIF-18-031, 2018): lateral resistance factors, p-multipliers, p-y parameters for stiff clay and sand",
    },
    "gec10": {
        "adapter": "funhouse_agent.adapters.gec10_adapter",
        "brief": "GEC-10 drilled shafts reference (FHWA-NHI-18-024, tables/figures/text)",
    },
    "gec11": {
        "adapter": "funhouse_agent.adapters.gec11_adapter",
        "brief": "GEC-11 MSE walls & reinforced soil slopes (FHWA-NHI-10-024, tables/figures/text)",
    },
    "gec12": {
        "adapter": "funhouse_agent.adapters.gec12_adapter",
        "brief": "GEC-12 driven piles reference (FHWA-NHI-16-009, tables/figures/text)",
    },
    "gec13": {
        "adapter": "funhouse_agent.adapters.gec13_adapter",
        "brief": "GEC-13 ground modification reference (FHWA-NHI-16-027, tables/figures/text)",
    },
    "gec14": {
        "adapter": "funhouse_agent.adapters.gec14_adapter",
        "brief": "GEC-14 QA for geotechnical reporting documents (FHWA-HIF-17-016, 2016): GRD types, ACMs/ADMs, QA process, checklists",
    },
    "micropile": {
        "adapter": "funhouse_agent.adapters.micropile_adapter",
        "brief": "Micropile design reference (FHWA-NHI-05-039, bond stress/tables/text)",
    },
    "ufc_backfill": {
        "adapter": "funhouse_agent.adapters.ufc_backfill_adapter",
        "brief": "UFC 3-220-04N backfill design (compaction, filter criteria, drainage)",
    },
    "ufc_expansive": {
        "adapter": "funhouse_agent.adapters.ufc_expansive_adapter",
        "brief": "UFC 3-220-07 expansive soils (swell potential, heave, pier design)",
    },
    "ufc_pavement": {
        "adapter": "funhouse_agent.adapters.ufc_pavement_adapter",
        "brief": "UFC 3-250-01 roads/parking pavement design (subgrade categories, frost classification, equivalency factors, minimum thickness, k-subgrade)",
    },
    "fema_p2082": {
        "adapter": "funhouse_agent.adapters.fema_p2082_adapter",
        "brief": "FEMA P-2082 (2020 NEHRP Provisions) seismic site design: REVISED site classification (Vs30 -> Site Class A/B/BC/C/CD/D/DE/E/F, with new intermediate classes BC/CD/DE), SDS/SD1, two-period design spectrum, Seismic Design Category, Risk Category",
    },
    "california_trenching": {
        "adapter": "funhouse_agent.adapters.california_trenching_adapter",
        "brief": "California (Caltrans) Trenching and Shoring Manual: temporary excavation support. Cal/OSHA Type A/B/C soil classification + maximum allowable slopes (A 3/4:1, B 1:1, C 1-1/2:1), Rankine/Coulomb/Bell earth pressure coefficients, apparent active coefficient (>=0.25), log-spiral passive Kp (Caquot-Kerisel), apparent earth pressure (AEP) envelopes for braced/anchored walls, soldier-pile arching, bottom-heave FS. US customary units (psf, pcf, tsf, ft)",
    },
}
