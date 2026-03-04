"""
DIGGS 2.6 XML parser → SiteModel.

Uses xml.etree.ElementTree (stdlib). Auto-detects 2.6 vs 2.5.a namespace.
Extracts borings, lithology, and 18 test types: SPT, Atterberg limits,
moisture content, GWL, CPT, vane shear, consolidation, triaxial, direct
shear, UCS, point load, specific gravity, gradation, permeability,
compaction, dynamic probe, pressuremeter, and DMT.
Resolves xlink:href references to associate tests with borings.

Functions
---------
parse_diggs : Parse DIGGS XML file or string → SiteModel
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Optional, Dict, List

from subsurface_characterization.site_model import (
    SiteModel, Investigation, PointMeasurement, LithologyInterval,
)
from subsurface_characterization.results import DiggsParseResult


# ---------------------------------------------------------------------------
# Namespace handling
# ---------------------------------------------------------------------------

_NS_26 = "http://diggsml.org/schemas/2.6"
_NS_25A = "http://diggsml.org/schemas/2.5.a"
_NS_GML = "http://www.opengis.net/gml/3.2"
_NS_XLINK = "http://www.w3.org/1999/xlink"

# Common DIGGS 2.6 namespace map (elements not prefixed use diggs ns)
_NS_MAP_26 = {
    "diggs": _NS_26,
    "gml": _NS_GML,
    "xlink": _NS_XLINK,
}

_NS_MAP_25A = {
    "diggs": _NS_25A,
    "gml": _NS_GML,
    "xlink": _NS_XLINK,
}


def parse_diggs(
    filepath: Optional[str] = None,
    content: Optional[str] = None,
) -> DiggsParseResult:
    """Parse DIGGS XML file or string into a SiteModel.

    Parameters
    ----------
    filepath : str, optional
        Path to DIGGS XML file.
    content : str, optional
        DIGGS XML string content.

    Returns
    -------
    DiggsParseResult
        Parsed site model with metadata.

    Raises
    ------
    ValueError
        If neither filepath nor content provided, or XML is invalid.
    """
    if filepath is None and content is None:
        raise ValueError("Must provide either filepath or content")

    warnings = []

    try:
        if content is not None:
            root = ET.fromstring(content)
        else:
            tree = ET.parse(filepath)
            root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML: {e}")

    # Auto-detect namespace
    ns_map = _detect_namespace(root)
    diggs_ns = ns_map["diggs"]

    # Extract project name
    project_name = _extract_project_name(root, ns_map)

    # Build gml:id → Investigation mapping
    # Borings are the primary investigation locations
    investigations: Dict[str, Investigation] = {}
    gml_id_map: Dict[str, str] = {}  # gml:id → investigation_id

    for borehole in _findall(root, ".//diggs:Borehole", ns_map):
        inv = _parse_borehole(borehole, ns_map, warnings)
        if inv.investigation_id:
            investigations[inv.investigation_id] = inv
            # Map gml:id to investigation_id for xlink resolution
            gml_id = borehole.get(f"{{{_NS_GML}}}id", "")
            if gml_id:
                gml_id_map[gml_id] = inv.investigation_id

    # Parse tests and associate with borings via xlink:href
    _parse_spt_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_atterberg_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_moisture_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_water_levels(root, ns_map, investigations, gml_id_map, warnings)
    # Tier 1
    _parse_cpt_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_vane_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_consolidation_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_triaxial_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_direct_shear_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_ucs_tests(root, ns_map, investigations, gml_id_map, warnings)
    # Tier 2
    _parse_point_load_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_specific_gravity_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_gradation_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_permeability_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_compaction_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_dynamic_probe_tests(root, ns_map, investigations, gml_id_map, warnings)
    # Tier 3
    _parse_pressuremeter_tests(root, ns_map, investigations, gml_id_map, warnings)
    _parse_dmt_tests(root, ns_map, investigations, gml_id_map, warnings)

    # Count totals
    n_measurements = sum(len(inv.measurements) for inv in investigations.values())
    n_lithology = sum(len(inv.lithology) for inv in investigations.values())

    site = SiteModel(
        project_name=project_name,
        investigations=list(investigations.values()),
    )

    return DiggsParseResult(
        site=site,
        n_investigations=len(investigations),
        n_measurements=n_measurements,
        n_lithology_intervals=n_lithology,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_namespace(root: ET.Element) -> dict:
    """Detect DIGGS namespace version from root element."""
    tag = root.tag
    if _NS_26 in tag:
        return _NS_MAP_26
    elif _NS_25A in tag:
        return _NS_MAP_25A
    # Try attributes
    for ns_uri in root.attrib.values():
        if _NS_26 in str(ns_uri):
            return _NS_MAP_26
        if _NS_25A in str(ns_uri):
            return _NS_MAP_25A
    # Default to 2.6
    return _NS_MAP_26


def _findall(element: ET.Element, path: str, ns_map: dict) -> list:
    """Find all elements with namespace substitution."""
    # Replace namespace prefixes with URIs
    for prefix, uri in ns_map.items():
        path_with_ns = path.replace(f"{prefix}:", f"{{{uri}}}")
        # Only replace if prefix is in path
        if f"{prefix}:" in path:
            path = path_with_ns
    try:
        return element.findall(path)
    except Exception:
        return []


def _find(element: ET.Element, path: str, ns_map: dict):
    """Find single element with namespace substitution."""
    for prefix, uri in ns_map.items():
        if f"{prefix}:" in path:
            path = path.replace(f"{prefix}:", f"{{{uri}}}")
    try:
        return element.find(path)
    except Exception:
        return None


def _text(element: ET.Element, path: str, ns_map: dict, default: str = "") -> str:
    """Get text content of a child element."""
    child = _find(element, path, ns_map)
    if child is not None and child.text:
        return child.text.strip()
    return default


def _float_text(element: ET.Element, path: str, ns_map: dict, default: float = 0.0) -> float:
    """Get float value from child element text."""
    text = _text(element, path, ns_map, "")
    if text:
        try:
            return float(text)
        except ValueError:
            pass
    return default


def _extract_project_name(root: ET.Element, ns_map: dict) -> str:
    """Extract project name from Project element."""
    # Try DIGGS 2.6 path
    name = _text(root, ".//diggs:Project/gml:name", ns_map)
    if not name:
        name = _text(root, ".//diggs:Project/diggs:name", ns_map)
    if not name:
        # Fallback to root-level name
        name = _text(root, "gml:name", ns_map)
    return name or "Unknown Project"


def _parse_borehole(element: ET.Element, ns_map: dict, warnings: list) -> Investigation:
    """Parse a Borehole element into an Investigation."""
    gml_id = element.get(f"{{{_NS_GML}}}id", "")
    name = _text(element, "gml:name", ns_map) or gml_id

    # Coordinates from referencePoint/gml:Point/gml:pos
    x, y, elev = 0.0, 0.0, 0.0
    pos_text = _text(element, ".//diggs:referencePoint/gml:Point/gml:pos", ns_map)
    if not pos_text:
        pos_text = _text(element, ".//gml:pos", ns_map)
    if pos_text:
        parts = pos_text.split()
        try:
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
            if len(parts) >= 3:
                elev = float(parts[2])
        except ValueError:
            warnings.append(f"Could not parse coordinates for {name}: {pos_text}")

    total_depth = _float_text(element, "diggs:totalMeasuredDepth", ns_map)
    if total_depth == 0.0:
        total_depth = _float_text(element, ".//diggs:totalDepth", ns_map)

    # Lithology observations
    lithology = []
    for lith_elem in _findall(element, ".//diggs:LithologyObservation", ns_map):
        lith = _parse_lithology(lith_elem, ns_map)
        if lith:
            lithology.append(lith)

    return Investigation(
        investigation_id=name,
        investigation_type="boring",
        x=x,
        y=y,
        elevation_m=elev,
        total_depth_m=total_depth,
        lithology=lithology,
    )


def _parse_lithology(element: ET.Element, ns_map: dict) -> Optional[LithologyInterval]:
    """Parse a LithologyObservation element."""
    top = _float_text(element, "diggs:depthTop", ns_map)
    bottom = _float_text(element, "diggs:depthBase", ns_map)
    if bottom == 0.0:
        bottom = _float_text(element, "diggs:depthBottom", ns_map)
    if bottom <= top:
        return None

    desc = _text(element, "diggs:description", ns_map)
    if not desc:
        desc = _text(element, "diggs:lithologyDescription", ns_map)
    uscs = _text(element, "diggs:uscs", ns_map)
    if not uscs:
        uscs = _text(element, "diggs:classificationCode", ns_map)
    color = _text(element, "diggs:color", ns_map)

    return LithologyInterval(
        top_depth_m=top,
        bottom_depth_m=bottom,
        description=desc,
        uscs=uscs,
        color=color,
    )


def _resolve_xlink(element: ET.Element, ns_map: dict, gml_id_map: dict) -> Optional[str]:
    """Resolve xlink:href to investigation_id."""
    # Look for investigationRef or samplingFeature with xlink:href
    for child_tag in ["diggs:investigationRef", "diggs:samplingFeature",
                       "diggs:borehole", "diggs:investigation"]:
        child = _find(element, child_tag, ns_map)
        if child is not None:
            href = child.get(f"{{{_NS_XLINK}}}href", "")
            if not href:
                href = child.get("xlink:href", "")
            if href:
                gml_id = href.lstrip("#")
                if gml_id in gml_id_map:
                    return gml_id_map[gml_id]
    return None


def _parse_spt_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse DrivenPenetrationTest (SPT) elements."""
    for elem in _findall(root, ".//diggs:DrivenPenetrationTest", ns_map):
        inv_id = _resolve_xlink(elem, ns_map, gml_id_map)
        if not inv_id or inv_id not in investigations:
            # Try to find by first investigation if only one exists
            if len(investigations) == 1:
                inv_id = list(investigations.keys())[0]
            else:
                continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)
        n_value = _float_text(elem, "diggs:blowCount", ns_map)
        if n_value == 0.0:
            n_value = _float_text(elem, "diggs:nValue", ns_map)

        if depth > 0 or n_value > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth,
                parameter="N_spt",
                value=n_value,
                source="field",
                test_type="SPT",
            ))


def _parse_atterberg_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse AtterbergLimitsTest elements."""
    for elem in _findall(root, ".//diggs:AtterbergLimitsTest", ns_map):
        inv_id = _resolve_xlink(elem, ns_map, gml_id_map)
        if not inv_id or inv_id not in investigations:
            if len(investigations) == 1:
                inv_id = list(investigations.keys())[0]
            else:
                continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        ll = _float_text(elem, "diggs:liquidLimit", ns_map)
        pl = _float_text(elem, "diggs:plasticLimit", ns_map)

        if ll > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth,
                parameter="LL_pct",
                value=ll,
                source="lab",
                test_type="Atterberg",
            ))
        if pl > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth,
                parameter="PL_pct",
                value=pl,
                source="lab",
                test_type="Atterberg",
            ))


def _parse_moisture_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse MoistureContent elements."""
    for elem in _findall(root, ".//diggs:MoistureContent", ns_map):
        inv_id = _resolve_xlink(elem, ns_map, gml_id_map)
        if not inv_id or inv_id not in investigations:
            if len(investigations) == 1:
                inv_id = list(investigations.keys())[0]
            else:
                continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        wn = _float_text(elem, "diggs:moistureContent", ns_map)
        if wn == 0.0:
            wn = _float_text(elem, "diggs:value", ns_map)

        if wn > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth,
                parameter="wn_pct",
                value=wn,
                source="lab",
                test_type="moisture_content",
            ))


def _parse_water_levels(root, ns_map, investigations, gml_id_map, warnings):
    """Parse water level observations and set gwl_depth_m."""
    for elem in _findall(root, ".//diggs:WaterLevelObservation", ns_map):
        inv_id = _resolve_xlink(elem, ns_map, gml_id_map)
        if not inv_id or inv_id not in investigations:
            if len(investigations) == 1:
                inv_id = list(investigations.keys())[0]
            else:
                continue

        depth = _float_text(elem, "diggs:waterDepth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:depth", ns_map)

        if depth > 0:
            investigations[inv_id].gwl_depth_m = depth


# ---------------------------------------------------------------------------
# Tier 1 parsers
# ---------------------------------------------------------------------------

def _resolve_inv(elem, ns_map, investigations, gml_id_map):
    """Resolve investigation ID from element, return None if not found."""
    inv_id = _resolve_xlink(elem, ns_map, gml_id_map)
    if not inv_id or inv_id not in investigations:
        if len(investigations) == 1:
            return list(investigations.keys())[0]
        return None
    return inv_id


def _parse_cpt_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse StaticConePenetrationTest (CPT/CPTu) elements."""
    for elem in _findall(root, ".//diggs:StaticConePenetrationTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        # Try nested readings first (multiple depths within one element)
        readings = _findall(elem, ".//diggs:ConePenetrationReading", ns_map)
        if readings:
            for rdg in readings:
                depth = _float_text(rdg, "diggs:depth", ns_map)
                _add_cpt_reading(rdg, ns_map, investigations[inv_id], depth)
        else:
            # Single-depth element
            depth = _float_text(elem, "diggs:depth", ns_map)
            if depth == 0.0:
                depth = _float_text(elem, "diggs:testDepth", ns_map)
            _add_cpt_reading(elem, ns_map, investigations[inv_id], depth)


def _add_cpt_reading(elem, ns_map, inv, depth):
    """Add CPT measurements from a reading element."""
    qc = _float_text(elem, "diggs:coneResistance", ns_map)
    if qc == 0.0:
        qc = _float_text(elem, "diggs:qc", ns_map)
    fs = _float_text(elem, "diggs:sleeveFriction", ns_map)
    if fs == 0.0:
        fs = _float_text(elem, "diggs:fs", ns_map)
    u2 = _float_text(elem, "diggs:porePressure", ns_map)
    if u2 == 0.0:
        u2 = _float_text(elem, "diggs:u2", ns_map)
    rf = _float_text(elem, "diggs:frictionRatio", ns_map)
    if rf == 0.0:
        rf = _float_text(elem, "diggs:Rf", ns_map)

    if qc > 0:
        inv.measurements.append(PointMeasurement(
            depth_m=depth, parameter="qc_kPa", value=qc,
            source="field", test_type="CPTu",
        ))
    if fs > 0:
        inv.measurements.append(PointMeasurement(
            depth_m=depth, parameter="fs_kPa", value=fs,
            source="field", test_type="CPTu",
        ))
    if u2 != 0.0:
        inv.measurements.append(PointMeasurement(
            depth_m=depth, parameter="u2_kPa", value=u2,
            source="field", test_type="CPTu",
        ))
    if rf > 0:
        inv.measurements.append(PointMeasurement(
            depth_m=depth, parameter="Rf_pct", value=rf,
            source="field", test_type="CPTu",
        ))


def _parse_vane_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse InsituVaneTest (field vane shear) elements."""
    for elem in _findall(root, ".//diggs:InsituVaneTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        su_peak = _float_text(elem, "diggs:peakShearStrength", ns_map)
        if su_peak == 0.0:
            su_peak = _float_text(elem, "diggs:undrainedShearStrength", ns_map)
        su_rem = _float_text(elem, "diggs:remoldedShearStrength", ns_map)
        sensitivity = _float_text(elem, "diggs:sensitivity", ns_map)

        inv = investigations[inv_id]
        if su_peak > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="Su_vane_kPa", value=su_peak,
                source="field", test_type="vane_shear",
            ))
        if su_rem > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="Su_vane_remolded_kPa", value=su_rem,
                source="field", test_type="vane_shear",
            ))
        if sensitivity > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="St", value=sensitivity,
                source="field", test_type="vane_shear",
            ))
        elif su_peak > 0 and su_rem > 0:
            # Compute sensitivity if not given directly
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="St", value=su_peak / su_rem,
                source="field", test_type="vane_shear",
            ))


def _parse_consolidation_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse ConsolidationTest elements (Cc, Cr, sigma_p, e0)."""
    for elem in _findall(root, ".//diggs:ConsolidationTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        e0 = _float_text(elem, "diggs:initialVoidRatio", ns_map)
        if e0 == 0.0:
            e0 = _float_text(elem, "diggs:e0", ns_map)
        cc = _float_text(elem, "diggs:compressionIndex", ns_map)
        if cc == 0.0:
            cc = _float_text(elem, "diggs:Cc", ns_map)
        cr = _float_text(elem, "diggs:recompressionIndex", ns_map)
        if cr == 0.0:
            cr = _float_text(elem, "diggs:Cr", ns_map)
        sigma_p = _float_text(elem, "diggs:preconsolidationPressure", ns_map)
        if sigma_p == 0.0:
            sigma_p = _float_text(elem, "diggs:sigmaP", ns_map)

        inv = investigations[inv_id]
        if e0 > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="e0", value=e0,
                source="lab", test_type="consolidation",
            ))
        if cc > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="Cc", value=cc,
                source="lab", test_type="consolidation",
            ))
        if cr > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="Cr", value=cr,
                source="lab", test_type="consolidation",
            ))
        if sigma_p > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="sigma_p_kPa", value=sigma_p,
                source="lab", test_type="consolidation",
            ))


def _parse_triaxial_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse TriaxialTest elements (cu or phi from peak strengths)."""
    for elem in _findall(root, ".//diggs:TriaxialTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        cu = _float_text(elem, "diggs:undrainedShearStrength", ns_map)
        if cu == 0.0:
            cu = _float_text(elem, "diggs:cu", ns_map)
        phi = _float_text(elem, "diggs:frictionAngle", ns_map)
        if phi == 0.0:
            phi = _float_text(elem, "diggs:phi", ns_map)
        cohesion = _float_text(elem, "diggs:cohesion", ns_map)
        if cohesion == 0.0:
            cohesion = _float_text(elem, "diggs:c", ns_map)

        inv = investigations[inv_id]
        if cu > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="cu_kPa", value=cu,
                source="lab", test_type="triaxial",
            ))
        if phi > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="phi_deg", value=phi,
                source="lab", test_type="triaxial",
            ))
        if cohesion > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="c_kPa", value=cohesion,
                source="lab", test_type="triaxial",
            ))


def _parse_direct_shear_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse DirectShearTest elements (phi, c peak/residual)."""
    for elem in _findall(root, ".//diggs:DirectShearTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        phi = _float_text(elem, "diggs:frictionAngle", ns_map)
        if phi == 0.0:
            phi = _float_text(elem, "diggs:peakFrictionAngle", ns_map)
        if phi == 0.0:
            phi = _float_text(elem, "diggs:phi", ns_map)
        cohesion = _float_text(elem, "diggs:cohesion", ns_map)
        if cohesion == 0.0:
            cohesion = _float_text(elem, "diggs:peakCohesion", ns_map)
        if cohesion == 0.0:
            cohesion = _float_text(elem, "diggs:c", ns_map)

        inv = investigations[inv_id]
        if phi > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="phi_deg", value=phi,
                source="lab", test_type="direct_shear",
            ))
        if cohesion > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="c_kPa", value=cohesion,
                source="lab", test_type="direct_shear",
            ))


def _parse_ucs_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse UnconfinedCompressiveStrengthTest elements."""
    for elem in _findall(root, ".//diggs:UnconfinedCompressiveStrengthTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        qu = _float_text(elem, "diggs:compressiveStrength", ns_map)
        if qu == 0.0:
            qu = _float_text(elem, "diggs:qu", ns_map)

        if qu > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth, parameter="qu_kPa", value=qu,
                source="lab", test_type="UCS",
            ))


# ---------------------------------------------------------------------------
# Tier 2 parsers
# ---------------------------------------------------------------------------

def _parse_point_load_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse PointLoadTest elements (Is50)."""
    for elem in _findall(root, ".//diggs:PointLoadTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        is50 = _float_text(elem, "diggs:pointLoadIndex", ns_map)
        if is50 == 0.0:
            is50 = _float_text(elem, "diggs:Is50", ns_map)

        if is50 > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth, parameter="Is50_MPa", value=is50,
                source="lab", test_type="point_load",
            ))


def _parse_specific_gravity_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse SpecificGravityTest elements."""
    for elem in _findall(root, ".//diggs:SpecificGravityTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        gs = _float_text(elem, "diggs:specificGravity", ns_map)
        if gs == 0.0:
            gs = _float_text(elem, "diggs:Gs", ns_map)

        if gs > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth, parameter="Gs", value=gs,
                source="lab", test_type="specific_gravity",
            ))


def _parse_gradation_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse ParticleSizeAnalysis / MaterialGradationTest elements."""
    tag_names = [
        ".//diggs:ParticleSizeAnalysis",
        ".//diggs:MaterialGradationTest",
    ]
    for tag in tag_names:
        for elem in _findall(root, tag, ns_map):
            inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
            if not inv_id:
                continue

            depth = _float_text(elem, "diggs:depth", ns_map)
            if depth == 0.0:
                depth = _float_text(elem, "diggs:testDepth", ns_map)

            inv = investigations[inv_id]
            params = {
                "pct_gravel": ["diggs:percentGravel", "diggs:gravel"],
                "pct_sand": ["diggs:percentSand", "diggs:sand"],
                "pct_fines": ["diggs:percentFines", "diggs:fines"],
                "D10_mm": ["diggs:D10", "diggs:d10"],
                "D30_mm": ["diggs:D30", "diggs:d30"],
                "D60_mm": ["diggs:D60", "diggs:d60"],
            }
            for param_name, xml_tags in params.items():
                val = 0.0
                for xt in xml_tags:
                    val = _float_text(elem, xt, ns_map)
                    if val != 0.0:
                        break
                if val > 0:
                    inv.measurements.append(PointMeasurement(
                        depth_m=depth, parameter=param_name, value=val,
                        source="lab", test_type="gradation",
                    ))


def _parse_permeability_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse LabPermeabilityTest elements."""
    for elem in _findall(root, ".//diggs:LabPermeabilityTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        k = _float_text(elem, "diggs:permeability", ns_map)
        if k == 0.0:
            k = _float_text(elem, "diggs:hydraulicConductivity", ns_map)
        if k == 0.0:
            k = _float_text(elem, "diggs:k", ns_map)

        if k > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth, parameter="k_m_per_s", value=k,
                source="lab", test_type="permeability",
            ))


def _parse_compaction_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse LabCompactionTest (Proctor) elements."""
    for elem in _findall(root, ".//diggs:LabCompactionTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        gamma_max = _float_text(elem, "diggs:maxDryDensity", ns_map)
        if gamma_max == 0.0:
            gamma_max = _float_text(elem, "diggs:maximumDryUnitWeight", ns_map)
        wn_opt = _float_text(elem, "diggs:optimumMoistureContent", ns_map)
        if wn_opt == 0.0:
            wn_opt = _float_text(elem, "diggs:optimumWaterContent", ns_map)

        inv = investigations[inv_id]
        if gamma_max > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="gamma_max_kNm3", value=gamma_max,
                source="lab", test_type="compaction",
            ))
        if wn_opt > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="wn_opt_pct", value=wn_opt,
                source="lab", test_type="compaction",
            ))


def _parse_dynamic_probe_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse DynamicProbeTest elements."""
    for elem in _findall(root, ".//diggs:DynamicProbeTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        n_dp = _float_text(elem, "diggs:blowCount", ns_map)
        if n_dp == 0.0:
            n_dp = _float_text(elem, "diggs:nValue", ns_map)

        if depth > 0 or n_dp > 0:
            investigations[inv_id].measurements.append(PointMeasurement(
                depth_m=depth, parameter="N_dp", value=n_dp,
                source="field", test_type="dynamic_probe",
            ))


# ---------------------------------------------------------------------------
# Tier 3 parsers
# ---------------------------------------------------------------------------

def _parse_pressuremeter_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse PressuremeterTest elements (E_pmt, p_limit)."""
    for elem in _findall(root, ".//diggs:PressuremeterTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        e_pmt = _float_text(elem, "diggs:pressureMeterModulus", ns_map)
        if e_pmt == 0.0:
            e_pmt = _float_text(elem, "diggs:modulus", ns_map)
        if e_pmt == 0.0:
            e_pmt = _float_text(elem, "diggs:Epmt", ns_map)
        p_limit = _float_text(elem, "diggs:limitPressure", ns_map)
        if p_limit == 0.0:
            p_limit = _float_text(elem, "diggs:pLimit", ns_map)

        inv = investigations[inv_id]
        if e_pmt > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="E_pmt_kPa", value=e_pmt,
                source="field", test_type="pressuremeter",
            ))
        if p_limit > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="p_limit_kPa", value=p_limit,
                source="field", test_type="pressuremeter",
            ))


def _parse_dmt_tests(root, ns_map, investigations, gml_id_map, warnings):
    """Parse FlatPlateDilatometerTest (DMT) elements."""
    for elem in _findall(root, ".//diggs:FlatPlateDilatometerTest", ns_map):
        inv_id = _resolve_inv(elem, ns_map, investigations, gml_id_map)
        if not inv_id:
            continue

        depth = _float_text(elem, "diggs:depth", ns_map)
        if depth == 0.0:
            depth = _float_text(elem, "diggs:testDepth", ns_map)

        ed = _float_text(elem, "diggs:dilatometerModulus", ns_map)
        if ed == 0.0:
            ed = _float_text(elem, "diggs:ED", ns_map)
        id_dmt = _float_text(elem, "diggs:materialIndex", ns_map)
        if id_dmt == 0.0:
            id_dmt = _float_text(elem, "diggs:ID", ns_map)
        kd = _float_text(elem, "diggs:horizontalStressIndex", ns_map)
        if kd == 0.0:
            kd = _float_text(elem, "diggs:KD", ns_map)

        inv = investigations[inv_id]
        if ed > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="ED_kPa", value=ed,
                source="field", test_type="DMT",
            ))
        if id_dmt > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="ID_dmt", value=id_dmt,
                source="field", test_type="DMT",
            ))
        if kd > 0:
            inv.measurements.append(PointMeasurement(
                depth_m=depth, parameter="KD_dmt", value=kd,
                source="field", test_type="DMT",
            ))
