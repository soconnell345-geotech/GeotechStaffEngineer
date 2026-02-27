"""
DIGGS 2.6 XML parser → SiteModel.

Uses xml.etree.ElementTree (stdlib). Auto-detects 2.6 vs 2.5.a namespace.
Extracts borings, lithology, SPT, Atterberg limits, moisture content, GWL.
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
