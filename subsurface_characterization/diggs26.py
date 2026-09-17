"""Reading DIGGS 2.6 as the published schema actually defines it.

WHY THIS MODULE EXISTS. ``diggs_parser`` reads a FLAT dialect --
``<DrivenPenetrationTest><depth/><blowCount/></DrivenPenetrationTest>`` and
the like, in the DIGGS namespace, with the test elements sitting at the
document root. That dialect is what the parser was built against, what its
fixtures are written in, and it keeps working exactly as before.

A file written to the PUBLISHED DIGGS 2.6 schema does not look like that, and
before this module none of it was read. Five differences, all structural:

* every test PROCEDURE -- ``DrivenPenetrationTest``, ``AtterbergLimitsTest``,
  ``WaterContentTest``, ``LabDensityTest``, ``ParticleSizeTest``,
  ``UnconfinedCompressiveStrengthTest``, ``PocketPenetrometerTest`` -- is in
  the ``.../2.6/geotechnical`` namespace, not the DIGGS one, and describes
  only HOW a test was done. It carries no result;
* the VALUE lives in ``Test/outcome/TestResult/results/ResultSet``, named by a
  ``propertyClass`` from the DIGGS property dictionary;
* a DEPTH is not an element. It is a position along the borehole's own linear
  reference system, written as a ``LinearExtent`` whose ``gml:posList`` holds
  the top and the base;
* lithology is an ``observation/LithologySystem`` at the document root that
  points back at its borehole, not a child of the borehole;
* there is no ``WaterLevelObservation`` and no ``MoistureContent`` element in
  2.6 at all. Water is the borehole's own ``waterStrike``.

The readers here run after the flat ones and only ever APPEND. A file in
either shape reads; a file carrying both reads both; and nothing that worked
before behaves differently.

The borehole header -- name, position, total depth -- needs nothing new: the
flat parser's own ``_parse_borehole`` already reads it out of a conformant
2.6 file, because that part of the two shapes agrees.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from subsurface_characterization.site_model import (
    LithologyInterval, PointMeasurement,
)

__all__ = [
    "PROPERTY_CLASS_MAP", "parse_diggs26_lithology", "parse_diggs26_tests",
    "parse_diggs26_water",
]

_NS_GML = "http://www.opengis.net/gml/3.2"
_NS_XLINK = "http://www.w3.org/1999/xlink"
_NS_25A = "http://diggsml.org/schemas/2.5.a"
#: Where the test procedures live. This namespace, not the DIGGS one, is the
#: single fact that made a conformant file read as empty.
_NS_GEO_26 = "http://diggsml.org/schemas/2.6/geotechnical"
_NS_GEO_25A = "http://diggsml.org/schemas/2.5.a/geotechnical"

#: ``propertyClass -> (parameter, source, test type)``. The left side is what
#: a DIGGS file names its values by -- the ids in pydiggs'
#: ``dictionaries/properties.xml`` -- and the right side is the vocabulary the
#: rest of this package uses (``site_model.STANDARD_PARAMETERS``). Adding a
#: row here is how a new test kind becomes readable.
PROPERTY_CLASS_MAP: Dict[str, Tuple[str, str, str]] = {
    "n_value": ("N_spt", "field", "SPT"),
    "n60": ("N60", "field", "SPT"),
    "n1_60": ("N160", "field", "SPT"),
    "blow_count": ("blow_count", "field", "SPT"),
    "water_content_natural": ("wn_pct", "lab", "moisture_content"),
    "soil_moisture_content": ("wn_pct", "lab", "moisture_content"),
    "water_content_optimum": ("wn_opt_pct", "lab", "compaction"),
    "dry_density": ("gamma_d_kNm3", "lab", "density"),
    "dry_density_max": ("gamma_max_kNm3", "lab", "compaction"),
    "bulk_density": ("gamma_kNm3", "lab", "density"),
    "unit_weight": ("gamma_kNm3", "lab", "density"),
    "liquid_limit": ("LL_pct", "lab", "Atterberg"),
    "plastic_limit": ("PL_pct", "lab", "Atterberg"),
    "plasticity_index": ("PI_pct", "lab", "Atterberg"),
    "percent_fines": ("pct_fines", "lab", "gradation"),
    "percent_sand": ("pct_sand", "lab", "gradation"),
    "percent_gravel": ("pct_gravel", "lab", "gradation"),
    "d10": ("D10_mm", "lab", "gradation"),
    "d30": ("D30_mm", "lab", "gradation"),
    "d60": ("D60_mm", "lab", "gradation"),
    "compressive_strength_unconfined": ("qu_kPa", "lab", "UCS"),
    "shear_strength_undrained": ("cu_kPa", "lab", "undrained_shear"),
    "shear_strength_undrained_residual": ("Su_vane_remolded_kPa", "field",
                                          "vane"),
    "sensitivity": ("St", "field", "vane"),
    "friction_angle_peak": ("phi_deg", "lab", "shear"),
    "cohesion_peak": ("c_kPa", "lab", "shear"),
    "specific_gravity_solids": ("Gs", "lab", "specific_gravity"),
    "point_load_test_index": ("Is50_MPa", "lab", "point_load"),
    "coef_permeability": ("k_m_per_s", "lab", "permeability"),
    "compression_index": ("Cc", "lab", "consolidation"),
    "recompression_index": ("Cr", "lab", "consolidation"),
    "preconsolidation_pressure": ("sigma_p_kPa", "lab", "consolidation"),
    "tip_resistance": ("qc_kPa", "field", "CPTu"),
    "tip_resistance_corrected": ("qc_kPa", "field", "CPTu"),
    "sleeve_friction": ("fs_kPa", "field", "CPTu"),
    "pore_pressure_u2": ("u2_kPa", "field", "CPTu"),
    "friction_ratio": ("Rf_pct", "field", "CPTu"),
    "dilatometer_modulus": ("ED_kPa", "field", "DMT"),
    "material_index": ("ID_dmt", "field", "DMT"),
    "horiz_stress_index": ("KD_dmt", "field", "DMT"),
    "limit_pressure": ("p_limit_kPa", "field", "pressuremeter"),
    "modulus_youngs": ("E_pmt_kPa", "field", "pressuremeter"),
    #: Not a measurement at a depth: it sets the investigation's water level.
    "water_depth": ("__water_level__", "field", "water_level"),
    "water_depth_calc": ("__water_level__", "field", "water_level"),
    "water_depth_estimated": ("__water_level__", "field", "water_level"),
}


def _geo_ns(ns_map: dict) -> str:
    """The geotechnical namespace matching this file's DIGGS version."""
    return _NS_GEO_25A if ns_map.get("diggs") == _NS_25A else _NS_GEO_26


def _numbers(text: str) -> List[float]:
    out: List[float] = []
    for token in str(text or "").replace(",", " ").split():
        try:
            out.append(float(token))
        except ValueError:
            out.append(float("nan"))
    return out


def _positions(element, find_text) -> List[float]:
    """The numbers in a descendant ``gml:posList`` or ``gml:pos``.

    A DIGGS depth is a position along the borehole's linear reference system:
    one number for a point, two for an interval.
    """
    if element is None:
        return []
    for path in (".//gml:posList", ".//gml:pos"):
        numbers = [n for n in _numbers(find_text(element, path)) if n == n]
        if numbers:
            return numbers
    return []


def _ref_id(element, find, ns_map: dict, *tags: str) -> Optional[str]:
    """The ``gml:id`` a reference element points at, without its hash."""
    for tag in tags:
        child = find(element, tag, ns_map)
        if child is None:
            continue
        href = (child.get(f"{{{_NS_XLINK}}}href", "")
                or child.get("xlink:href", ""))
        if href:
            return href.lstrip("#")
    return None


def _resolve_feature(element, find, ns_map: dict, investigations: dict,
                     gml_id_map: dict) -> Optional[str]:
    """Which investigation a 2.6 element belongs to.

    ``samplingFeatureRef`` is how 2.6 says it. The one-investigation fallback
    is the same courtesy the flat readers extend: a file with a single hole
    cannot be ambiguous.
    """
    gml_id = _ref_id(element, find, ns_map, "diggs:samplingFeatureRef",
                     "diggs:investigationRef")
    if gml_id and gml_id in gml_id_map:
        return gml_id_map[gml_id]
    if gml_id is None and len(investigations) == 1:
        return next(iter(investigations))
    if gml_id and gml_id not in gml_id_map and len(investigations) == 1:
        return next(iter(investigations))
    return None


# ---------------------------------------------------------------------------
# the three readers
# ---------------------------------------------------------------------------

def parse_diggs26_lithology(root, ns_map, investigations, gml_id_map,
                            warnings, *, find, findall, text) -> None:
    """Lithology from ``observation/LithologySystem``."""
    for system in findall(root, ".//diggs:LithologySystem", ns_map):
        inv_id = _resolve_feature(system, find, ns_map, investigations,
                                  gml_id_map)
        if not inv_id or inv_id not in investigations:
            continue
        inv = investigations[inv_id]
        for obs in findall(system, ".//diggs:LithologyObservation", ns_map):
            location = find(obs, "diggs:location", ns_map)
            depths = _positions(location,
                                lambda e, p: text(e, p, ns_map, ""))
            if len(depths) < 2:
                # A stratum with a top and no base is a contact, not an
                # interval, and LithologyInterval cannot hold one. Said,
                # rather than dropped silently.
                warnings.append(
                    f"{inv_id}: a lithology observation carries "
                    f"{len(depths)} depth(s), not a top and a base; skipped")
                continue
            top, bottom = depths[0], depths[1]
            if bottom <= top:
                continue
            lith = find(obs, ".//diggs:Lithology", ns_map)
            if lith is None:
                continue
            props = find(lith, ".//diggs:FieldProperties", ns_map)
            inv.lithology.append(LithologyInterval(
                top_depth_m=top,
                bottom_depth_m=bottom,
                description=(text(lith, "diggs:lithDescription", ns_map, "")
                             or text(lith, "diggs:description", ns_map, "")),
                uscs=text(lith, "diggs:classificationCode", ns_map, ""),
                color=text(lith, ".//diggs:colorName", ns_map, ""),
                moisture=("" if props is None else
                          text(props, "diggs:moistureCondition", ns_map, "")),
                consistency_density=(
                    "" if props is None else
                    text(props, "diggs:consistency", ns_map, "")),
            ))


def parse_diggs26_tests(root, ns_map, investigations, gml_id_map, warnings, *,
                        find, findall, text) -> None:
    """Values from ``measurement/Test``.

    The depth comes from the result's own location; each value comes from the
    ResultSet, named by its Property's ``propertyClass``. The procedure
    element is read only for its name, which is what it is for.
    """
    geo = _geo_ns(ns_map)

    def find_text(element, path):
        return text(element, path, ns_map, "")

    for test in findall(root, ".//diggs:Test", ns_map):
        inv_id = _resolve_feature(test, find, ns_map, investigations,
                                  gml_id_map)
        if not inv_id or inv_id not in investigations:
            continue
        inv = investigations[inv_id]
        procedure = ""
        proc = find(test, "diggs:procedure", ns_map)
        if proc is not None:
            for child in proc:
                tag = child.tag
                procedure = tag.split("}", 1)[1] if tag.startswith("{") else tag
                break
        for result in findall(test, ".//diggs:TestResult", ns_map):
            location = find(result, "diggs:location", ns_map)
            depths = _positions(location, find_text)
            if not depths:
                continue
            depth = depths[0]
            classes = [find_text(prop, "diggs:propertyClass")
                       for prop in findall(result, ".//diggs:Property",
                                           ns_map)]
            values = _numbers(find_text(result, ".//diggs:dataValues"))
            if not classes or not values:
                continue
            for klass, value in zip(classes, values):
                if value != value:                    # NaN: not a number
                    continue
                mapped = PROPERTY_CLASS_MAP.get((klass or "").strip())
                if mapped is None:
                    continue
                parameter, source, test_type = mapped
                if parameter == "__water_level__":
                    if inv.gwl_depth_m is None and value > 0:
                        inv.gwl_depth_m = value
                    continue
                inv.measurements.append(PointMeasurement(
                    depth_m=depth, parameter=parameter, value=value,
                    source=source, test_type=procedure or test_type,
                ))
        if procedure == "DrivenPenetrationTest" and proc is not None:
            _drive_sets(test, proc, ns_map, geo, inv, find, find_text)


def _drive_sets(test, procedure, ns_map, geo, inv, find, find_text) -> None:
    """Each drive of a driven test, at the test's own depth.

    A log that prints only the drives and no N value is not a gap: the drives
    ARE what the page said, and they are recorded so that a round trip can
    find them. A drive already carried by the ResultSet is not recorded twice.
    """
    location = find(test, ".//diggs:TestResult/diggs:location", ns_map)
    depths = _positions(location, find_text)
    if not depths:
        return
    depth = depths[0]
    already = {(m.depth_m, m.value) for m in inv.measurements
               if m.parameter == "blow_count"}
    for drive in procedure.iter(f"{{{geo}}}DriveSet"):
        count = drive.find(f"{{{geo}}}blowCount")
        if count is None or not (count.text or "").strip():
            continue
        try:
            value = float(count.text.strip())
        except ValueError:
            continue
        if (depth, value) in already:
            continue
        already.add((depth, value))
        inv.measurements.append(PointMeasurement(
            depth_m=depth, parameter="blow_count", value=value,
            source="field", test_type="SPT"))


def parse_diggs26_water(root, ns_map, investigations, gml_id_map, warnings, *,
                        find, findall, text) -> None:
    """Water from the borehole's own ``waterStrike``.

    The record can hold several readings -- while drilling, at completion,
    after a day -- and DIGGS keeps them all. ``Investigation`` has one
    ``gwl_depth_m``, so the INITIAL strike is the one it gets; the rest stay
    in the file for a consumer that wants them.
    """
    def find_text(element, path):
        return text(element, path, ns_map, "")

    for feature in ("diggs:Borehole", "diggs:TrialPit"):
        for element in findall(root, f".//{feature}", ns_map):
            gml_id = element.get(f"{{{_NS_GML}}}id", "")
            inv_id = gml_id_map.get(gml_id)
            if inv_id is None or inv_id not in investigations:
                continue
            inv = investigations[inv_id]
            if inv.gwl_depth_m is not None:
                continue
            for strike in findall(element, ".//diggs:WaterStrike", ns_map):
                reading = find(strike, "diggs:initialWaterStrikeReading",
                               ns_map)
                if reading is None:
                    continue
                depths = _positions(
                    find(reading, ".//diggs:waterLocation", ns_map),
                    find_text)
                if depths:
                    inv.gwl_depth_m = depths[0]
                    break
