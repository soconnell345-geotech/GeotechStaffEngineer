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
    "parse_diggs26_samples", "parse_diggs26_water",
    "parse_diggs26_soundings", "SOUNDING_PROCEDURES",
    "SOUNDING_DEPTH_CLASS", "SOUNDING_EXTRA_MAP",
    "parse_diggs26_result_sets", "ResultSetRead",
]

_NS_GML = "http://www.opengis.net/gml/3.2"
_NS_XLINK = "http://www.w3.org/1999/xlink"
_NS_26 = "http://diggsml.org/schemas/2.6"
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
    #: What a laboratory sheet adds (WP3). Each is a dictionary term the lab
    #: writer emits; the right-hand names extend this package's own
    #: vocabulary, which is what makes the value reachable from a SiteModel.
    "shrinkage_limit": ("SL_pct", "lab", "Atterberg"),
    "percent_cobbles": ("pct_cobbles", "lab", "gradation"),
    "percent_silt": ("pct_silt", "lab", "gradation"),
    "d85": ("D85_mm", "lab", "gradation"),
    "coef_uniformity": ("Cu", "lab", "gradation"),
    "coef_curvature": ("Cc_grading", "lab", "gradation"),
    "coef_consolidation_vertical": ("cv_m2_per_s", "lab", "consolidation"),
    "degree_of_saturation": ("S_pct", "lab", "density"),
    "cohesion_residual": ("c_residual_kPa", "lab", "shear"),
    "friction_angle_residual": ("phi_residual_deg", "lab", "shear"),
    "LOI": ("organic_pct", "lab", "loss_on_ignition"),
    "pH": ("pH", "lab", "chemical"),
    "resistivity": ("resistivity_ohm_m", "lab", "chemical"),
    "resistivity_minimum": ("resistivity_min_ohm_m", "lab", "chemical"),
    "sulfate_content": ("sulfate", "lab", "chemical"),
    "chloride_content": ("chloride", "lab", "chemical"),
    "redox_potential": ("redox_mV", "lab", "chemical"),
    "conductivity": ("conductivity", "lab", "chemical"),
    "temperature": ("temperature_C", "lab", "chemical"),
    "cbr_0.1": ("CBR_0p1_pct", "lab", "CBR"),
    "cbr_0.2": ("CBR_0p2_pct", "lab", "CBR"),
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


def _split(text: str, separator: str) -> List[str]:
    """Split on a separator, treating whitespace as one."""
    if separator.strip() == "":
        return [part for part in str(text or "").split() if part != ""]
    return [part.strip() for part in str(text or "").split(separator)
            if part.strip() != ""]


def _rows(data_values, n_columns: int) -> List[List[str]]:
    """A ``dataValues`` element as a table of cells, as its own attributes say.

    ``cs`` separates the cells of a row and ``ts`` separates the rows; both
    are written on the element, and both are read here rather than assumed,
    because a single row of numbers is written with a space and anything with
    text or with more than one row is written with a semicolon.
    """
    if data_values is None:
        return []
    text = (data_values.text or "").strip()
    if not text:
        return []
    cs = data_values.get("cs", ",") or ","
    ts = data_values.get("ts", " ") or " "
    out: List[List[str]] = []
    for chunk in _split(text, ts):
        cells = _split(chunk, cs) if cs.strip() else [chunk]
        out.append(cells)
    if not out:
        return []
    # A row that is one cell long where several columns are declared is a
    # separator that did not separate -- a space-separated row of one value
    # per row, say. Re-reading it as one row keeps the file readable rather
    # than failing on it.
    if n_columns > 1 and all(len(row) == 1 for row in out) \
            and len(out) == n_columns:
        return [[row[0] for row in out]]
    return out


def _one_number(cell: str) -> Optional[float]:
    """One cell as a number, or None when it is text or missing."""
    text = str(cell or "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


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
        if procedure in SOUNDING_PROCEDURES:
            # A sounding's rows are read by parse_diggs26_soundings, which
            # takes each row's depth off the set's own depth column. Reading
            # it here as well would place a one-row sounding twice.
            continue
        for result in findall(test, ".//diggs:TestResult", ns_map):
            location = find(result, "diggs:location", ns_map)
            depths = _positions(location, find_text)
            if not depths:
                continue
            depth = depths[0]
            properties = findall(result, ".//diggs:Property", ns_map)
            classes = [find_text(prop, "diggs:propertyClass")
                       for prop in properties]
            data = find(result, ".//diggs:dataValues", ns_map)
            rows = _rows(data, len(classes))
            if not classes or not rows:
                continue
            if len(rows) > 1:
                # More than one row is a CURVE -- a grading, a consolidation,
                # an envelope, one row per point. A SiteModel holds values at
                # a depth and has nowhere to put a curve, and flattening one
                # into twelve measurements at the same depth would read as
                # twelve tests. Curves are read by
                # parse_diggs26_result_sets, which returns the table.
                continue
            for klass, cell in zip(classes, rows[0]):
                value = _one_number(cell)
                if value is None:
                    continue                      # text, or a missing cell
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


#: The two procedures whose result set is a SOUNDING: many rows against one
#: position, with the depth of each row in a column of the set rather than in
#: the test's own location. Everything else with many rows is a curve at one
#: depth (a grading, a consolidation) and is read by
#: :func:`parse_diggs26_result_sets`, which returns the table whole.
SOUNDING_PROCEDURES: Dict[str, str] = {
    "StaticConePenetrationTest": "CPTu",
    "DynamicProbeTest": "DCP",
}

#: What the depth column of a sounding's result set is named. Not a term of
#: the DIGGS property dictionary -- the dictionary has no depth property,
#: because in DIGGS a depth is a POSITION and not a value -- so a table of
#: readings against depth has to name its own, and this is the name
#: :mod:`report_ingest.diggs_writer` writes it under.
SOUNDING_DEPTH_CLASS = "sounding_depth"

#: The properties a sounding's result set carries that the dictionary does
#: not name, and what this package calls them. The dictionary's own terms
#: (``tip_resistance``, ``sleeve_friction``, ``pore_pressure_u2``,
#: ``blow_count``) come through :data:`PROPERTY_CLASS_MAP` like everything
#: else; these are the two a DCP sheet prints that DIGGS has no term for.
SOUNDING_EXTRA_MAP: Dict[str, Tuple[str, str, str]] = {
    "penetration": ("penetration_m", "field", "DCP"),
    "penetration_index": ("DPI", "field", "DCP"),
    "cbr_estimated": ("CBR_pct", "field", "DCP"),
}


def parse_diggs26_soundings(root, ns_map, investigations, gml_id_map,
                            warnings, *, find, findall, text) -> None:
    """A cone sounding or a dynamic probe: every row at its own depth.

    WHY THIS IS A SECOND READER AND NOT A BRANCH OF THE FIRST.
    :func:`parse_diggs26_tests` reads a result at THE TEST'S OWN POSITION,
    which is right for a blow count and a water content -- one value, one
    depth. A sounding is four hundred readings over a run of hole, written
    as ONE Test with one linear extent and a result set of four hundred
    rows, because that is what a ResultSet is for and writing four hundred
    Tests would be writing four hundred soundings. Its depth is therefore a
    COLUMN, and reading it means reading down the table rather than off the
    location.

    The procedures that get this treatment are named in
    :data:`SOUNDING_PROCEDURES` and nothing else does: a many-rowed result
    set under any other procedure is a curve at one depth and is left to
    :func:`parse_diggs26_result_sets`, which hands it back as a table.
    """
    def find_text(element, path):
        return text(element, path, ns_map, "")

    for test in findall(root, ".//diggs:Test", ns_map):
        procedure = ""
        proc = find(test, "diggs:procedure", ns_map)
        if proc is not None:
            for child in proc:
                tag = child.tag
                procedure = tag.split("}", 1)[1] if tag.startswith("{") \
                    else tag
                break
        if procedure not in SOUNDING_PROCEDURES:
            continue
        inv_id = _resolve_feature(test, find, ns_map, investigations,
                                  gml_id_map)
        if not inv_id or inv_id not in investigations:
            continue
        inv = investigations[inv_id]
        for result in findall(test, ".//diggs:TestResult", ns_map):
            properties = findall(result, ".//diggs:Property", ns_map)
            classes = [find_text(prop, "diggs:propertyClass")
                       for prop in properties]
            data = find(result, ".//diggs:dataValues", ns_map)
            rows = _rows(data, len(classes))
            if not classes or not rows:
                continue
            try:
                depth_at = classes.index(SOUNDING_DEPTH_CLASS)
            except ValueError:
                warnings.append(
                    f"{inv_id}: a {procedure} result set carries no "
                    f"{SOUNDING_DEPTH_CLASS!r} column, so its rows have no "
                    f"depth and none of them was read")
                continue
            for row in rows:
                if depth_at >= len(row):
                    continue
                depth = _one_number(row[depth_at])
                if depth is None:
                    continue
                for i, (klass, cell) in enumerate(zip(classes, row)):
                    if i == depth_at:
                        continue
                    value = _one_number(cell)
                    if value is None:
                        continue      # a channel this reading did not carry
                    key = (klass or "").strip()
                    mapped = PROPERTY_CLASS_MAP.get(key) \
                        or SOUNDING_EXTRA_MAP.get(key)
                    if mapped is None:
                        continue
                    parameter, source, _test_type = mapped
                    if parameter == "__water_level__":
                        continue
                    inv.measurements.append(PointMeasurement(
                        depth_m=depth, parameter=parameter, value=value,
                        source=source,
                        test_type=SOUNDING_PROCEDURES[procedure],
                    ))


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


class ResultSetRead:
    """One ``Test`` read as the table it is, with nothing thrown away.

    ``parse_diggs`` builds a :class:`SiteModel`, which holds a value at a
    depth. That is the right shape for a blow count and the wrong one for a
    grading curve, a consolidation curve or a failure envelope -- twelve
    numbers at one depth that mean one thing together. This is the other
    reading: every result set in the file, exactly as written, columns and
    rows and units, whether or not the property is one the SiteModel has a
    name for.

    It is what a round-trip check compares against, and what a consumer wants
    when it is after the laboratory data rather than a profile.
    """

    __slots__ = ("test_id", "name", "procedure", "investigation_id",
                 "sampling_feature", "depth_m", "bottom_depth_m", "columns",
                 "rows", "properties")

    def __init__(self, test_id: str, name: str, procedure: str,
                 investigation_id: Optional[str], sampling_feature: str,
                 depth_m: Optional[float], bottom_depth_m: Optional[float],
                 columns: List[Dict[str, str]],
                 rows: List[List[Any]],
                 properties: Dict[str, str]) -> None:
        self.test_id = test_id
        self.name = name
        self.procedure = procedure
        self.investigation_id = investigation_id
        self.sampling_feature = sampling_feature
        self.depth_m = depth_m
        self.bottom_depth_m = bottom_depth_m
        #: ``{"name", "class", "uom", "type", "codeSpace"}`` per column.
        self.columns = columns
        #: One list per row; a numeric cell is a float, a text cell a string,
        #: a missing cell None.
        self.rows = rows
        #: What the Test carried as ``otherMeasurementProperty``: the sample
        #: label, the laboratory, the date, the pages.
        self.properties = properties

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def column(self, property_class: str) -> Optional[int]:
        """The index of the column with that ``propertyClass``, or None."""
        for i, column in enumerate(self.columns):
            if column["class"] == property_class:
                return i
        return None

    def values(self, property_class: str) -> List[Any]:
        """Every cell of that column, in row order."""
        i = self.column(property_class)
        if i is None:
            return []
        return [row[i] if i < len(row) else None for row in self.rows]

    def scalar(self, property_class: str) -> Any:
        """The single value of that column, or None."""
        got = self.values(property_class)
        return got[0] if len(got) == 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return {"test_id": self.test_id, "name": self.name,
                "procedure": self.procedure,
                "investigation_id": self.investigation_id,
                "depth_m": self.depth_m,
                "bottom_depth_m": self.bottom_depth_m,
                "columns": [dict(c) for c in self.columns],
                "rows": [list(r) for r in self.rows],
                "properties": dict(self.properties)}

    def __repr__(self) -> str:                     # pragma: no cover - repr
        return (f"ResultSetRead({self.name!r}, {self.investigation_id!r}, "
                f"{self.depth_m!r} m, {len(self.columns)} column(s), "
                f"{len(self.rows)} row(s))")


def parse_diggs26_result_sets(filepath: Optional[str] = None,
                              content: Optional[str] = None
                              ) -> List[ResultSetRead]:
    """Every ``measurement/Test`` in a DIGGS 2.6 file, as its own table.

    Additive and standalone: ``parse_diggs`` is untouched and still returns a
    SiteModel. This reads the same file a second way, for the results a
    SiteModel has no shape for -- curves, values the dictionary has no term
    for, and values a laboratory printed as words.
    """
    import xml.etree.ElementTree as ET

    if filepath is None and content is None:
        raise ValueError("Must provide either filepath or content")
    root = (ET.fromstring(content) if content is not None
            else ET.parse(filepath).getroot())
    ns = _NS_26 if _NS_26 in root.tag else (
        _NS_25A if _NS_25A in root.tag else _NS_26)
    d = f"{{{ns}}}"

    names: Dict[str, str] = {}
    for feature in ("Borehole", "TrialPit"):
        for element in root.iter(f"{d}{feature}"):
            gml_id = element.get(f"{{{_NS_GML}}}id", "")
            label = element.find(f"{{{_NS_GML}}}name")
            if gml_id:
                names[gml_id] = ((label.text or "").strip() if label is not None
                                 else gml_id)

    out: List[ResultSetRead] = []
    for test in root.iter(f"{d}Test"):
        test_id = test.get(f"{{{_NS_GML}}}id", "")
        label = test.find(f"{{{_NS_GML}}}name")
        name = (label.text or "").strip() if label is not None else ""
        feature_id = ""
        ref = test.find(f"{d}samplingFeatureRef")
        if ref is not None:
            feature_id = (ref.get(f"{{{_NS_XLINK}}}href", "")
                          or ref.get("xlink:href", "")).lstrip("#")
        procedure = ""
        proc = test.find(f"{d}procedure")
        if proc is not None:
            for child in proc:
                tag = child.tag
                procedure = (tag.split("}", 1)[1] if tag.startswith("{")
                             else tag)
                break
        properties: Dict[str, str] = {}
        for parameter in test.iter(f"{d}Parameter"):
            key = parameter.find(f"{d}parameterName")
            value = parameter.find(f"{d}parameterValue")
            if key is not None and (key.text or "").strip():
                properties[(key.text or "").strip()] = (
                    (value.text or "").strip() if value is not None else "")
        for result in test.iter(f"{d}TestResult"):
            depths: List[float] = []
            location = result.find(f"{d}location")
            if location is not None:
                for path in (f".//{{{_NS_GML}}}posList",
                             f".//{{{_NS_GML}}}pos"):
                    element = location.find(path)
                    if element is not None:
                        depths = [n for n in _numbers(element.text or "")
                                  if n == n]
                        if depths:
                            break
            columns: List[Dict[str, str]] = []
            for prop in result.iter(f"{d}Property"):
                klass = prop.find(f"{d}propertyClass")
                columns.append({
                    "name": _child_text(prop, f"{d}propertyName"),
                    "class": _child_text(prop, f"{d}propertyClass"),
                    "uom": _child_text(prop, f"{d}uom"),
                    "type": _child_text(prop, f"{d}typeData"),
                    "codeSpace": (klass.get("codeSpace", "")
                                  if klass is not None else ""),
                })
            data = result.find(f".//{d}dataValues")
            rows: List[List[Any]] = []
            for cells in _rows(data, len(columns)):
                row: List[Any] = []
                for i, cell in enumerate(cells):
                    kind = columns[i]["type"] if i < len(columns) else ""
                    if kind == "string" or _one_number(cell) is None:
                        row.append(None if str(cell).strip() in ("", "-")
                                   else str(cell).strip())
                    else:
                        row.append(_one_number(cell))
                rows.append(row)
            out.append(ResultSetRead(
                test_id=test_id, name=name, procedure=procedure,
                investigation_id=names.get(feature_id),
                sampling_feature=feature_id,
                depth_m=(depths[0] if depths else None),
                bottom_depth_m=(depths[1] if len(depths) > 1 else None),
                columns=columns, rows=rows, properties=properties))
    return out


def _child_text(element, tag: str) -> str:
    child = element.find(tag)
    return (child.text or "").strip() if child is not None else ""


def parse_diggs26_samples(root, ns_map, investigations, gml_id_map, warnings,
                          *, find, findall, text) -> None:
    """What a ``SamplingActivity`` records about the sample it took.

    Recovery and rock quality designation are properties of the ACTIVITY in
    DIGGS, not of a test, so nothing in the Test reader reaches them. They
    are the two numbers a rock-core log prints most often, and before this
    they read back as nothing.
    """
    def find_text(element, path):
        return text(element, path, ns_map, "")

    for activity in findall(root, ".//diggs:SamplingActivity", ns_map):
        inv_id = _resolve_feature(activity, find, ns_map, investigations,
                                  gml_id_map)
        if not inv_id or inv_id not in investigations:
            continue
        inv = investigations[inv_id]
        depths = _positions(find(activity, "diggs:samplingLocation", ns_map),
                            find_text)
        if not depths:
            continue
        depth = depths[0]
        rqd = find_text(activity, "diggs:samplingActivityRQD")
        if rqd.strip():
            try:
                inv.measurements.append(PointMeasurement(
                    depth_m=depth, parameter="RQD_pct", value=float(rqd),
                    source="field", test_type="core_run"))
            except ValueError:
                pass
        for parameter in findall(activity,
                                 ".//diggs:otherSamplingActivityProperty"
                                 "/diggs:Parameter", ns_map):
            name = find_text(parameter, "diggs:parameterName").strip()
            if name != "recovery_percent":
                continue
            try:
                inv.measurements.append(PointMeasurement(
                    depth_m=depth, parameter="recovery_pct",
                    value=float(find_text(parameter, "diggs:parameterValue")),
                    source="field", test_type="core_run"))
            except ValueError:
                pass


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
