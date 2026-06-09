"""
GEF / BRO-XML format adapter (pygef-backed).

Wraps the optional ``pygef`` library for reading GEF (Dutch Geotechnical Exchange
Format) and BRO-XML CPT and borehole files. Converts to SI/kPa convention.

``pygef`` is an OPTIONAL dependency — call ``has_pygef()`` to check availability at
runtime; the parse functions raise a clear RuntimeError if it is missing.

Public API
----------
parse_cpt_file : Parse a CPT file (GEF or BRO-XML) -> CPTParseResult
parse_bore_file : Parse a borehole file (GEF or BRO-XML) -> BoreParseResult
has_pygef : Check if pygef is installed
"""

import numpy as np

from subsurface_characterization.formats.gef_results import (
    CPTParseResult,
    BoreParseResult,
)


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

def has_pygef():
    """Return True if pygef is installed and importable."""
    try:
        import pygef  # noqa: F401
        return True
    except ImportError:
        return False


def import_pygef():
    """Import and return the pygef package, raising RuntimeError if missing."""
    try:
        import pygef
        return pygef
    except ImportError:
        raise RuntimeError(
            "pygef is not installed. Install with: pip install pygef"
        )


# ---------------------------------------------------------------------------
# CPT parsing
# ---------------------------------------------------------------------------

def _validate_cpt_parse_inputs(file_path, engine):
    """Validate CPT parsing inputs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not file_path:
        raise ValueError("file_path is required")
    valid_engines = {"auto", "gef", "xml"}
    if engine not in valid_engines:
        raise ValueError(
            f"engine must be one of {sorted(valid_engines)}, got '{engine}'"
        )


def parse_cpt_file(
    file_path,
    engine="auto",
    index=0,
) -> CPTParseResult:
    """Parse a CPT file (GEF or BRO-XML) into standardized arrays.

    Reads the file using pygef and converts all pressures from MPa to kPa.

    Parameters
    ----------
    file_path : str or Path
        Path to the CPT file (.gef or .xml).
    engine : str, optional
        Parser engine: 'auto', 'gef', or 'xml'. Default: 'auto'.
    index : int, optional
        Record index for multi-record XML files. Default: 0.

    Returns
    -------
    CPTParseResult
        Parsed CPT data with arrays in kPa.
    """
    _validate_cpt_parse_inputs(file_path, engine)

    pygef = import_pygef()
    cpt = pygef.read_cpt(file_path, index=index, engine=engine)

    # Extract metadata
    alias = cpt.alias or str(file_path)
    predrilled = float(cpt.predrilled_depth) if cpt.predrilled_depth else 0.0
    gwl = float(cpt.groundwater_level) if cpt.groundwater_level is not None else None

    # pygef may misparse final_depth from MEASUREMENTVAR — use data max instead
    depth_col = np.asarray(cpt.data["penetrationLength"].to_list(), dtype=float)
    final_depth = float(np.max(depth_col)) if len(depth_col) > 0 else 0.0

    # Fallback: extract GWL from raw GEF headers if pygef didn't populate it
    if gwl is None and hasattr(cpt, 'raw_headers') and cpt.raw_headers:
        mvars = cpt.raw_headers.get('MEASUREMENTVAR', [])
        for mv in mvars:
            if len(mv) >= 2 and mv[0].strip() == '30':
                try:
                    gwl = float(mv[1])
                except (ValueError, TypeError):
                    pass
                break

    # Location
    x, y, srs = None, None, ""
    if cpt.delivered_location is not None:
        x = float(cpt.delivered_location.x)
        y = float(cpt.delivered_location.y)
        srs = cpt.delivered_location.srs_name or ""

    # Available columns
    available_columns = cpt.columns

    # Extract data arrays (polars → numpy)
    df = cpt.data

    # Depth (always present as penetrationLength)
    depth_m = np.asarray(df["penetrationLength"].to_list(), dtype=float)

    # Cone resistance (always present, in MPa → kPa)
    q_c_kPa = np.asarray(df["coneResistance"].to_list(), dtype=float) * 1000

    # Sleeve friction (optional, MPa → kPa)
    f_s_kPa = np.array([])
    if "localFriction" in available_columns:
        vals = df["localFriction"].to_list()
        f_s_kPa = np.asarray(vals, dtype=float) * 1000

    # Pore pressure u2 (optional, MPa → kPa)
    u_2_kPa = np.array([])
    if "porePressureU2" in available_columns:
        vals = df["porePressureU2"].to_list()
        u_2_kPa = np.asarray(vals, dtype=float) * 1000

    # Friction ratio (optional, already in %)
    Rf_pct = np.array([])
    if "frictionRatio" in available_columns:
        vals = df["frictionRatio"].to_list()
        Rf_pct = np.asarray(vals, dtype=float)
    elif "frictionRatioComputed" in available_columns:
        vals = df["frictionRatioComputed"].to_list()
        Rf_pct = np.asarray(vals, dtype=float)

    # Handle NaN values from null polars values
    for arr_name in ('q_c_kPa', 'f_s_kPa', 'u_2_kPa', 'Rf_pct'):
        arr = locals()[arr_name]
        if len(arr) > 0:
            arr[~np.isfinite(arr)] = 0.0

    n_points = len(depth_m)

    return CPTParseResult(
        n_points=n_points,
        alias=alias,
        final_depth_m=final_depth,
        predrilled_depth_m=predrilled,
        gwl_m=gwl,
        x=x,
        y=y,
        srs_name=srs,
        depth_m=depth_m,
        q_c_kPa=q_c_kPa,
        f_s_kPa=f_s_kPa,
        u_2_kPa=u_2_kPa,
        Rf_pct=Rf_pct,
        available_columns=available_columns,
    )


# ---------------------------------------------------------------------------
# Borehole parsing
# ---------------------------------------------------------------------------

def _validate_bore_parse_inputs(file_path, engine):
    """Validate borehole parsing inputs.

    Raises
    ------
    ValueError
        If inputs are invalid.
    """
    if not file_path:
        raise ValueError("file_path is required")
    valid_engines = {"auto", "gef", "xml"}
    if engine not in valid_engines:
        raise ValueError(
            f"engine must be one of {sorted(valid_engines)}, got '{engine}'"
        )


def parse_bore_file(
    file_path,
    engine="auto",
    index=0,
) -> BoreParseResult:
    """Parse a borehole file (GEF or BRO-XML) into standardized format.

    Parameters
    ----------
    file_path : str or Path
        Path to the borehole file (.gef or .xml).
    engine : str, optional
        Parser engine: 'auto', 'gef', or 'xml'. Default: 'auto'.
    index : int, optional
        Record index for multi-record XML files. Default: 0.

    Returns
    -------
    BoreParseResult
        Parsed borehole data with layer boundaries and soil descriptions.
    """
    _validate_bore_parse_inputs(file_path, engine)

    pygef = import_pygef()
    bore = pygef.read_bore(file_path, index=index, engine=engine)

    # Extract metadata
    alias = bore.alias or str(file_path)
    gwl = float(bore.groundwater_level) if bore.groundwater_level is not None else None

    # Location
    x, y, srs = None, None, ""
    if bore.delivered_location is not None:
        x = float(bore.delivered_location.x)
        y = float(bore.delivered_location.y)
        srs = bore.delivered_location.srs_name or ""

    # Extract layer data (polars → numpy/lists)
    df = bore.data
    columns = bore.columns

    top_m = np.asarray(df["upperBoundary"].to_list(), dtype=float)
    bottom_m = np.asarray(df["lowerBoundary"].to_list(), dtype=float)

    # Compute final depth from data (pygef may not populate final_bore_depth)
    if bore.final_bore_depth is not None and bore.final_bore_depth > 0:
        final_depth = float(bore.final_bore_depth)
    elif len(bottom_m) > 0:
        final_depth = float(np.max(bottom_m))
    else:
        final_depth = 0.0

    soil_name = df["geotechnicalSoilName"].to_list()

    soil_code = []
    if "geotechnicalSoilCode" in columns:
        soil_code = df["geotechnicalSoilCode"].to_list()

    n_layers = len(top_m)

    return BoreParseResult(
        n_layers=n_layers,
        alias=alias,
        final_depth_m=final_depth,
        gwl_m=gwl,
        x=x,
        y=y,
        srs_name=srs,
        top_m=top_m,
        bottom_m=bottom_m,
        soil_name=soil_name,
        soil_code=soil_code,
    )
