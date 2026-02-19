"""
Borehole file parser using pygef.

Reads GEF and BRO-XML borehole files and converts to project conventions.
"""

import numpy as np

from pygef_agent.pygef_utils import import_pygef
from pygef_agent.results import BoreParseResult


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
