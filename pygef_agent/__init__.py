"""
pygef_agent — CPT and borehole file parser.

Wraps the pygef library for reading GEF (Dutch Geotechnical Exchange Format)
and BRO-XML CPT and borehole files. Converts to SI/kPa convention.

Public API
----------
parse_cpt_file : Parse a CPT file (GEF or BRO-XML) → CPTParseResult
parse_bore_file : Parse a borehole file (GEF or BRO-XML) → BoreParseResult
has_pygef : Check if pygef is installed
"""

from pygef_agent.pygef_utils import has_pygef
from pygef_agent.cpt_parser import parse_cpt_file
from pygef_agent.bore_parser import parse_bore_file
from pygef_agent.results import CPTParseResult, BoreParseResult

__all__ = [
    "parse_cpt_file",
    "parse_bore_file",
    "has_pygef",
    "CPTParseResult",
    "BoreParseResult",
]
