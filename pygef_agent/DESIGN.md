# pygef Agent — Design Notes

## Purpose

Wraps the pygef library for parsing CPT and borehole files in GEF
(Dutch Geotechnical Exchange Format) and BRO-XML formats. Converts
data to project conventions (kPa, numpy arrays) and provides a bridge
to liquepy_agent for CPT-based liquefaction analysis.

## Architecture

```
pygef_agent/
  __init__.py              # exports public API
  pygef_utils.py           # has_pygef(), import helpers
  cpt_parser.py            # parse_cpt_file()
  bore_parser.py           # parse_bore_file()
  results.py               # CPTParseResult, BoreParseResult
  tests/
    test_pygef_agent.py
    sample_cpt.gef         # test CPT file
    sample_bore.gef        # test borehole file
  DESIGN.md
pygef_agent_foundry.py     # Foundry wrapper (project root)
```

## Key Design Decisions

1. **Optional dependency** — pygef is checked with `has_pygef()` following
   the standard pattern. Tier 1 tests work without pygef installed.

2. **MPa → kPa conversion** — pygef stores cone resistance, friction, and
   pore pressures in **MPa**. We multiply by 1000 at the parse boundary.

3. **polars → numpy** — pygef uses polars DataFrames internally. We extract
   columns via `.to_list()` and convert to numpy arrays.

4. **GEF/XML only** — pygef only supports GEF format and BRO-XML. No CSV.

5. **Bridge to liquepy** — `CPTParseResult.to_liquepy_inputs()` produces a
   dict ready for `analyze_cpt_liquefaction()`.

6. **Robust metadata extraction** — pygef sometimes misparsms GEF headers:
   - `final_depth` computed from max penetration length (data), not header
   - GWL extracted from raw `MEASUREMENTVAR #30` as fallback
   - `final_depth_offset` property has infinite recursion bug — avoided

## Unit Conversions

| pygef column | pygef unit | Project unit | Conversion |
|-------------|-----------|-------------|------------|
| coneResistance | MPa | kPa | × 1000 |
| localFriction | MPa | kPa | × 1000 |
| porePressureU2 | MPa | kPa | × 1000 |
| frictionRatio | % | % | none |
| penetrationLength | m | m | none |

## Supported File Formats

1. **GEF** (Geotechnical Exchange Format)
   - Detected by `#GEFID` header
   - CPT: `#REPORTCODE= GEF-CPT-Report` or `#PROCEDURECODE` with "cpt"
   - Bore: `#REPORTCODE= GEF-BORE-Report` or `#PROCEDURECODE` with "bore"

2. **BRO-XML** (Dutch subsurface registration)
   - CPT: `dispatchDocument` root with `cptcommon:` namespace
   - Bore: `dispatchDocument` root with `bhrgtcom:` namespace

## Edge Cases

- **Missing columns**: Only `penetrationLength` and `coneResistance` are
  guaranteed. All other columns (fs, u2, Rf) may be absent.
- **Void values**: pygef replaces column void markers with null. We replace
  null/NaN with 0.0 after numpy conversion.
- **Pre-drilled depth**: Rows above pre-drilled depth are removed by default.
- **Multi-record XML**: Use `index` parameter to select specific record.
