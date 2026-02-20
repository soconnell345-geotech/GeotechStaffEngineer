# AGS4 Agent — Design Notes

## Purpose

Wraps the python-ags4 library for reading, parsing, and validating
AGS4 (Association of Geotechnical and Geoenvironmental Specialists)
data exchange files.

## Background

AGS4 is the standard digital data format for exchanging geotechnical
and geoenvironmental data in the UK and increasingly worldwide. An
AGS4 file is a structured text file with:

- **GROUPs** (tables) — e.g. PROJ (project), HOLE (boreholes),
  SAMP (samples), GEOL (geology), ISPT (SPT results)
- **HEADINGs** — column names
- **UNIT** and **TYPE** rows — units and data types
- **DATA** rows — actual measurements

## Architecture

```
ags4_agent/
  __init__.py          # exports read_ags4, validate_ags4 + result classes
  ags4_utils.py        # has_ags4(), import_ags4()
  ags4_reader.py       # read_ags4(), validate_ags4()
  results.py           # AGS4ReadResult, AGS4ValidationResult
  tests/
    test_ags4_agent.py
  DESIGN.md
ags4_agent_foundry.py  # Foundry wrapper (project root)
```

## Key Design Decisions

1. **String input supported** — AGS4 content can be passed as a string
   (via StringIO), not just file paths. This allows LLM agents to work
   with AGS4 data embedded in API responses.

2. **Two methods only** — `read_ags4` (parse) and `validate_ags4` (check).
   Writing AGS4 is not exposed because LLM agents should not generate
   geotechnical data files.

3. **Numeric conversion** — By default, numeric columns (DP, SF, SCI, MC
   types) are converted from text to numbers, enabling calculations.

4. **JSON-friendly output** — Tables converted to list-of-dicts format
   for JSON serialization via to_dict().

## Geotechnical Applications

1. **Data import** — Parse boring logs, lab results, SPT data from
   AGS4 files for use in analysis modules.
2. **Quality control** — Validate AGS4 files before importing data.
3. **Data exchange** — Read data from other engineers' deliverables.

## Common AGS4 Groups

- PROJ — Project information
- HOLE — Borehole/trial pit locations
- SAMP — Sample information
- GEOL — Field geological descriptions
- ISPT — Standard Penetration Tests
- IVAN — Vane shear tests
- TRIG — Triaxial tests
- CONS — Consolidation tests
- CONG — Consolidation general info
- GCHM — Geochemical tests

## Edge Cases

- **Encoding**: Some AGS4 files use Windows-1252 encoding, not UTF-8.
  The `encoding` parameter handles this.
- **Duplicate headers**: python-ags4 renames duplicates by default.
- **Empty groups**: Some files have GROUP headers with no data rows.
- **Row counting**: AGS4 dataframes include UNIT and TYPE rows at
  positions 0-1; actual data rows start at index 2.
