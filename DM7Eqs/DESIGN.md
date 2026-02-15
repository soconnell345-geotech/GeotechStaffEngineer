# DM7Eqs — NAVFAC Design Manual 7 Equations

## Purpose
354 LLM-callable functions implementing equations from NAVFAC DM7.01
(Soil Mechanics) and DM7.02 (Foundations & Earth Structures), organized
by chapter.

## References
- NAVFAC DM 7.01 — Soil Mechanics (1986)
- NAVFAC DM 7.02 — Foundations and Earth Structures (1986)

## Structure
```
DM7Eqs/
  geotech/
    dm7_1/          # DM7.01 chapters
      chapter1.py through chapter11.py
    dm7_2/          # DM7.02 chapters
      chapter1.py through chapter6.py
  tests/            # 1852 tests
```

## Foundry Agent (dm7_agent_foundry.py)
- Auto-discovers 354 functions via `inspect` module at import time
- Builds METHOD_REGISTRY and METHOD_INFO from function signatures + numpy-style docstrings
- Handles name collisions across chapters with chapter prefix
- Excludes Callable-parameter functions (ch7 probability)
- Import path: `sys.path.insert(0, "DM7Eqs")` then `from geotech.dm7_1.chapterX import ...`

## Key Notes
- 365 total functions, 354 LLM-callable (11 excluded for Callable params)
- 1852 tests across all chapters
- Functions are standalone — no class dependencies, plain float I/O
