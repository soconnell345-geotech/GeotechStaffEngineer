# DM7Eqs — UFC 3-220-10 / 3-220-20 Equations

## Purpose
354 LLM-callable functions implementing equations from UFC 3-220-10
(Soil Mechanics, the UFC successor to NAVFAC DM 7.01) and UFC 3-220-20
(Foundations & Earth Structures, the UFC successor to NAVFAC DM 7.02),
organized by chapter. The package directories are still named `dm7_1`
and `dm7_2` for historical reasons; the actual digitized source is the
current UFC editions, not the 1986 NAVFAC manuals.

## References
- UFC 3-220-10 — Soil Mechanics (2022, with Change 1, 11 Mar 2025)
- UFC 3-220-20 — Foundations and Earth Structures (2025)

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
