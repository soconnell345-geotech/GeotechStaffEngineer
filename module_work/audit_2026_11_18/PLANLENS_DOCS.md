<!-- Filed here by the lead on 2026-11-18 from planlens' branch `claude/docs-audit` (planlens keeps no review file on its main). Its twelve corrections are merged to planlens' main on 2026-11-18 after a green run of the whole suite on GitHub (after 0.17.0, not released, no tag). What it left open is the app's BACKLOG DOC-25. -->

About 780 statements checked, 12 false, 12 corrected here

The count of statements is an estimate (README about 150, the DESIGN files and CHANGELOG about 300, tool descriptions, exported docstrings and the pyproject description about 320, CLAUDE.md 12). Checked on `main` at f9b5b38 against the code, and against the package index (latest 0.17.0; 19 versions, 0.1.0 to 0.17.0).

## False statements

| Document | The statement | What is true | |
|---|---|---|---|
| `CLAUDE.md` | open work is in the app's `BACKLOG.md`, "§5 (PL-nn)" | section 5 is the references; planlens is "## 8. planlens" | corrected |
| `README.md`, 0.17.0 line | "three `Document` options no tool passes yet" | true of 0.17.0 only: on `main` (not released) `ReviewToolkit` and `ReviewHost(toolkit_options=...)` pass them; `captions_on_every_page` is the outline's option (`document/roles.py`), not `Document`'s | corrected; the line says which is released |
| `README.md`, confidence ladder | 0.5 is "the **default** call threshold" | every finder's `min_confidence` defaults to 0.0 (`ir/queries.py`); 0.5 is a convention | corrected in the table; "at the 0.5 default" beside corpus figures (about lines 330-455) left: those lines hold published numbers |
| `README.md`, MCP flags | `--image-budget` takes five names | six: `gpt-5.2-high` was missing (`planlens.document.BUDGETS`) | corrected |
| `planlens/document/DESIGN.md` | "Since PL-11 (unreleased)" | released in 0.15.0 | corrected |
| `planlens/document/DESIGN.md` | "Sheet index and cross-references (unreleased; ...)" | released in 0.16.0 | corrected |
| `planlens/document/DESIGN.md` | "Sheet numbers as addresses (unreleased, ...)" | released in 0.17.0 | corrected |
| `planlens/document/DESIGN.md` | "... one narrowed in PL-43, unreleased" | released in 0.17.0 | corrected |
| `planlens/ir/DESIGN.md` | `raster.py` "isolated so cv2 stays optional" | OpenCV is a core dependency since 0.4.0 (`pyproject.toml`) | corrected |
| `planlens/ir/DESIGN.md` | `from_raster` delegates to `drawing_ir.raster.trace_raster` | the module is `planlens.ir.raster`; there is no `drawing_ir` | corrected |
| `planlens/pdf/DESIGN.md` | PyMuPDF (`fitz`) is "optional, same pattern as ezdxf in dxf_import" | a core dependency (`PyMuPDF>=1.27.2`), imported as `pymupdf`; planlens has no `dxf_import` | corrected |
| `planlens/pdf/__init__.py` docstring | "Requires: PyMuPDF >= 1.23 (optional dependency)" | `PyMuPDF>=1.27.2`, core | corrected |

Found true, nothing changed: `CHANGELOG.md` ("Unreleased" and 0.17.0 in full, every version heading), `planlens/review/DESIGN.md`, the `pyproject.toml` description (word for word the index's summary), every tool description in `planlens/tools/specs.py` (21 tools run, the sheet and drawing tools on) and `planlens/review/specs.py` (16 run through `review.call`), and the docstrings of what `planlens` and `planlens.review` export. No source line changed but the one docstring above; no tool description changed, so no pinned specification moved.

## The items

1. **Unreleased work.** `v0.17.0` is 7eed79b. `CHANGELOG.md` holds PL-09, PL-37's printed scale and VIS-30 (said to be not wired) under `## Unreleased`, above `## 0.17.0 — 2026-11-18`; every name, default and field it gives is in the code. `README.md` and `CLAUDE.md` were unchanged since 7eed79b and named nothing unreleased; README now names the reading options once, as "on `main`, not released". Every version either file names is on the index ("0.11" stands for 0.11.0).
2. **Changes left by the PL-09 review.** README's 0.17.0 line (first README row above). `planlens/review/DESIGN.md` gains one sentence: one file read with and without a reading option is two open documents, each counted under `max_open` (8), and a document closed under the limit is reopened from its source at its host's next call under the same handle (`review/tests/test_rv_b15_pl09_host_options.py` proves it). `REVIEW_B15_PL09.md` is not at the root on `main` (item 4); it was read from 5a4c5b9, and the README note from `CHECK_B15_PL09.md` at bc9c986.
3. **CLAUDE.md.** One false statement (the section number), corrected. It gives the test command (`pytest planlens -q`, judged by exit code) and can be followed in order. Not false, so left for the owner:
   - It says releases are owner-gated and that a `v*` tag publishes; it does not say "never create a tag", never push to `main`, or never push a `ci/...` branch.
   - It gives no install step; CI's is `pip install -e ".[mcp]"`, then `pytest "Pillow>=8"`.
   - It says nothing of one session at a time or of the usage allowance.
   - The published-numbers script is in the app's repository, which a session here must clone to see.
4. **Finished checks and reviews at the root.** None: `main` has no `CHECK_*.md` or `REVIEW_*.md`. The three merge commits left them out. In history:
   - `CHECK_B15_PL09.md` (bc9c986) and `REVIEW_B15_PL09.md` (5a4c5b9): the reading options, PL-09, merged in 7428ecc.
   - `CHECK_B15_PL37.md` (95aed91) and `REVIEW_B15_PL37.md` (d85d353): the printed scale in the sheet index, PL-37, merged in 85fe5e5.
   - `CHECK_B15_VIS30.md` (3ed9892) and `REVIEW_B15_VIS30.md` (35132b5): region tiles, VIS-30 first slice, merged in f9b5b38.
   - `CHECK_PL38.md` (added 0031bd4, removed 3038b9f): the earlier PL-37/38 work.

## Cannot tell

- Four changelog dates are one day before the index's upload time in UTC (0.6.0, 0.11.0, 0.14.0, 0.17.0; all uploaded between 00:15 and 02:31 UTC). They read as local dates; the other 15 match.
- Every corpus figure in README and the DESIGN files: `doc_claims_check.py` and the corpus were not run.
- "(owner's rule, 2026-10-12)" in CLAUDE.md; the Python 3.14 wheel note, the size of the GUI OpenCV, and the headless OCR recipe in README.
- `planlens/pdf/DESIGN.md`'s adapter pipeline (`to_dxf_parse_result`, `build_slope_geometry`, `build_fem_inputs`): those functions are the app's; the document's own status note says this part predates the split.
- Tool descriptions: `find_like` "in seconds, at any scale and rotation" (one run); marks opening in desktop PDF viewers; image inputs; Word and Excel; `log_grid` on a scan; the "it misses" lists of the two checks.

## Noted, not changed (not false)

- The toolkit's `annotate_document` schema requires `markups`, and the handler defaults it to none.
- `measure`'s `kind` takes `text`; its description does not say so.
- `sheet_index`'s result `note` names sheet, title and rev as printed, not the scale.
- A DXF result in the host layer names `annotate_dxf` and `save_file`, which planlens does not serve.
- A place name stands for the drawing corpus 8 times across the public `.md` files; whether the public-wording rule covers it is the owner's call.

## Not checked

- `planlens/document/DESIGN.md` in prose beyond about 250 of its 2,275 lines: its backticked names, paths and constants were checked by script (483 tokens), its thresholds and rules were not compared with the code. `planlens/ir/DESIGN.md` lines 441-535 likewise.
- CHANGELOG entries below 0.17.0, beyond each heading's version and date.
- README's dated history of measurement rounds, and the `ir` architecture prose beyond the existence of its names.
- `sheet_index` on a sheet that prints a scale was read in code, not run; the MCP server serving the tools; the OCR extra; a host with FIPS; docstrings outside the two `__all__` lists.
- The app's tests against this branch: no specification or result changed.

## Tests

`pytest planlens -q -p no:cacheprovider -n 4` (CI's install plus pytest-xdist; Python 3.12, PyMuPDF 2.1, OpenCV 5.1): 2746 passed, 5 skipped, exit 0. On `main` before the changes, without Pillow: 2644 passed, 16 skipped.
