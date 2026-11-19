<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/fx-pl-b14` (planlens keeps no review file on its main); the branch was then deleted. -->

# FIX_B14: branch `claude/fx-pl-b14`

From `origin/claude/b13-integration` at `a99192c`. Three commits and this
report. Nothing was pushed to `main`, a `ci/...` branch or a tag; `git stash`
was not used.

## 1. `a8609f6` PL-09: the column warning is an option of `Document`, off by default

- `planlens/document/document.py`: `Document(..., column_warning: bool = False)`,
  beside `continued_tables` and documented the same way in the class
  docstring. `_read_page` calls `order_runs_across_columns` only when it is
  set. Set, the page is what it was at `a99192c`.
- `planlens/document/reading_order.py`: the module docstring only. The
  functions (`order_runs_across_columns`, `column_order`,
  `across_columns_warning`) are unchanged.
- **Pass-through: there is none, so none was added.** `ReviewToolkit.open`
  builds its documents as `Document(content=..., name=..., text_source=...)`
  (`planlens/tools/toolkit.py`), and `planlens.review` builds its own with
  `content` or `filepath` alone (`frames.py`, `find_like.py`,
  `markup_check.py`). Neither `continued_tables` nor
  `captions_on_every_page` is named anywhere in `planlens/tools` or
  `planlens/review`. So no tool or host can turn the warning on today; only
  a caller that builds a `Document` itself can.
- **Proof that unset is `6196f13`.** `to_dict()` of every page of the five
  suite fixtures (review document, report, submittal, sheet set, schedule
  set: 49 pages) and of two made-up pages of columns drawn row by row (2 and
  3 columns), each read with words on and off and tables on and off, 204
  reads in all, serialised to JSON:

  | checkout | sha256 of the 204 reads |
  |---|---|
  | `6196f13` | `60e4e2dd…dcd245a` |
  | `a99192c` | `0efe639f…181384e` |
  | this branch, option unset | `60e4e2dd…dcd245a` (byte-identical to `6196f13`) |
  | this branch, `column_warning=True` | `0efe639f…181384e` (byte-identical to `a99192c`) |

  The suite fixtures alone (196 reads) are the same at all three commits,
  since none of them carries the warning; the two column pages are what
  tell `6196f13` from `a99192c`. The PDFs were built once and read by each
  checkout (a `git worktree` per commit, one venv).
- Tests.
  - New file `planlens/document/tests/test_pl09_column_warning_option.py`
    (4 tests): columns drawn row by row carry no warning with the option
    unset, and the warning is the whole difference from the page with it
    set (2 and 3 columns); the page a tool renders has no note; the check
    is not called unless asked. All 4 fail on `a99192c`.
  - `test_pl09_reading_order.py` and `test_rv_pl09_columns_not_tables.py`:
    every `Document` opened to look for the warning **or for its absence**
    now sets the option (7 places and 1). The "says nothing" tests set it
    too, or they would pass without testing anything. No assertion was
    changed.
- Notes: `CHANGELOG.md` (the PL-09 entry) and `planlens/document/DESIGN.md`
  (three places) say it is an option, off by default, and why.

## 2. `be5c045` PL-37 review: the openai style compared with the plain style

`planlens/tools/tests/test_sheet_tools.py`,
`test_the_sheet_tools_are_offered_only_when_asked_for`: for each tool of the
asking toolkit, `asked.specs("openai")[i]["function"] == asked.specs("plain")[i]`,
the type is `"function"`, and the two lists are the same length. **It passes
on today's code: no finding, no code change.**

## 3. `3d9ed3f` Notes: what the review commits added

Words only (`CHANGELOG.md`, `planlens/document/DESIGN.md`,
`planlens/ir/DESIGN.md`, `planlens/review/DESIGN.md`), each sentence checked
against the code.

- **`drawing_geometry` (DR-44):** `value_offset_pt` / `text_offset_pt` (the
  finder's `text_distance`, 0.1 pt, only on a row that has a text),
  `TIED_TEXT_NOTE`, no threshold; `NONE_FOUND_NOTE` naming each kind asked
  for that came back empty.
- **The DXF plot (DR-19):** `MAX_FILE_MB` 16, `MAX_ENTITIES` 50,000,
  `MAX_SECONDS` 30 (`PlotTooLarge`); `MIN_SIDE_MM` 297 less the margins;
  `PlotUnavailable` naming Pillow; `dxf_plots` as a callable asked at each
  open, the plot kept under its own key, a host that did not ask never
  handed one. **One sentence was wrong, not just missing:**
  `planlens/review/DESIGN.md` still said the conversation's toolkit "is
  built with `dxf_plots` on"; it is given a callable.
- **PL-09:** the column check leaves out a row of dot leaders (a word has a
  letter) and counts no line without a small letter; the continuation rule
  leaves unlinked two borings' logs on one form, tables of fewer than two
  columns, pages of different widths and pages with a line that has no box.
- **PL-18:** an array met after the budget is read as its first cell, in
  the ingest and in `planlens.dxf.truth`, and the warning counts the copies
  that arrived (`planlens/ir/DESIGN.md` already had most of this; the
  changelog had none); an ATTRIB whose extrusion differs from its
  reference's is stepped through the WCS (in neither file before).

## Commands run

Python 3.12, `pip install -e ".[mcp]" pytest` (planlens' CI install; ezdxf
1.4.5, PyMuPDF 2.1), on the final code head `3d9ed3f`:

| command | install | result |
|---|---|---|
| `pytest planlens -q -p no:cacheprovider` | without Pillow | **2529 passed, 15 skipped**, exit 0 (403 s) |
| `pytest planlens -q -p no:cacheprovider` | with Pillow 13.0.0 | **2620 passed, 5 skipped**, exit 0 (428 s) |

The 91 more passes with Pillow are the tests that make a DXF plot; without
it they show as 10 more skips because a file skipped at import counts once.

Also run while working:

- the six PL-09 test files after task 1: 42 tests, 2 failed on the first
  run (my new test compared `to_dict()` wrongly: an empty `warnings` is
  left out of it), then 24 passed on the three files touched;
- `test_pl09_column_warning_option.py` against `a99192c`: 4 failed;
- `planlens/tools/tests/test_sheet_tools.py` after task 2: 19 passed.

## Found and not fixed

- **Nothing in the app reads the warning or the option.** A search of the
  GeotechStaffEngineer checkout for the warning's words, `column_warning`
  and `order_runs_across_columns` finds nothing. Document Review's tests
  were not run against this branch: no tool specification changed, and the
  tool results are those of the released version.
- **The column check is still unmeasured.** It is now off, but it has still
  not been run over real pages or put in front of a model, and no host can
  switch it on without a pass-through (see task 1).
- **Pillow is still not a dependency**, so planlens' CI install does not run
  the tests that make a DXF plot (the difference in the table above).
- `continued_tables` and `captions_on_every_page` have no pass-through
  either: both are reachable only by a caller that builds its own
  `Document` or calls `document_outline` itself.
