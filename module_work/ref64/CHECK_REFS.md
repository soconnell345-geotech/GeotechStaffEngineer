# CHECK: REF-64 part (A), the refs half (local reference packs)

Branch `claude/ref-64-packs`, from `origin/main` at `8a09592`. Plan of record:
the app's `module_work/REFERENCE_PACKS_PLAN.md` (sections 1.1, 1.3, 2, 5), refs
rows of the table in 2.8. Nothing was merged, tagged or released, and no app
branch was made. All test data is made up (a fictitious "Example Practice
Manual" about a fictitious fill, zorblite). No real manual is used.

## What was built

| Piece | Where |
|---|---|
| The pack reader: `GEOTECH_REFERENCES_PACKS` (folders split by `os.pathsep`; a pack with `pack.json` or a folder of packs), read at call time; `scan()`, `packs()`, `pack_dirs()`, `pack_dir(id)`, `problems()`; the `local_` prefix; skip and name a built-in-id collision, a missing prefix, a duplicate id, an unreadable `pack.json`, a missing folder | `geotech_references/_packs.py:38-111` (scan at `:76`), `:114-134` |
| The pack's indexed files, and the key part (each indexed file's path, size and mtime; `""` with no valid pack) | `_packs.py:137-180` (`key_part` at `:163`) |
| `pack_records()` for the app's build record (`id`, `title`, `reference_id`, `edition`, `files`, a sha256 over the files' contents) | `_packs.py:183` |
| Text index walks the package then the packs | `geotech_references/_retrieval_db.py:147-160` (`out.extend(_packs.text_jsons())` at `:159`) |
| `keyed_db_path` covers the pack set, unchanged when unset | `_retrieval_db.py:116-133` |
| `pack_dirs` re-exported as the app's feature flag (plan 2.8: `hasattr(_retrieval_db, "pack_dirs")`) | `_retrieval_db.py:140-144` |
| Figure index walks the package then the packs; a pack's figures filed under its pack id | `geotech_references/_figures_db.py:58-69`, `:131-135` |
| The optional catalog fields `bbox` and `read_notes`: two columns, carried to `figure_get` only where a catalog gives them (an old database without the columns still reads) | `_figures_db.py:112-113`, `:153`, `:170-174`, `:320-336` |
| `resolve_pdf`: the pack's own folder first (by file name), then the existing chain (`GEOTECH_REFERENCES_DOCS`, repo-relative, `FileNotFoundError`) | `_figures_db.py:582-614` (pack step at `:600-604`) |
| File-based retrieval (`retrieve_section`, `load_chapter`, `list_chapters`, `search_sections`) finds a pack's text folder | `geotech_references/_retrieval.py:232-245` |
| One schema validator: `validate_schema` and its helpers moved to `geotech_references/_schema.py`; the audit script imports them back | `geotech_references/_schema.py:19-157`; `scripts/audit_chapter_text.py:149-163` |
| CLI `python -m geotech_references.packs new\|check\|list\|index` | `geotech_references/packs.py:507` (`main`) |
| `check`: pack.json (`:149`), text (`:181`), catalog and PDF (`:240`; page count, number printed on the page, `bbox` inside it, with PyMuPDF), print-fidelity warnings (`:209-222`), smoke test (`:345`), location refusal (`location_problem`, `:81`: inside the installed package, the refs source checkout, or a git work tree whose remote is `geotech-references`, `GeotechStaffEngineer` or `planlens`) | `packs.py:396` (`check_pack`) |
| `new` (`:423`), `list` (`:461`), `index [--rebuild]` (`main`) | `packs.py` |
| `scripts/build_figure_catalog.py --pack <dir> [--pdf FILE] [--sequential]`: same no-API List-of-Figures parse and caption search; descriptions from the pack's own text; `pdf_path` the file name only; an earlier catalog's `bbox` / `read_notes` kept | `scripts/build_figure_catalog.py:873` (`build_pack_catalog`), `:967` (`main`), `:304-310` (`_enrich_from_text(text_dir=)`) |
| `docs/REFERENCE_PACKS.md`: the format, the print-fidelity rules (plan 2.2 step 2), making and checking a pack, where packs live (plan 2.3's table, generic: Foundry, Databricks, Tiny Apps) | `docs/REFERENCE_PACKS.md` |
| README section pointing at the doc | `README.md`, "Local reference packs" |
| Tests | `tests/test_packs.py` (30 tests) |

## Unchanged when unset: the proof

1. **Iterators and key, by test** (`tests/test_packs.py::test_unset_is_unchanged`,
   parametrized over the setting absent, `""` and a lone `os.pathsep`):
   `_iter_text_jsons()`, `_iter_catalogs()`, `keyed_db_path(name)` for both
   database names and both `_db_path()`s equal the pre-REF-64 code's results,
   reimplemented in the test; `key_part()` is `""`. A setting naming only
   broken packs also keeps the key and the iterators
   (`test_folder_with_no_valid_pack_keeps_the_key`).
2. **Outputs, byte for byte.** With the setting unset, a script dumped
   `figure_get` for every built-in figure (3,736), `list_indexed_references`,
   `list_indexed_figures`, and `reference_search` / `figure_search` (limit 10)
   for nine queries ("pile setup", "passive earth pressure", "drilled shaft
   side resistance", "liquefaction", "wick drain smear zone", "bearing
   capacity factors", "Nordlund", "soil nail bond strength", "frost
   susceptibility"), once on unmodified `origin/main` and once on this
   branch. The two JSON files are identical (`cmp`).
3. **Old temp databases stay valid.** The text database's key input and
   layout are unchanged. The figure database gained two columns; a figure
   database built by the `origin/main` code (no `bbox` / `read_notes`
   columns) was read by the branch's `figure_get` without a rebuild and gave
   the same keys (`_row_to_full` reads the columns only when present).
4. **One validator, same results.** `scripts/audit_chapter_text.py` over all
   31 references (321 chapters): output identical before and after the move
   (`cmp`), exit 0, "AUDIT PASSED — 31 references, 0 errors".
   `validate_schema` on every built-in chapter: the same (empty) error lists
   before and after. `test_audit_script_uses_the_package_validator` pins that
   the script's `validate_schema` and `SECTION_ID` are the package's objects,
   and `test_validator_passes_every_builtin_chapter` runs it over every
   built-in chapter.

## Tests, with exit codes

Environment: Python 3.11, `pip install -e . pytest pytest-xdist pymupdf
matplotlib` (matplotlib was missing at first: unmodified main then had 4
collection errors in the DM7 chapter tests, all `No module named
'matplotlib'`; with it installed both runs are clean).

| Run | Result | Exit |
|---|---|---|
| `pytest -q -n 8 tests/test_packs.py` (branch) | 30 passed | 0 |
| `pytest -q -n 8` whole refs suite, branch | 6,720 passed, 1 xfailed | 0 |
| `pytest -q -n 8` whole refs suite, unmodified `origin/main` (own worktree and venv) | 6,690 passed, 1 xfailed | 0 |
| `python scripts/audit_chapter_text.py` (branch) | AUDIT PASSED, 31 references | 0 |
| App reference tests against this branch installed editable in the app's venv (app master `f063047`, Python 3.12, `pip install -e packages/aec-agent-core -e .`): `funhouse_agent/tests/test_reference_db_dispatch.py`, `test_figure_db_adapter.py`, `test_read_reference_figure.py`, `test_reference_adapters.py`, `test_reference_chart.py`, `test_reference_docs.py`, `test_reference_example_ids.py`, `test_reference_result_sizes.py`, `test_fhwa_soils_foundations_reference.py`, `test_vision_locations.py`, `webapp/tests/test_diagnostics.py`, `webapp/tests/test_review_boundary.py` | 536 passed | 0 |
| The same app tests against refs `origin/main` | 536 passed | 0 |

6,720 - 6,690 = 30, the new tests. No failure needed excusing.

`tests/test_packs.py` covers: with the setting, the pack listed in both
inventories; a made-up term found only in it; `reference_get` with
`children` and `parent_id`; `retrieve_section` on the pack; `figure_search`
and `figure_get` (with `bbox` / `read_notes` on the figure that has them and
not on the one without, and not on a built-in); `resolve_pdf` from the pack
folder and falling back to `GEOTECH_REFERENCES_DOCS`; unset equals today; the
key changes with a pack edit (content, and a same-size copy with a new mtime);
a pack folder and a folder of packs; a collision, a missing prefix, a
duplicate id and a missing folder skipped and named; `check` passing the good
pack (with the smoke test) and catching each error class (pack.json keys,
prefix, id vs folder, unreadable JSON, `pdf_files`; schema error, reference
id mismatch, repeated chapter, no chapters; catalog repeated number,
`read_notes` > 500, negative page, folder in `pdf_path`, malformed `bbox`;
with PyMuPDF a `bbox` outside the page, a page past the PDF, a number not
printed on its page and `page_estimated` silencing it; inside the package;
inside a git work tree with a public remote, and not with a private one);
print-fidelity warnings; `new`, `list`, `python -m` exit codes;
`build_figure_catalog.py --pack` on a made-up three-figure PDF;
`pack_records()`; no `pack.json` tracked in this repo.

A manual run of the app's own adapters with a fixture pack set
(`GEOTECH_REFERENCES_PACKS`): `reference_db.reference_search("zorblite
settlement")` returned the pack's sections, `reference_get("local_example_manual",
"1.2")` its section, and `figure_db.figure_get` its figure with `bbox` and
`read_notes`. No app change was needed for that.

## What the app half needs (plan 2.8 app rows; after step 11 (a) merges)

1. **`funhouse_agent/reference_vision.py`** (on master still in
   `vision_tools.py`): after `figure_get`, when the record has `bbox`, pass it
   to `vision_view.render_view(path, page=..., bbox=rec["bbox"])` (PDF points,
   top-left origin, `render_region`'s frame); when it has `read_notes`, append
   them to the read prompt under a fixed heading ("Notes from the reference's
   catalog on reading this figure"). Built-in figures carry neither, so their
   render and prompt stay byte-identical. Tests: a fake vision engine sees the
   crop and the notes; a built-in figure's prompt pinned unchanged.
2. **The packs line** in the librarian's prompt and `_REFERENCES_DESCRIPTION`
   (`deep/agent.py`), only when packs are present: feature-detect with
   `hasattr(geotech_references._retrieval_db, "pack_dirs")` (refs 1.5.0 has
   no such name), then build "This deployment adds: <title> (`local_x`), ..."
   from `geotech_references._packs.pack_records()` (`id`, `title`). No packs,
   no line, no fingerprint change (the fingerprint builds clear every
   `GEOTECH_*`).
3. **`reference_packs` in the build record**: after `record_build`, set
   `agent.geotech_build["reference_packs"] = _packs.pack_records()` when it is
   not empty (`[{id, title, reference_id, edition, files, sha}]`); absent
   otherwise.
4. **Diagnostics** (`webapp/geotech_page.py`, the reference-PDF row): read
   catalogs through `_figures_db._iter_catalogs()` (it now includes pack
   catalogs) and show `_packs.problems()` so a skipped pack is visible.
5. **The PDF fallback** already works: `resolve_pdf` raises
   `FileNotFoundError` when a pack PDF is in neither the pack nor
   `GEOTECH_REFERENCES_DOCS`, and the app's reader then asks the host's
   fetcher by file name (catalog `pdf_path` is the file name only).
6. **Guard test**: no `pack.json` tracked in the app repo (`git ls-files`),
   beside `test_public_text.py` (the refs repo now has its own:
   `test_no_pack_is_tracked_in_this_repo`).
7. **Docs**: `FOUNDRY.md` §3d, `webapp/README.md`, `DATABRICKS_INSTALL.md`
   (point at the refs `docs/REFERENCE_PACKS.md` host table).
8. **Pin**: `geotech-references>=1.6.0` once the owner releases it.

## Noted, not fixed

- The app's `reference_db_adapter.resolve_reference` resolves an id or a
  printed designation, not a full title: "Example Practice Manual" is refused
  with `did_you_mean: ["local_example_manual", ...]`. Built-ins behave the same
  ("Soil Nail Walls Reference Manual" is refused too), so it is not
  pack-specific; the planned packs line names the id.
- `agents/references_agent.py` lists packages with Python modules; a pack has
  none, so packs are not in that catalog (they are searched through the
  databases, as planned).
- The figure database has no schema version: a catalog edited to add `bbox`
  in a source checkout is picked up because the catalog's mtime triggers a
  rebuild, which is enough.
- The version in `pyproject.toml` is still 1.5.0; the bump is for the release.

## Review fixes (branch `merge/ref64`, 2026-11-09)

An independent review (verdict: merge after fixes) found two gaps with packs
set and one for hosts with only the wheel. Fixed as separate commits on
`merge/ref64`, before the 1.6.0 release commit; nothing merged, pushed or
tagged.

| Commit | Fix |
|---|---|
| `4cbab24` | **A pack's PDF paths stay inside the pack.** `pdf_files` and a catalog's `pdf_path` take a bare file name only (`_packs.bare_file_name`: no separator, `..` or drive). `check` refuses any other; the figure index skips such a catalog and `_packs.problems()` names it, so `figure_get` never returns a host path for a pack figure. For a pack, `resolve_pdf` looks in the pack folder, then `GEOTECH_REFERENCES_DOCS`, by that name, then raises `FileNotFoundError`; it never takes the repo-relative step (row "resolve_pdf" above is superseded). Before, an absolute or `../../` `pdf_path` resolved to a file outside the pack when `GEOTECH_REFERENCES_DOCS` was unset. `build_figure_catalog.py --pack` takes only a bare name from `pdf_files`. |
| `098a3cc` | **A bad pack file cannot break search.** Before, one list where text belongs (a chapter's `applicability`, `chapter_title`, `summary` or `body`; a catalog's `caption` or `description`) failed the whole index build, built-in queries included, and `check` crashed with a traceback on some. Now each pack file goes through the same type checks the indexes use (`_packs.chapter_problem` / `catalog_problem`); a file that fails is skipped and named in `problems()`. Each pack file is indexed under its own SAVEPOINT (`_packs.index_pack_file`), so one that fails anyway is rolled back and named while the built-ins and other packs still index. `check` type-checks `pack.json`, every chapter field at every level and every catalog field and reports a wrong kind as an `ERROR` line; it guards the shared schema check, the PDF open and the smoke test, and the smoke test reports any pack file left out. |
| `ba61080` | **What a host with only the wheel needs.** `chapter_schema.json` moved from `scripts/` into the package as package data; `python -m geotech_references.packs schema` prints it byte for byte. The README links `docs/REFERENCE_PACKS.md` by its absolute GitHub URL on main (works on PyPI), and the CLI help ends with it. The doc gained a minimal made-up chapter file (one chapter, two sections, `[p. N]` markers, one equation with `source`), which a test runs through `check`. |

The built-ins' index path is the same INSERTs, moved into `_index_chapter` /
`_index_catalog`. With the setting unset, the review's unset probe (keys,
inventories, 16 text and 16 figure queries plain and fitted, `reference_get`,
`reference_query`, `figure_get` on all 3,723 built-in figures,
`resolve_pdf`, the file-based layer) is byte-identical to `origin/main`'s
dump after the fixes (`cmp`, 3,337,616 bytes).

Tests: `tests/test_packs.py` grew from 30 to 64; the re-runs
after the fixes are below.

| Run (release commit `9e01d21`; single process, foreground) | Result | Exit |
|---|---|---|
| `pytest -q tests/test_packs.py` | 64 passed | 0 |
| Whole refs suite in two chunks (files 1-40, 41-115) | 3,218 passed, 1 skipped (POSIX file modes, on Windows), 1 xfailed; 3,535 passed. Together 6,753 passed of 6,755 collected | 0, 0 |
| `scripts/audit_chapter_text.py` | AUDIT PASSED, 31 references; output identical to before the fixes | 0 |
| Fresh venv (Python 3.14), wheel built from `git archive HEAD`, run outside a checkout: `new`; `check` on the good pack (0) and on 24 bad copies (each 1 with an ERROR line; no traceback in the whole run); the location refusals; `list`; `index [--rebuild]`; `schema` (identical to the wheel's file, valid JSON); a bad pack beside a good one (built-in and pack search work, both bad files skipped and named, the `resolve_pdf` message carries no host path) | as expected | 0 |
| App reference tests (the review's 30 files plus `webapp/tests/test_page_fingerprints.py`, app master `7f89ef4`, this branch on PYTHONPATH) | 1,090 passed, 3 skipped (live tests) | 0 |
| Unset probe, branch vs `origin/main` | byte-identical | 0 |

Commits on `merge/ref64` after the review: `4cbab24`, `098a3cc`, `ba61080`
(the fixes), `2f260de` (this file leaves the refs repo), `9e01d21` (1.6.0:
the version, `CHANGELOG.md` from the draft below plus the fixes, CLAUDE.md's
test counts). Not tagged, pushed or merged.

This file then left the refs branch (its last commit before the release
commit); this copy is filed in the app's `module_work/`.

## CHANGELOG draft: geotech-references 1.6.0

(The entry released in the refs `CHANGELOG.md` adds the review fixes above.)

> **Local reference packs (REF-64).** A deployment can add its own references
> in the library's structured format without putting them in the public
> repository or the wheel. Set `GEOTECH_REFERENCES_PACKS` to a pack folder or
> a folder of packs (several separated by `os.pathsep`); each pack holds
> `pack.json`, `text/chapterNN.json` (the chapter schema the built-ins use)
> and optionally `figures_catalog.json` and its PDF. Pack ids start with
> `local_`. `reference_search`, `reference_get`, `reference_query`,
> `list_indexed_references`, `figure_search`, `figure_get`, `resolve_pdf`
> (which looks in the pack folder first) and the file-based `retrieve_section`
> family cover packs beside the built-in references, with the same ranking and
> synonym expansion. The search databases are keyed by the pack set as well as
> the package copy; with the setting unset every key, every database and every
> result is unchanged. A figure catalog may give any figure two optional
> fields, `bbox` (the chart's box on the page, PDF points, top-left origin)
> and `read_notes` (at most 500 characters on how to read it), which
> `figure_get` returns where given. New: `python -m geotech_references.packs
> new|check|list|index` (`check` exits 0 or 1: schema, catalog, pages and
> boxes against the PDF with PyMuPDF, print-fidelity warnings, a search smoke
> test; it refuses a pack inside the package or a public repository's work
> tree), `scripts/build_figure_catalog.py --pack`, `_packs.pack_records()` for
> build records, and `docs/REFERENCE_PACKS.md`. The chapter schema check moved
> from `scripts/audit_chapter_text.py` into `geotech_references/_schema.py`
> (the script imports it; results unchanged).
