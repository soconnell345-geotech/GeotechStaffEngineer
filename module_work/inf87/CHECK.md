# INF-87: test files that leave the process changed (CHECK)

Branch `claude/inf-87-test-env`, from `claude/b13-integration` at f150fab. Test files only; nothing under `aec_document_review/` or `aec_agent_core/` changed. Venv (a) of `COMMON_DOCREVIEW.md`: Python 3.12.3, pytest 9.2.0, planlens 0.16.0 from PyPI. Every command ran from `packages/aec-document-review` as `MPLBACKEND=Agg python -m pytest -q -p no:cacheprovider ...`, judged by exit code.

## Reproduced before any change (at f150fab)

1. `pytest tests/test_foundry_host.py tests/test_core.py`: **3 failed, 92 passed** (`tests/test_core.py` alone: 86 passed).
   - `test_core.py::test_resolve_engine_no_engine` (`assert 'No engine ...`)
   - `test_core.py::test_resolve_engine_with_key_is_anthropic_or_error` (`assert 'none' in ('anthropic', 'error')`)
   - `test_core.py::test_model_choices_data_and_labels` (`IndexError: list index out of range` at `ids[0]`)
   - Why: `GEOTECH_DEPLOYMENT=foundry` and `GEOTECH_PLOT_CODE` were still set after the first file. Its `_clean` fixture deleted each name with `monkeypatch.delenv(..., raising=False)` (no undo is recorded for a name that is absent) and restored by hand after `yield`. Two tests call the entry, which writes `os.environ` directly, and then `monkeypatch.setenv` the same name (`foundry_entry.main()` then `setenv("GEOTECH_DEPLOYMENT", "made-up-host")`; `mark_deployment()` then `setenv("GEOTECH_PLOT_CODE", "1")`). That `setenv` records the entry's value as "the value before". The fixture's hand restore runs first, monkeypatch's undo runs after it and puts the entry's value back.
2. `pytest "tests/test_review_binding_offline.py::test_the_process_host_is_the_apps"`: **1 failed** (the whole file alone: 1 failed, 6 passed).
   - `AssertionError: assert 'Document Review' in ('AEC agent', '(AI draft)')` on `host.own_mark_words`.
   - Why: the process host is made when `aec_agent_core.document_tools` is first imported, and takes its own-mark words from the application named by then. The file imported only the core; the name is handed in by `aec_document_review/__init__.py`, which another file had imported first in the whole-folder order.
3. Found by running each file alone: `pytest tests/test_review_page_vocabulary_offline.py`: **3 failed, 3 passed**.
   - `test_review_page_names_no_geotech_report_things[legacy]`, `[minimal]`, `[legacy+switches]`: `NoPageHost: The review suite has no page to run its tasks on`.
   - Why: the hook `review_eval.page_host` is filled when `aec_document_review.review_eval` is imported; the test imported only the core's runner. This is the "laptop-only" failure CLAUDE.md lists under "Tests and environment": it is this pattern, not the laptop.

## What changed (one commit each)

- 17982ec `tests/test_foundry_host.py`, fixture `_clean`: for each name it now does `monkeypatch.setenv(name, "")` then `monkeypatch.delenv(name)`, so monkeypatch holds a first record with the true earlier value; the first record for a name is the last one undone. The hand restore is gone (monkeypatch now does all of it). No test body or assertion changed.
- 355322f `tests/test_review_binding_offline.py`: `import aec_document_review` before `from aec_agent_core import document_tools`.
- f5a89e2 `tests/test_review_page_vocabulary_offline.py`: `import aec_document_review.review_eval` at the top of the parametrised test, before the core's runner.

No assertion weakened, no test skipped or deleted. No fingerprint, prompt, setting or tool description moved: `tests/test_boundary.py tests/test_review_fingerprints.py` 9 passed. `tests/test_record_masks_fx11_page.py` untouched.

## Runs after the change

| Run | Before | After |
|---|---|---|
| `pytest tests` (default order) | 1284 passed, 18 skipped | 1284 passed, 18 skipped (run twice) |
| `pytest $(ls tests/test_*.py \| sort -r)` | 1284 passed, 18 skipped (after repairs 1 and 2) | 1284 passed, 18 skipped (after all three) |
| `test_foundry_host.py` alone / `test_core.py` alone | 9 passed / 86 passed | 9 passed / 86 passed |
| `test_foundry_host.py test_core.py`, and the other order | 3 failed, 92 passed / not run | 95 passed / 95 passed |
| `test_review_binding_offline.py` alone; its one test alone | 1 failed, 6 passed; 1 failed | 7 passed; 1 passed |
| binding with `test_core.py` and with `test_foundry_host.py`, both orders | not run | 93, 93, 16, 16 passed |
| `test_review_page_vocabulary_offline.py` alone | 3 failed, 3 passed | 6 passed |
| each of the 97 `tests/test_*.py` alone | 96 exit 0, 1 failed (vocabulary; run after repairs 1 and 2) | vocabulary re-run: 6 passed |

The reverse-order run was not made at f150fab itself, so "nothing fails in reverse order" is known only with the repairs in.

**The core package** (`packages/aec-agent-core`, same venv, no change made): each of its 78 test files alone exits 0; `pytest tests` 1150 passed, 1 skipped; reverse order 1150 passed, 1 skipped.

**The environment after each test.** A scratch plugin (not committed) compared `os.environ` before and after every test of both folders, with the repairs in. Left changed: `GEOTECH_WEBAPP_DATA` once per run (the session fixture in `conftest.py`, on purpose) and `LD_LIBRARY_PATH` once (in `test_review_binding_offline.py`; not traced, probably a library on first import). Nothing else, so no other file leaves the environment changed in the default order.

## Fail alone or in reverse order and NOT fixed

None. Read but not changed, because no run shows a failure: the other test files that write `os.environ` by hand (`conftest.py`, `page_builds.py`, `test_review_harness_offline.py`, `test_review_method.py`, `test_tinyapps_entry.py`, `test_tinyapps_keys.py`, `test_tinyapps_wiring.py`, `test_workdir_durability.py`; in the core `test_ev49_image_ceiling_between_arms.py`). `test_tinyapps_entry.py` is the nearest in shape to repair 1 (its entry also does `os.environ.setdefault("GEOTECH_DEPLOYMENT", ...)`); the plugin shows it leaves nothing behind today.

## Findings about the source (not changed)

- `aec_agent_core/document_tools.py`: `BASE_HOST = _review.configure(app_host())` runs at import and fixes `own_mark_words` from the application named at that moment. A host that imports `aec_agent_core.document_tools` (or anything that imports it) before the application's package gets `('AEC agent', '(AI draft)')` for the life of the process, so the app's own earlier marks (titled "Document Review ...") would not be recognised as its own. Made-up input: `python -c "from aec_agent_core import document_tools; import aec_document_review; from planlens import review; print(review.current().own_mark_words)"`. The page itself imports the package first, so testers do not meet it; it is one process-wide value, not per tester.
- `foundry_entry.mark_deployment` and `tinyapps_entry` write `os.environ` with `setdefault`: process-wide by design, the same for every tester in the process. Not a defect; it is why a test around them needs monkeypatch to hold the first record.

## Not checked

- Python 3.11 (CI's `document-review` job runs 3.11 and 3.12; only 3.12 here). GSE's own suite (`webapp/`, `funhouse_agent/`), which no change here reaches. Pairs beyond those in the table: pairs in general are covered only by the default and reverse orders and by each file alone.
- Random orders, and pytest-xdist splits, were not run.

**Calls made:** repair 3 was not named in the item; it is the item's second pattern and a run showed it. **Release note:** None. **For the lead:** CLAUDE.md's "laptop-only failures" line about the vocabulary file's three cases can go once this merges; INF-87 can close. **Noted, not fixed:** none beyond the findings above.
