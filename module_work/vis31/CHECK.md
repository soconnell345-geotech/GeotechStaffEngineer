# VIS-31 part 1 and VIS-30 part 1 (task E): CHECK

Branch `claude/vis-31-one-image-readings`. Both pieces are repairs; no setting, no default to list.

## Confirmed on today's code
- `vision_tools.ask_vision` added only the bound user question (`turn_question.with_question`); the tile prompts (`_tile_jobs`, `_image_tile_jobs`) say "Report only what is IN this tile"; nothing said a reading cannot compare with another image.
- `_read_pdf_page` did `out.pop("legibility", None)` whenever tiles were read. The image-file path (`_read_image_file`/`_image_tile_jobs`) never carried a legibility note, so no other tiled path drops it.

## Changed
- `turn_question.py`: `ONE_IMAGE_NOTE` and `with_one_image(prompt, kind)` (kinds in `QUESTION_KINDS` only).
- `vision_tools.ask_vision`: applies it once, before the user question, with or without a bound question (so every page, overview, tile, region, image, chart and worked-example read gets it).
- `vision_tools._tile_lettering_px`, `_tile_legibility`; `_tile_jobs` rows carry a private `_lettering_px` (measured `text_px`, else `ASSUMED_CAD_LETTERING_PT` at the tile's scale when the page has under 40 text chars, as `tile_grid` does), popped before the result leaves. The `legibility` note stays while any tile is under `LEGIBLE_TEXT_PX`, names those tiles, and says to zoom with `render_region`; dropped only when all tiles read.

Example: a 36x48 in sheet with 4.3 pt lettering, `tiles=4`: before, no `legibility` key; after, "…still under 12 px tall in tile(s) r1c1 (~9 px)…, zoom with render_region". A tile asked "what changed against issue B?" now has the one-image note and is told to say the comparison needs the other image.

## Tests (venv a, `packages/aec-agent-core/tests/`)
- `test_vis31_one_image.py` (all fail before the change except the two `test_code_reads_are_not` cases, which pass without it by design: those kinds are left unchanged), `test_vis30_tile_legibility.py` (the note and assumed-px tests fail before; "dropped when every tile reads" passes on master by design). I checked the VIS-30 pair against the pre-part-2 file and the VIS-31 set against the pre-part-1 files; I did not run a clean checkout of master (the local `master` ref here is stale).
- venv a: new two files + `test_page_reads_offline.py`: 37 passed (before the review fixes; rerun results are in the commit's report). Guards (`test_public_text`, `test_review_freeze`): 13 passed. DR `test_boundary` + `test_review_fingerprints`: 9 passed.
- venv b: `funhouse_agent/tests/test_turn_question.py`, `test_turn_question_isolation.py`, the new tests, `test_page_reads_offline.py`, every `test_vision_*.py` naming legibility/tile prompt, `webapp/tests/test_page_fingerprints.py`: 98 passed.
- Existing tests edited (review fix 4): `webapp/tests/test_record_ev06.py` and `packages/aec-document-review/tests/test_record_ev06.py`, in `test_a_side_calls_model_end_says_what_it_was_asked` and `test_a_long_side_prompt_keeps_its_start_and_a_hash`. They pin the recorded prompt of a `region` read; the note is now part of that prompt, so the expectations are computed from `ONE_IMAGE_NOTE`. They still check that the record gives the prompt's length, keeps its start, and carries a hash when it is cut.

## Fingerprints
Not moved (side-call prompts are not pinned; tool names, descriptions and schemas unchanged). Nothing re-recorded.

## Calls made
- The note's wording is the one in `ONE_IMAGE_NOTE` after the review (one image and nothing else; report what the image shows, as the request sets out; the comparison needs the other image). The legibility text prints whole pixels, says "assumed" in plain words only where the height is the assumed lettering size, and no longer carries the "box a few tens of points" aside.
- The note sits under the main prompt and above the user's question, and applies to all `QUESTION_KINDS` including `reference_figure` and `worked_example_page`, as the brief says.
- Tile rows keep their shape: the px measure is a private key removed before output.
- The `other tiled path` check found none; the other-branch file check (`git diff` against `claude/gm-316-317-warnings-allowed-values`) failed here for lack of a merge base, but none of my files (`vision_tools.py`, `turn_question.py`) are in the lists the brief names.

## Release note
Every reading that goes back to the main model (page, overview, tile, region, image, reference figure and worked-example page) is now TOLD it is shown one image and nothing else, and that a comparison with something not in the image needs the other image. No model has been run, so what it will answer is not known. Tiled page reads of large sheets keep the "small lettering" warning and name the tiles still too small to read reliably, with whole-number pixel heights and a plain statement where the height is assumed because the page has no text to measure.

## For the lead
- Next suite run of the revision task shows whether the one-image line changes conclusions.
- VIS-30's rest (tiling by the sheet's own regions) needs a design and a measurement on E-size sets; VIS-31 closes once task B's part 2 is merged.

## Review fixes (2026-11-13)
1. `turn_question.ONE_IMAGE_NOTE` reworded; the new tests assert on the constant.
2. `ask_vision`: the "does the prompt already carry the question" check now sees the prompt without the note; order kept (agent's prompt, note, user's question block); test added.
3. `_tile_legibility`: whole-number px, assumed heights said so in plain words, parenthetical removed; tests adjusted and one added.
4. Both `test_record_ev06.py` files: expectations computed from `ONE_IMAGE_NOTE`.

## Noted, not fixed
An agent that is not offered `render_region` (the geotech page's calc helper) still gets "zoom with render_region" in the legibility note on a tiled read, because `deep/vision_kit._without_zoom_tool` strips `zoom_hint` and not `legibility`; master already did this on untiled reads.
