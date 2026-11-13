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
- `test_vis31_one_image.py` (all fail before the change: no note), `test_vis30_tile_legibility.py` (the note and assumed-px tests fail before; "dropped when every tile reads" passes on master by design). I checked the VIS-30 pair against the pre-part-2 file and the VIS-31 set against the pre-part-1 files; I did not run a clean checkout of master (the local `master` ref here is stale).
- venv a: new two files + `test_page_reads_offline.py`: 37 passed. Guards (`test_public_text`, `test_review_freeze`): 13 passed. DR `test_boundary` + `test_review_fingerprints`: 9 passed.
- venv b: `funhouse_agent/tests/test_turn_question.py`, `test_turn_question_isolation.py`, the new tests, `test_page_reads_offline.py`, every `test_vision_*.py` naming legibility/tile prompt, `webapp/tests/test_page_fingerprints.py`: 98 passed.
- Existing tests edited: none.

## Fingerprints
Not moved (side-call prompts are not pinned; tool names, descriptions and schemas unchanged). Nothing re-recorded.

## Calls made
- The note sits under the main prompt and above the user's question, and applies to all `QUESTION_KINDS` including `reference_figure` and `worked_example_page`, as the brief says.
- Tile rows keep their shape: the px measure is a private key removed before output.
- The `other tiled path` check found none; the other-branch file check (`git diff` against `claude/gm-316-317-warnings-allowed-values`) failed here for lack of a merge base, but none of my files (`vision_tools.py`, `turn_question.py`) are in the lists the brief names.

## Release note
Page, region and tile readings are now told they see one image, so a question to compare with another issue is answered with a transcription and a statement that the other image is needed, instead of "none identifiable". Tiled page reads of large sheets keep the "small lettering" warning and name the tiles still too small to read reliably.

## For the lead
- Next suite run of the revision task shows whether the one-image line changes conclusions.
- VIS-30's rest (tiling by the sheet's own regions) needs a design and a measurement on E-size sets; VIS-31 closes once task B's part 2 is merged.

## Noted, not fixed
None.
