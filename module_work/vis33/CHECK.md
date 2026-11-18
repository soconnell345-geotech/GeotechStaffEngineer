# VIS-33 and VIS-34: check notes

Both items are in the vision tools of the shared core
(`packages/aec-agent-core/aec_agent_core/vision_tools.py`). One notes file
covers both. Commits: `9b6bb8e` (VIS-33), `ffabaf2` (VIS-34).

## VIS-33: a page of photographs takes one look on `tiles='auto'`

**What the item asked.** Under `tiles='auto'` a page of photographs was
read whole and in tiles: five or more vision calls, with readings five to
seven times longer than one whole-page look. Such a page should take one
look, and its result should say that tiles or a zoom can be asked for. A
scanned drawing (pictures with no text layer) must still be tiled.

**Why it tiled.** `vision_view.tile_grid` treats a page with fewer than 40
characters in its text layer as a drawing and assumes 0.06 in drawing
lettering. That is too small at the whole-page budget, so it tiles. A
page of photographs with short captions, or none, falls into that rule.

**What was built.**
- `vision_tools._photo_page(pdf_bytes, page)` asks planlens for the page's
  role. It reads only that one page, with no OCR: `Document.summary` and
  `Document.page` give the facts, `planlens.document.roles.facts_for`
  turns them into PageFacts, and `assign` gives the role. It answers True
  only for role `photos`, and only when a picture is drawn on the page.
  It answers False whenever the kind cannot be told: planlens has no page
  roles, an import fails, or anything raises.
- `_read_pdf_page`: a new rule beside the blank-page rule. It applies only
  when `tiles == "auto"` and `auto` would have tiled. If `_photo_page` says
  yes, the page takes one look and `tiles_note` is `PHOTO_PAGE_NOTE`
  ("this is a page of photographs, so it was read in one look; for a
  closer look ask for tiles (tiles='2' to '4') or zoom with
  render_region").
- On by default, with no switch, as the lead ruled: the change refines
  what `auto` means. An explicit `tiles` value is honoured exactly as
  before.

**Calls made.**
- The role comes from the page alone, not from the whole document. A
  whole-document role pass reads every page, which on a dense 40-sheet
  set takes many seconds. The page's own pass costs one page. What is
  lost: the appendix tab. A page that is a photograph only by its tab
  ("APPENDIX E, SITE PHOTOGRAPHS") and has no caption of its own is not
  named photographs, so it is tiled as before.
- `assign` numbers pages from 0 (it indexes ranges by `len(facts)`), so
  the one page is passed as page 0 (`dataclasses.replace`).
- A picture must be drawn on the page (`get_image_info()`) before planlens
  is asked. This is part of the meaning, since a page with no picture is
  not a page of photographs. It also skips the planlens pass on vector
  drawing sheets.
- The note goes in the existing `tiles_note` key, which the blank-page
  rule also uses, rather than a new key.
- The record: the per-call vision record shows one call of kind `page` for
  that page, not `overview` plus tiles. The tool result in
  `activity.jsonl`, about 1 KB here and well under its 32,000-character
  cap, carries `tiles_note` saying why. No new field was needed.
- No tool description changed. The agent can ask for exactly what it
  could ask for before, and the result says the rest.

**Tests** (`packages/aec-agent-core/tests/test_vis33_photo_pages_one_look.py`;
made-up pages built in the test, a fake vision model):
- `test_planlens_names_the_made_up_photo_page_photographs`: planlens names
  a page of two photographs captioned "Photo 1" and "Photo 2" as
  photographs, and the scanned drawing as not.
- `test_a_photo_page_takes_one_look_on_auto_and_says_tiles_can_be_asked`:
  `auto` would tile this page by its lettering. It now makes one vision
  call (record: kind `page`, page 0), returns no tiles, and its
  `tiles_note` names tiles and render_region.
- `test_a_photo_page_is_tiled_when_tiles_are_asked_for`: `tiles='2'`
  gives the page plus four tiles, with no note.
- `test_a_scanned_drawing_with_no_text_layer_is_still_tiled`: one
  page-sized line-art picture with no text gets 1 + n*n calls.
- `test_a_page_whose_kind_cannot_be_told_is_tiled_as_before`: with
  planlens' roles module made unimportable, the photo page is tiled.
- `test_a_failing_role_read_tiles_as_before`: when `assign` raises, the
  page is tiled.
- `test_the_efficient_policy_still_reads_one_look_without_the_note`: a
  guard.
- On the code before the commit, the first five fail and the last two pass
  (they are guards on unchanged behaviour).

## VIS-34: tile reads get an answer shape, behind a switch

**What the item asked.** Give tile reads the answer shape that page and
zoom reads have, and a one-line reply when a tile holds nothing on the
ask.

**What was built.**
- A new switch, `GEOTECH_VISION_TILE_ANSWERS`. Default off; `1` turns it
  on. It lives in `review_flags.py` (`TILE_ANSWERS_ENV`, `tile_answers()`)
  and is documented in that module's docstring with the others.
- `vision_tools.TILE_ANSWER_SHAPE` and `TILE_NOTHING_LINE` ("Nothing on
  the request in this tile."). The shape asks the model to answer from
  what is in the tile, to quote the lettering that bears on the request,
  to leave none of it out, and, when nothing bears on it, to give that one
  line without listing what is absent. `_SHAPE_TAIL` follows, as for page
  and zoom reads.
- `_answer_shape("tile")` returns the shape only while the switch is on,
  otherwise `None`. `_tile_jobs` (PDF tiles) and `_image_tile_jobs` (image
  tiles) pass it to `_vision_prompt`. Each read reads the switch once.
- Unchanged: the whole-page layout overview, a page read whole, a zoom,
  the caps, and `_fit_tiles`. Nothing is cut from a tile's answer.

**Calls made.**
- The switch is kept out of `ALL_ENVS` and `ARMS`. Two of the app's own
  tests pin those sets exactly:
  `funhouse_agent/deep/tests/test_judgment_calls.py` (readers ==
  `ALL_ENVS`) and `funhouse_agent/deep/tests/test_review_switches_offline.py`
  (`ARMS ==`). A builder here may not edit the app. So the switch sits in
  a new tuple, `review_flags.VISION_ENVS`: `switches()` clears it like the
  others, and `describe()` names it when set, so the turn record's
  `build.switches` shows it. The suite takes the arm as a pair:
  `("tile_answers", {"GEOTECH_VISION_TILE_ANSWERS": "1"})`.
- It applies to image tiles too: both kinds of tile read are the same
  kind of call, and the rule is general.
- A fixed one-line reply was chosen so that a measurement can count empty
  tiles by text.
- The record: switch state is in `build.switches` per turn. Each tile
  call's prompt is kept in its first 300 characters plus a fingerprint;
  the shape sits after the ask, so the fingerprint differs when the switch
  is on. No new field was needed.

**Tests** (`packages/aec-agent-core/tests/test_vis34_tile_answers.py`):
- `test_the_switch_is_off_unless_set`: off when unset or set to
  ``/0/no/off/false``; on when set to `1`.
- `test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone`:
  `describe()` names it when set, an arm pair turns it on, the baseline
  clears a leaked value, and the environment is restored afterwards.
- `test_off_every_prompt_and_result_is_as_before` (PDF tiles and image
  tiles, each with the switch unset, empty or 0): every prompt and the
  whole result are byte-for-byte those of the code path from before
  VIS-34. Checked once more outside the suite: a capture of seven reads
  (PDF tiles 2 and 3, auto, off, turned, a zoom, image tiles) on the code
  before the commit and after it, with the switch unset, gave identical
  files.
- `test_on_each_tile_gets_the_answer_shape` (PDF and image): every tile
  prompt reads, in order, the ask, the tile line, the shape, then the
  location. The overview gets no shape.
- `test_on_the_shape_asks_for_nothing_less_from_a_tile_that_holds_it`.
- `test_on_page_and_zoom_reads_are_unchanged`.
- `test_on_nothing_is_cut_from_a_tile_that_holds_something`: one-line
  tiles come back as answered, and the full tile comes back whole.
- On the code before the commit, all of them fail: the switch does not
  exist there.

**Fingerprints.** Neither moved: `test_review_fingerprints.py` and
`test_page_fingerprints.py` pass.

## Noted, not fixed

- `analyze_image` on a large photograph file (not a PDF page) still tiles
  under `auto`. planlens gives no role to an image file.
- A scanned page of photographs with no text layer still tiles. Its kind
  cannot be told without a model, so it falls back to today's behaviour.
- An identical read repeated within one conversation is served from the
  repeat cache. If the switch flips mid-conversation, which only a
  notebook can do, that read keeps the earlier prompt's answer.

## For the lead: what a measurement on the hosted platform should compare

- **VIS-33.** Take report pages of photographs (captioned photo logs),
  read with `tiles='auto'`, master against this branch, with the same
  model and questions. Compare: vision calls per photo page (5 or more
  before, 1 after); vision output tokens and seconds per page; and whether
  the agent's answers about the photographs keep the same content (the
  facts it states and cites). Count how often the agent then asks for
  tiles or a zoom on those pages, since that is the cost moved back. As a
  control, check that scanned drawing sheets and vector sheets in the same
  run take the same number of calls as before.
- **VIS-34.** Take the large drawing sheets read in tiles, with the suite
  arm `("tile_answers", {"GEOTECH_VISION_TILE_ANSWERS": "1"})` against
  the baseline, same model. Compare: vision output tokens per tiled read
  and per tile (the item's figure was about 89 thousand on two sheets);
  seconds per tiled read; the share of tiles answering exactly "Nothing on
  the request in this tile."; and, on tiles that do hold the asked items,
  whether every item and its quoted lettering is still there (recall
  against the sheet's text layer). Also compare suite scores, to show the
  agent's answers did not lose anything.
