# VIS-33 and VIS-34: check notes

Both items are in the vision tools of the shared core
(`packages/aec-agent-core/aec_agent_core/vision_tools.py`). One notes file
covers both. Commits: `9b6bb8e` (VIS-33), `ffabaf2` (VIS-34), then the
review's fixes: `fb0b826` (VIS-33: behind a switch, with the page's own
evidence), `ef946b7` (VIS-34: the cut sentence).

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

**The switch.** `GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK`, default off; `1` turns
it on. It lives in `review_flags.py` (`PHOTO_PAGE_ONE_LOOK_ENV`,
`photo_page_one_look()`), is documented in that module's docstring beside
`GEOTECH_VISION_TILE_ANSWERS`, and sits in `review_flags.VISION_ENVS` with
it (out of `ALL_ENVS` and `ARMS`, for the reasons under VIS-34). Off, the
page's kind is never read and every read is byte-for-byte as on master,
pages of photographs included. The lead's ruling: robust first, cost second.
A drawing that loses its tiles is the serious failure; a page of
photographs that is still tiled only costs calls.

**What was built.**
- `_read_pdf_page`: a rule beside the blank-page rule. It applies only when
  the switch is on, `tiles == "auto"` and `auto` would have tiled. If
  `_photo_page` says yes, the page takes one look and `tiles_note` is
  `PHOTO_PAGE_NOTE` ("this is a page of photographs, so it was read in one
  look; for a closer look ask for tiles (tiles='2' to '4') or zoom with
  render_region"). An explicit `tiles` value is honoured exactly as before.
- `_photo_page(pdf_bytes, page)`: true only when BOTH hold.
  1. `_planlens_says_photos`: planlens names the page a page of
     photographs (role `photos`), reading that one page with no OCR
     (`Document.summary` and `Document.page`, `roles.facts_for`, `assign`).
     That cue alone is weak (below).
  2. `_not_photographs` finds nothing against it: the page shows by itself,
     read with PyMuPDF on that one page (no OCR, no model call, no pass over
     the document), what a page of photographs looks like. The checks
     (`_page_evidence`), cheapest first, stopping at the first that fails:
     - 2a. under 60 words in the text layer, an OCR layer included
       (planlens' own bar for a photograph plate, a picture with a caption;
       a drawing's title block alone runs to more);
     - 2b. at most 12 pictures, together covering under 75 % of the page (a
       scanned drawing is one page-sized picture or strips of one; a plan
       drawn over an aerial photograph has its base filling the sheet);
     - 2c. under 200 vector paths on the page (a frame, a caption box, a
       small sketch; a drawing has hundreds, and one whose lettering is
       drawn as paths has no text layer for 2a to count);
     - 2d. no path and no word over a picture of 1 % of the page or more
       (inside its box less 3 % a side; a frame round it does not count):
       marks on a picture make it the base of a plan or an annotated figure;
     - 2e. photographs by their pixels (`_photographic`) covering at least a
       fifth of the page. Each picture of 1 % of the page or more is rendered
       once as a thumbnail of at most 96 pixels on its long side. It is a
       photograph when at least 48 colours at 4 bits a channel each hold
       0.1 % of it or more, its commonest colour holds at most half of it,
       and at most half its pixels equal the one to their right. A bilevel
       or paper-toned scan is mostly paper with a short ramp to its ink; a
       logo or a chart is a few flat colours.
  Any failure in either part answers false, so the page is tiled.
- The rule and its reasons are stated once, in the comment above
  `PHOTO_ROLE` in `vision_tools.py`; each number is a named constant there.

**Why the role alone was not enough (the review).** planlens 0.16.0 names a
page `photos` from one phrase of `PHOTO_TITLES` in its title or its top or
bottom band ("photo", "plate no", "looking north", "view looking",
"stockpile", "core box", "upon completion" ...), at least one picture and
under 220 words; with two phrases, at any length. It runs before the plan,
profile and figure cues. With the role alone, each of the seven drawings
below took one vision call instead of five to seventeen.

**The seven drawings, each tiled with the switch on.** Each is a made-up
page in `test_on_a_drawing_the_role_alone_would_pass_is_tiled`; the test
first asserts that planlens' role alone names it photographs (so the old
rule would have read it in one look), then that the named check stops it,
then that the read takes 1 + n*n calls.

| Drawing | Stopped by | Would also be stopped by |
|---|---|---|
| D-size location plan titled "BORING LOCATION PLAN / PLATE NO. 1" over an aerial base, 155 words | 2a (155 words) | 2d (boring symbols over the base) |
| D-size "CROSS SECTION A-A / LOOKING NORTH" with a 40 x 20 logo | 2e (photographs cover 0 %: a logo is under 1 % and not a photograph) | |
| D-size "GRADING AND STOCKPILE PLAN" with a logo | 2c (over 200 paths: 240 contour segments) | 2e |
| D-size "FOUNDATION PLAN" with a photo inset captioned "PHOTO 1 EXISTING FOOTING" | 2e (the inset is a photograph but covers 3 % of the sheet) | |
| D-size "PHOTO LOCATION PLAN" over an aerial image | 2d (arrows drawn over the aerial) | 2d (numbers lettered over it) |
| D-size profile sheet under the top-band header "APPENDIX B SUBSURFACE PROFILES AND SITE PHOTOGRAPHS" | 2a (120 words) | 2e (a logo only) |
| Scanned drawing (one page-sized picture) with an invisible OCR layer of 44 words holding "LOOKING NORTH" or "PLATE NO. 4" (two cases) | 2b (pictures cover 100 %) | 2e (a bilevel scan) |

**Correct as built, and still so with the switch on.**
`test_on_other_pages_that_are_not_photographs_are_tiled` and
`test_on_a_scanned_drawing_with_no_text_layer_is_still_tiled`: a scanned
drawing with no text layer (2b), the same in three strips (2b), a sheet of
400 words with a photograph (2a), a sheet with no picture, and a page of
photographs with one word lettered over a photograph (2d) are all tiled.
`test_on_a_page_of_photographs_takes_one_look`: a page of two captioned
photographs, the same page turned 90 degrees (`/Rotate`), and a photograph
beside a dimensioned sketch each take one look.

**Calls made.**
- Behind a switch, off by default: the lead's ruling, which replaced the
  first build's "on by default, it refines what `auto` means".
- The page's own evidence runs before planlens is asked: most drawings stop
  at the word count, so they cost less than before, not more. The planlens
  pass runs only on pages that already look like photographs.
- Rule 2 reads that one page; the page's `/Rotate` is set to 0 in the copy
  it reads (boxes, words and paths then share one coordinate space; a turn
  says nothing about what is on the page).
- The thresholds are mine, set to the robust side and tested both ways on
  the made-up pages: 60 words is planlens' own plate bar; 75 % keeps a page
  of two or four photographs with margins (about 55 to 65 %) and stops a
  scan or a base filling the sheet; 200 paths admits a small sketch; a
  fifth of the page admits one photograph beside a sketch on a letter page
  (25 % in the test) and stops an inset on a drawing.
- A greyscale photograph has only 16 tones at 4 bits a channel and does not
  count as a photograph (`test_a_picture_is_a_photograph_by_its_pixels`):
  a page of black-and-white photographs is tiled, at the cost of calls.
  Telling a greyscale photograph from a dense downsampled line drawing by
  its tones alone was not robust enough to trust.
- The boundary, stated so a reviewer can judge it: a small sheet that is
  half photograph and half sketch, with under 60 words, under 200 paths and
  nothing drawn over the photograph, takes one look. That is the "photograph
  beside a dimensioned sketch" page the ruling wants read in one look; a
  figure built the same way cannot be told from it by the page alone.
- `assign` numbers pages from 0 (it indexes ranges by `len(facts)`), so the
  one page is passed as page 0 (`dataclasses.replace`).
- The note goes in the existing `tiles_note` key, which the blank-page rule
  also uses. The record: the per-call vision record shows one call of kind
  `page` for that page, not `overview` plus tiles, and the tool result
  carries `tiles_note` saying why. No new field, no tool description changed.

**Two statements of the first notes were wrong (corrected by the review).**
- "It also skips the planlens pass on vector drawing sheets": the picture
  check before planlens passed whenever a sheet carried a logo or a base
  image, so most drawing sheets were still read by planlens and could be
  named photographs. Now a drawing stops at its own evidence, mostly at the
  word count, before planlens is asked.
- "What is lost: the appendix tab": reading the page alone also loses
  planlens' masking of running headers. A header printed on every page of an
  appendix ("APPENDIX B SUBSURFACE PROFILES AND SITE PHOTOGRAPHS") counts as
  the page's own caption, which is how a profile sheet was named
  photographs. Rule 2 now tiles such a sheet by what is on it.

**Tests** (`packages/aec-agent-core/tests/test_vis33_photo_pages_one_look.py`;
made-up pages built in the test with PyMuPDF, a fake vision model):
- `test_the_switch_is_off_unless_set`,
  `test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone`: the
  switch's name, default and off values; `describe()` names it when set; an
  arm pair turns it on, the baseline clears a leaked value.
- `test_off_every_read_is_as_before` (a photo page and a drawing, each with
  the switch unset, empty or 0): `_photo_page` is never called, and every
  prompt and the whole result are byte-for-byte those of a read whose photo
  rule can never fire, which is the code path on master; the photo page is
  tiled (1 + n*n calls).
- `test_on_a_page_of_photographs_takes_one_look` (three pages, above),
  `test_on_a_photo_page_takes_one_look_and_says_tiles_can_be_asked` (one
  call, record kind `page`, the note names tiles and render_region),
  `test_on_a_photo_page_is_tiled_when_tiles_are_asked_for`.
- `test_on_a_drawing_the_role_alone_would_pass_is_tiled` (the seven, eight
  cases), `test_on_other_pages_that_are_not_photographs_are_tiled` (four),
  `test_on_a_scanned_drawing_with_no_text_layer_is_still_tiled`.
- `test_a_picture_is_a_photograph_by_its_pixels`: a colour photograph is
  one; a greyscale photograph, a logo, a bilevel scan and a paper-toned scan
  are not.
- Doubt tiles: `test_a_page_whose_kind_cannot_be_told_is_tiled_as_before`
  (planlens' roles unimportable), `test_a_failing_role_read_tiles_as_before`
  (`assign` raises), `test_a_failing_look_at_the_page_tiles_it` (rule 2
  raises), `test_the_efficient_policy_still_reads_one_look_without_the_note`
  (a guard).
- On master every test in the file fails at import: it names
  `review_flags.PHOTO_PAGE_ONE_LOOK_ENV`, which master lacks.

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
- The review's fix: `TILE_CUT_SENTENCE`, placed in the shape before the
  one-line exit: "Where this tile's edge cuts a table, a dimension string or
  a note that may bear on the request, quote the part of it this tile holds,
  even if its labels or what it refers to lie outside the tile." Without it,
  a schedule cut with its values in one tile and its row labels in the next
  lost the value: the value tile showed nothing that visibly bore on the
  ask and took the one-line exit.
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
  VIS-34, and no prompt carries the shape, the one line or the cut
  sentence. Checked once more outside the suite at the first build: a
  capture of seven reads (PDF tiles 2 and 3, auto, off, turned, a zoom,
  image tiles) on the code before the commit and after it, with the switch
  unset, gave identical files.
- `test_on_each_tile_gets_the_answer_shape` (PDF and image): every tile
  prompt reads, in order, the ask, the tile line, the shape, then the
  location. The overview gets no shape.
- `test_on_a_tile_quotes_its_part_of_what_its_edge_cuts` (PDF and image):
  with the switch on, every tile prompt carries the cut sentence once,
  before the one-line exit; the overview does not.
- `test_on_the_shape_asks_for_nothing_less_from_a_tile_that_holds_it`.
- `test_on_page_and_zoom_reads_are_unchanged`.
- `test_on_nothing_is_cut_from_a_tile_that_holds_something`: one-line
  tiles come back as answered, and the full tile comes back whole.
- On the code before the commit, all of them fail: the switch does not
  exist there.

**Fingerprints.** Neither moved: `test_review_fingerprints.py` and
`test_page_fingerprints.py` pass after both fixes.

## Noted, not fixed

- A page of black-and-white photographs, a photo log whose descriptions run
  to 60 words or more, and a photograph stored in more than 12 pieces are
  tiled with the switch on (the robust side; they cost calls only).
- `analyze_image` on a large photograph file (not a PDF page) still tiles
  under `auto`. planlens gives no role to an image file.
- A scanned page of photographs (one page-sized picture) still tiles, by
  rule 2b. Its kind cannot be told from the page alone without a model.
- An identical read repeated within one conversation is served from the
  repeat cache. If a switch flips mid-conversation, which only a notebook
  can do, that read keeps the earlier prompt's answer.

## For the lead: what a measurement on the hosted platform should compare

- **VIS-33.** The suite arm
  `("photo_one_look", {"GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK": "1"})` against
  the baseline, same model and questions, on documents that hold report
  pages of photographs (captioned photo logs) AND drawing sheets, among them
  sheets carrying a photo phrase ("PLATE NO.", "LOOKING NORTH", a photo
  location plan, an appendix header naming photographs). Compare: vision
  calls per photo page (5 or more in the baseline, 1 in the arm); vision
  output tokens and seconds per page; whether the agent's answers about the
  photographs keep the same content (the facts it states and cites); and
  how often the agent then asks for tiles or a zoom on those pages, since
  that is the cost moved back. The failure to look for: list every page the
  arm read in one look (its `tiles_note` is `PHOTO_PAGE_NOTE`) and check each
  is a page of photographs; a drawing in that list is the serious failure.
  Drawing sheets should take the same number of calls in both arms.
- **VIS-34.** Take the large drawing sheets read in tiles, with the suite
  arm `("tile_answers", {"GEOTECH_VISION_TILE_ANSWERS": "1"})` against
  the baseline, same model. Compare: vision output tokens per tiled read
  and per tile (the item's figure was about 89 thousand on two sheets);
  seconds per tiled read; the share of tiles answering exactly "Nothing on
  the request in this tile."; and, on tiles that do hold the asked items,
  whether every item and its quoted lettering is still there (recall
  against the sheet's text layer), including items in tables or dimension
  strings that a tile edge cuts. Also compare suite scores, to show the
  agent's answers did not lose anything.
