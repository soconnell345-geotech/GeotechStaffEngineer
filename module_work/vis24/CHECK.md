# VIS-24: check notes

## What the item asked

Two halves of one defect in the vision side calls (`analyze_pdf_page`,
`render_region`), which see one image and never the page:

1. A request for positions in page units ("the centre in PDF points",
   "coordinates on the page") should be asked of the vision model as pixel
   boxes, which the tool converts itself. In the field trace that raised
   it, fourteen zooms asked for "symbol centre in PDF page points" answered
   in at least four frames (pixels of the crop, points from the crop's
   corner, page points, an origin at the lower left).
2. Page coordinates a side call writes into its prose answer should not be
   passed on as if measured (was VIS-15: one model gave some beyond the
   page).

The lead's ruling: both halves change what the vision model is asked or
what the agent is handed back, and neither is measured, so both sit behind
one new switch, off by default. Unset, prompts and results are byte for
byte as before, and no fingerprint moves.

## What was built

All in `packages/aec-agent-core/aec_agent_core/`.

- **The switch.** `GEOTECH_VISION_PIXEL_REQUESTS`, default off, `1` turns
  it on. `review_flags.PIXEL_REQUESTS_ENV` and `review_flags.pixel_requests()`,
  documented in that module's docstring with the others, and added to
  `LATER_ENVS` beside the three newest (not `ALL_ENVS` or `ARMS`, which
  the app's own tests pin whole). `switches()` clears it unless an arm
  names it; `describe()` names it when set, so a turn's `build.switches`
  shows it. The vision tools read it as they look; the build is unchanged.
- **What counts as a request for positions in page units:**
  `vision_tools.asks_page_positions(prompt)`. One sentence must both name
  the page's own units or frame (PDF points, PDF or page coordinates, units
  or space, coordinates on the page) and ask where something is
  (coordinates, position, location, centre, box, corners, x and y, where).
  The other way in: a position word takes points as its unit in one clause
  ("the bbox in pt"), with no word of a size or distance between.
  Everything else is left alone.
- **Half 1, the prompt.** `_vision_prompt(..., positions=False)`. With
  `positions` true (the switch is on and the request is one) and the image
  size known, `POSITIONS_AS_PIXELS` goes right after the agent's own words.
  It tells the model that it sees only the image, that it should give no
  page coordinates or point values for a position, and that it should give
  each position as `px=[x0, y0, x1, y1]` (a small box round a point). It
  also says the tool converts every pixel box to page points. The
  location instruction that ends every prompt is unchanged. Passed by
  `_read_pdf_page` (the page read whole, or the layout overview of a tiled
  read), by `_tile_jobs` (each tile) and by `_dispatch_render_region` (a
  zoom on a PDF page).
- **Half 1, the conversion.** `_px_page_boxes(raw, info)` takes each box
  the answer tagged as pixels (`px=[..]`, `[..] px`). It converts it with
  the image's size as sent and the turn undone, through
  `vision_view.px_box_to_page`, which is the conversion the `found` boxes
  use without the 0-999 rounding in between. Each box is labelled, as
  `answer_boxes` labels one, with the words before it on its line. A box
  with a page unit written after it ("px=[..] pt") is skipped. A tiled
  read does the same for each look, never from `found`
  (`_tiled_page_points`): the overview's answer is read before its boxes
  are rewritten, and each tile's answer is converted in that tile's own
  frame inside `_tile_jobs`, before its boxes are rewritten, and carried on
  the tile's row as a private `_px_points` key that the page read takes
  off. So it never reaches the result, the repeat cache or a kept reading.
  The same thing seen in two looks is one row, with `seen_in` naming the
  looks (`page` for the overview, as in `found`); see "Calls made". The
  result gets `page_points` (at most `MAX_PAGE_POINTS` = 24; then
  `page_points_cut`): each row has `what`, `page_bbox` and `centre`, in PDF
  points, origin top-left, y down: the frame `bbox` takes. It also gets
  `page_points_note` (`PAGE_POINTS_NOTE`), which says how the points were
  made and that a box is not a measurement. If no pixel box came back, the
  result gets `NO_PAGE_POINTS_NOTE` instead.
- **Half 2, the note.** `vision_tools.states_page_coordinates(text)` reads
  the answer as the model wrote it, before the tool converts its boxes. A
  statement is one of these:
  - two or four numbers in brackets, or x and y given separately
    ("x = 612", '"x": 612'), with a page unit right after them ("(612,
    396) pt", "[..] in PDF points", "(x=612, y=396) pt", "(..) PDF-view
    pt") or the page's frame named just before them in the same sentence
    ("PDF points: (612, 396)", "page coordinates [..]");
  - the same with a unit after a number inside them ("(612 pt, 396 pt)"),
    or with points named as the unit just before them ("in points:
    {..}"), when no word of a size sits just before or right after them;
  - two bare numbers after a position word, with a page unit ("centre at
    612.4, 396.0 pt"), no size word beside them.

  Pixel boxes are excluded unless a page unit follows them ("px=[..] pt":
  then no pixel box is taken from it either). Also excluded: sizes ("(8,
  10) pt lettering", "text heights 8, 10 pt"), dimensions ("1224 x 792
  pt", "12 ft"), the verb (a bare "points" is a unit only before
  punctuation or the end of a line, so "(..) points right to the boring"
  is not one; "pt", "pts", "PDF points", "page points" and "in points"
  always are), stations,
  elevations, sheet numbers, printed survey coordinates and boxes with no
  unit. When the answer (or any tile of a tiled read) states page
  coordinates, the result carries `coordinates_note` (`COORDINATES_NOTE`)
  once. The note says those numbers are the vision model's own estimate,
  not measured, possibly in another frame. It says a box with a page unit
  after it is one of them even where its numbers were rewritten on the
  0-999 grid, and is not to be used as a place on the page. It points to
  the boxes the tool converted. A tile marks itself with a private
  `_page_units` key, which the page read takes off every row before the
  result goes out.
- Unchanged: image files (`analyze_image`, `render_region` on an image),
  inline looks, every tool description and schema, the caps, the repeat
  cache and kept readings.

## Calls made

- **One switch for both halves**, as the lead ruled.
- **The agent's words are kept and an instruction is added**; the request
  is not rewritten in place. Turning "in PDF points" into "as pixel boxes"
  inside the agent's own sentence cannot be done grammatically in general,
  and it could change what a compound request asks. The added paragraph
  sits right after the request and says plainly which part it replaces.
- **"Coordinates" alone does not count.** On a plan it is usually the
  survey coordinates printed there (northing and easting), which the vision
  model should read as printed. "Points" alone does not count either: it
  may be a size. A request for a location with no unit already gets pixel
  boxes from the location instruction. So when unsure, the rule leaves the
  request alone, and those requests stay byte for byte as with the switch
  off (tested).
- **Only boxes tagged as pixels become `page_points`, in every read,
  tiled included.** An untagged box is converted to the 0-999 grid in the
  prose when the answer shows it is in pixels, as before. It is not listed
  as a page point: it may be the model's own page numbers, and listing it
  would pass them on as converted. A tagged box with a page unit written
  after it is not listed either. A tiled read used to list its `found`
  rows, which hold every box the tiles' prose had on the grid, so a box
  stated in points was listed as if converted from pixels, and a bare box
  was listed with no note. It now converts the tagged boxes of each look
  itself.
- **Region-sized boxes are kept in `page_points`.** `answer_boxes` (and so
  `found`) drops boxes over 300 of 999. A zoom on one symbol, asked for its
  centre, boxes a symbol that fills most of the window, which is exactly
  the case the item names; a title block asked for in a tiled read is
  large on every tile. Neither is dropped.
- **One row per thing in a tiled read.** Boxes from two looks are one
  thing when they are within the reach `found` merges by (2 % of the page,
  at least `vision_view.MIN_LOCATION_ERROR_PT`) and their names carry no
  different codes, the rule `found` uses. Two boxes from one look are never
  joined: that look named them apart. A row's box is its tiles' boxes
  joined, or the overview's box where no tile gave one: the tiles are the
  closer look, and a thing a seam cuts (a title block down the sheet's
  edge) is seen in part in each tile. `found` keeps the smallest view's
  box instead, which for a cut thing is one part. `page_points_cut`
  counts the whole list, not `found`'s capped one.
- **The note is true with either planlens.** planlens 0.16.0 rewrites a
  box with a page unit after it onto the grid as if it were pixels; the
  companion repair (below) leaves it as written. `COORDINATES_NOTE` says
  such a box is the model's own estimate "even where its numbers were
  rewritten on the 0-999 grid", and does not claim the tool left it alone.
  The core does not depend on the repair. `page_points` is built by the
  core alone and is the same with both.
- **Straight from pixels to points.** `px_box_to_page` is planlens'
  conversion for this. It uses the same size and turn as the `found` path,
  minus the 0-999 rounding (up to 0.6 pt on a whole 17 x 11 in sheet).
- **Half 2 applies to every page or zoom answer while the switch is on**,
  not only to requests for page units. A model can volunteer page
  coordinates unasked, and the note costs nothing when the answer has none.
- **PDF pages only.** An image file has no page frame (its boxes are its own
  pixels), and an inline look has no side call.
- **Cap of 24 rows** keeps a zoom's result under `REGION_RESULT_CHARS`
  without cutting much prose. The prose keeps every box as before.

## Tests

`packages/aec-agent-core/tests/test_vis24_pixel_requests.py` (made-up
sheets and pictures; a fake vision model finds a black square in the
image it is sent and answers with its pixel box):

- `test_the_switch_is_off_unless_set_and_kept_with_the_later_switches`: off
  unless `1`; in `LATER_ENVS`, not in `ALL_ENVS` or any arm of `ARMS`.
- `test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone`:
  `describe()` names it; the pair turns it on; the baseline clears it.
- `test_a_request_for_positions_in_page_units_is_one` (9 made-up phrasings)
  and `test_anything_else_is_left_alone` (14: printed coordinates, sizes in
  points, a location beside a text height in points, stations, sheet
  numbers, a file name with a page number, survey points, plain places).
- `test_page_coordinates_in_prose_are_found` (16, among them "center≈
  (516,1170) PDF-view pt", "(x=612, y=396) pt", "(612 pt, 396 pt)",
  "centre at 612.4, 396.0 pt", `{"x": 612, "y": 396}` beside "points", and
  a pixel-tagged box with "pt" after it) and
  `test_ordinary_numbers_are_not_page_coordinates` (24: stations,
  elevations, sheet numbers, scales, dimensions in points and in feet,
  sizes in points (in brackets, bare, with a unit on each number, "8 pt
  text"), printed survey coordinates, pixel boxes alone, boxes with no
  unit, points on a chart, the verb "points to").
- `test_the_verb_points_is_not_a_page_unit` (six made-up sentences:
  "points right to the boring.", "points at", "points upward", "points
  northeast", "points left", "points roughly north"): the pixel box is
  converted, is in a tiled read's `found` and `page_points`, and raises no
  note, with either planlens. `test_a_page_unit_after_a_box_still_holds`
  ("points." / "points," / "points" at the end of a line, pt, pts, "PDF
  points", "in points"): still a statement, and no page point.
- `test_off_every_prompt_and_result_is_as_before` (zoom, page whole, page
  in 2x2 tiles; switch unset, empty and `0`): a request for page points
  whose answer states page coordinates is run twice. The second run puts
  back the `_vision_prompt` from before VIS-24 and makes every new rule
  raise if it is called. Prompts and results are identical, and none of
  the new keys or text appears.
- `test_on_a_zoom_asked_for_page_points_gets_them_converted` (as sent, and
  sent turned 90 degrees): the prompt reads, in order, the ask, the new
  paragraph, the zoom's answer shape and the location instruction. The
  converted `page_points` land within 1.5 pt of the drawn square and its
  centre, so the turn is undone.
- `test_on_a_page_read_whole_or_in_tiles_converts_too`: every look carries
  the paragraph once; the page read whole converts its own box, and the
  tiled read converts the overview's and the tile's into one row with
  `seen_in` `["page", "r1c1"]`, also within 1.5 pt, and no private key is
  left on a tile.
- `test_on_a_box_in_page_units_never_becomes_a_page_point` (zoom, page,
  tiles): every look answers the pixel box and the square's place stated
  as "[..] pt". Only the pixel box is listed, the note is given, and no
  private key reaches the result. The test also checks what the prose holds
  with the planlens installed: the box as written with the repair, a
  rewritten box with "pt" after it with 0.16.0. Run with both.
- `test_on_a_tiled_read_a_bare_box_is_not_a_page_point`: no `page_points`,
  `NO_PAGE_POINTS_NOTE`, no `coordinates_note`.
- `test_on_a_tiled_read_a_large_pixel_box_is_listed`: a title block down
  the sheet's right edge, asked for "the page coordinates of the title
  block", is one row seen in `page`, `r1c2` and `r2c2`, within 1.5 pt of
  the drawn block (the two tiles' parts joined), though `found` holds no
  row for it.
- `test_on_a_tiled_read_one_thing_in_four_tiles_is_one_row`: a square
  where all four tiles overlap is one row at its place, so each tile's box
  was converted in that tile's frame.
- `test_on_a_tiled_read_the_cut_counts_the_whole_list`: 71 rows (past
  `found`'s cap of 60) give 24 and "47 more not listed".
- `test_on_a_request_without_page_units_is_left_alone` (zoom, page, tiles):
  with the switch on, prompts and results are what they are with it off.
- `test_on_an_answer_with_no_pixel_box_gives_no_page_points`: no
  `page_points`, `NO_PAGE_POINTS_NOTE`, and the model's own numbers get the
  note.
- `test_on_stated_page_coordinates_are_said_once` (zoom, page) and
  `test_on_a_tile_stating_page_coordinates_is_said_once_for_the_read`: one
  note per result, and no private key is left on a tile.
- `test_on_ordinary_numbers_and_the_tools_own_boxes_raise_no_note` (zoom,
  page, tiles; both kinds of request): made-up prose full of stations,
  elevations, sheet numbers, sizes and printed survey coordinates, plus the
  pixel boxes the tool converts, raises no note.
- `test_on_an_image_file_is_not_changed` (`analyze_image`, `render_region`
  on an image): identical with the switch on.
- `test_the_page_points_list_is_capped`.

Each fails on the code before the commit: the switch, the two rules and
the new keys do not exist there.

Also run, unchanged and passing: the core's tests that read the changed
functions (`test_vis34_tile_answers.py`, `test_wave3_latency_offline.py`,
`test_page_reads_offline.py`, `test_vis30_tile_legibility.py`,
`test_vis33_photo_pages_one_look.py`, `test_ta43_image_ceiling.py`,
`test_coverage_pages_offline.py`, `test_dr48_photo_pages.py`); the app's
tests that call `_vision_prompt` through the core's aliases or pin the
switch lists (`test_vision_turned.py`, `test_review_harness_offline.py`,
`test_review_switches_offline.py`, `test_judgment_calls.py`,
`test_build_record.py`, `test_kept_readings_offline.py`); and Document
Review's `test_review_fingerprints.py` (no fingerprint moved),
`test_review_harness_offline.py`, `test_wave2b_turns.py`,
`test_dr48_photo_pages_turn.py`, `test_dr50_reader_open_items_offline.py`.

## The review's fixes (second commit pair)

An independent review found the tiled read's `page_points` built from
`found`, a large pixel box lost there, `states_page_coordinates` missing
common forms, and claims here untrue of tiled reads. All four are fixed
above, and every new test fails on the code before the fix. Runs with the
fix, by the shared runner, all exit 0: this file's tests and the app's
`test_vision_locations.py` and `test_vision_brief4.py` with planlens
0.16.0 installed, and again with the repaired planlens; the app's
`test_page_fingerprints.py` and Document Review's
`test_review_fingerprints.py` (nothing moved).

**The companion planlens repair** (default path, no switch; planlens
branch `claude/vis-24-pt-boxes`). `boxes_to_grid` used to rewrite a box
with a page unit after it ("[300, 200, 320, 220] pt") onto the 0-999 grid
as if it were pixels, so the agent was handed a wrong grid box. It now
leaves such a box exactly as written. `boxes_note` names it first, and
`answer_boxes` (so `found` and the zoom's distance from its aim) does not
read it as a place. Every other box converts as before. In the core,
`_finish_answer` is the one caller of `boxes_to_grid`; the app's two tests
that call it pass unchanged.

## Noted, not fixed

- With planlens 0.16.0, a box with a page unit after it in an answer that
  also holds pixel boxes is still put on the 0-999 grid in the prose, and
  a tiled read's `found` may list it. With the switch on, the answer gets
  the note, which says so; `page_points` never holds it. The planlens
  repair ends this.
- Two known false catches of `states_page_coordinates`, sizes in brackets
  with the unit after: "Font sizes (8, 10) pt" and "Sheet size (1224, 792)
  pt". They give one needless note; nothing is converted wrongly.
- Two limits of `asks_page_positions`, which reads one sentence at a
  time. A unit named in the next sentence ("... with its bbox. Use PDF
  points.") is missed, and that request is put to the model as it was.
  "Shown as points" is wrongly caught: that costs one paragraph of prompt,
  and its `page_points` are still right.
- An identical read repeated in one conversation is served from the repeat
  cache. If the switch flips mid-conversation, which only a notebook can
  do, that read keeps the earlier answer.
- `read_reference_figure`, chart read-offs and `find_like` are not changed.
  They are not asked for page positions in the same way, and the item names
  only the page and zoom tools.

## For the lead: what a measurement on the hosted platform should compare

The suite arm `("pixel_requests", {"GEOTECH_VISION_PIXEL_REQUESTS": "1"})`
against the baseline, same model and questions, on tasks that need
positions on a page: a plan whose symbols must be located (a grid or
transform built from a figure, marks placed on a sheet), and a drawing set
where the agent zooms and marks things. Compare:

- How many side calls the agent asks for page units, and in the arm,
  whether each of them gets `page_points` (this rule's recall).
- For each located symbol, how far the position the agent finally uses is
  from the symbol's true page position (from the text layer or the drawing).
  In the baseline that position is the model's own numbers; in the arm it
  is the converted `page_points`. Also count positions off the page or in a
  frame other than the page's: the item's failure.
- How often `coordinates_note` appears, and whether the agent then uses
  the converted boxes, not the model's numbers.
- How many `calculate` calls or hand conversions the agent spends turning
  boxes into page points (the trace saw 94 for one transform).
- Vision output tokens and seconds per look, and the suite scores, to show
  nothing else moved. Reads that do not ask for page units should be
  identical in both arms.
