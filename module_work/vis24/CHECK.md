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
  `answer_boxes` labels one, with the words before it on its line. A tiled
  read takes its merged `found` rows instead, each with `seen_in`. The
  result gets `page_points` (at most `MAX_PAGE_POINTS` = 24; then
  `page_points_cut`): each row has `what`, `page_bbox` and `centre`, in PDF
  points, origin top-left, y down: the frame `bbox` takes. It also gets
  `page_points_note` (`PAGE_POINTS_NOTE`), which says how the points were
  made and that a box is not a measurement. If no pixel box came back, the
  result gets `NO_PAGE_POINTS_NOTE` instead.
- **Half 2, the note.** `vision_tools.states_page_coordinates(text)` reads
  the answer as the model wrote it, before the tool converts its boxes. A
  statement is two or four numbers in brackets, or x and y given
  separately. It needs a page unit right after the numbers ("(612, 396)
  pt", "[..] in PDF points") or the page's frame named just before them in
  the same sentence ("PDF points: (612, 396)", "page coordinates [..]").
  Pixel boxes are excluded, as are sizes ("(8, 10) pt lettering"),
  dimensions ("1224 x 792 pt"), stations, elevations, sheet numbers,
  printed survey coordinates and boxes with no unit. When the answer (or
  any tile of a tiled read) states page coordinates, the result carries
  `coordinates_note` (`COORDINATES_NOTE`) once. The note says the tool did
  not convert those numbers, that they are the vision model's own estimate
  and not measured, and that they may be in another frame. It points to the
  boxes the tool did convert. A tile marks itself with a private
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
- **Only boxes tagged as pixels become `page_points`.** An untagged box is
  converted to the 0-999 grid in the prose when the answer shows it is in
  pixels, as before. It is not listed as a page point: it may be the
  model's own page numbers, and listing it would pass them on as converted.
- **Region-sized boxes are kept in `page_points`.** `answer_boxes` (and so
  `found`) drops boxes over 300 of 999. A zoom on one symbol, asked for its
  centre, boxes a symbol that fills most of the window, and that is exactly
  the case the item names. A tiled read uses `found` as it stands: its
  tiles make things small.
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
- `test_page_coordinates_in_prose_are_found` (8) and
  `test_ordinary_numbers_are_not_page_coordinates` (15: stations,
  elevations, sheet numbers, scales, dimensions in points, sizes in points,
  printed survey coordinates, pixel boxes, boxes with no unit, points on a
  chart).
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
  tiled read lists its `found` row with `seen_in`, also within 1.5 pt.
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

## Noted, not fixed

- A four-number box that the model tags in points, in an answer that also
  holds pixel boxes, is still converted to the 0-999 grid as if it were
  pixels (planlens' `boxes_to_grid` reads only px tags). With the switch on,
  that answer gets the note, so the numbers are not passed on as measured.
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
