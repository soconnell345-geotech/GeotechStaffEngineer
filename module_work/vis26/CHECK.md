# VIS-26: image files, the rest — check notes

## What the item asked

When the document is an image file (a phone photograph, a pasted
screenshot, a scan saved as PNG) and not a PDF, five things were still
wrong: `measure` and `log_grid` worked in planlens' page points while the
agent's positions on an image are its pixels; planlens' view-anchor rule
refused a small mark from any image view wider than 300 pixels, in words
about points; `kept_readings` listed an image zoom as pdf_page 1 with no
word that its box is in pixels; the minimal arm's prompt said an uploaded
picture "is shown" and its `page_count` failed on an image; a marked copy
re-used for a different image under the same name was converted with the
wrong size. The lead's ruling: each is the repair of a defect, on, no
switch; a PDF's results byte for byte as before; the default build's
fingerprint does not move.

## What was built, and where

**1. `measure` and `log_grid` (core, `aec_agent_core/measure_tool.py`).**
`_image_frame(entry)` finds, for an image's open entry, its size as a
viewer shows it (`image_view.shown_size`, EXIF applied) and the points a
pixel of planlens' page of it (from the page itself: 0.75 with no dpi
declared, 72/dpi otherwise). `run_measure` puts `bbox`, `to` and a look's
`view` + `image_box` (all in the image's pixels) on that page, scales the
location-error pad, says the location note in pixels
(`LOCATED_IMAGE_NOTE`) and adds `frame`: the result's own positions are
that page's points and how to turn them into pixels. `run_log_grid` adds
the same `frame` on an image (`_log_grid_result` is the old body). A PDF
entry gives `_image_frame` = None and every line runs as before.

**2. The view-anchor rule (planlens `planlens/review/marked_copy.py`,
`host.py`; core `document_tools.host_for`).** planlens' `ImageSupport`
gains an optional `sent_scale(w, h)`; `judge_image_views` judges each
image mark anchored on a view: a view sent at 1 or more is placed (handed
to the writer as the `bbox` its view + image_box name, converted as the
writer converts them, so the written row is the same); a view sent shrunk
is refused as before. Refusals of image marks (the width rule and the
whole-view rule) are reworded in pixels, with the scale the view went at.
The core's `host_for(engine)` sets `sent_scale` from
`image_view.sent_size` for that engine, only when the installed planlens'
`ImageSupport` has the field (`_takes_sent_scale`).

**3. `kept_readings` (core, `vision_tools._kept_ref`).** A kept reading
whose result carries `image_px` (only an image look's does) is listed with
`frame` = `KEPT_IMAGE_FRAME`: pdf_page 1 is the whole image and a bbox is
in its own pixels.

**4. The minimal arm (`aec_document_review/minimal_agent.py`).** The
prompt sentence now says the picture is not shown whole: `analyze_image`
has a vision call answer, and `render_region` shows a zoom (both true of
the code: `vision_kit` inlines an image FILE by path only, an upload keeps
the vision call; a zoom on an upload is inlined). `page_count` on an image
returns `{source, pages: 1, image_px, note}`; a PDF's result and a
non-image's error are as before.

**5. A copy of another image (planlens `marked_copy._destination`,
`_copy_fits_image`).** With append on, an existing copy is added to only
when its page holds an image of this image's pixels (as shown, or as
stored before an EXIF turn) drawn at one scale; otherwise the marks go on a
new copy under the next free name (`_2`, ...) with a `saved_elsewhere`
note. `remove` with marks on such a copy writes nothing and says why.

## Which parts need the new planlens, and what happens with 0.16.0

| Part | With planlens 0.16.0 (installed today) | With this branch's planlens |
|---|---|---|
| 1 measure / log_grid | fixed (core only) | fixed |
| 2 view rule | unchanged: a wide image view is still refused by its width, with the writer's "pt" words; `host_for` sets nothing | fixed |
| 3 kept_readings | fixed (core only) | fixed |
| 4 minimal arm | fixed (DR only) | fixed |
| 5 copy of another image | unchanged (the old wrong conversion) | fixed |

`test_a_mark_on_a_wide_image_view_sent_whole` and
`test_a_build_host_tells_planlens_the_scale_it_sends_at` check both cases
(the second skips with 0.16.0 after checking the host is unchanged).

## Calls made

- **measure's output stays in planlens' page points, with a `frame` note,
  instead of every position converted back to pixels.** A result carries
  positions under many keys (`where`, `pad_pt`, `snapped_to`, `at_pt`,
  `position_pt`, `length_pt`, scale extents, label boxes, `per_point`),
  several named "pt"; converting them all is a redesign of how image units
  are carried, past the repair. The measured VALUE was the defect (the
  position was put at 0.75 or 0.24 of where the agent meant) and is right
  now; the note says exactly how the positions read.
- **`bbox` on an image is taken in pixels**, as `open_document`'s
  coordinates line says for an image, not only a look's view + image_box.
  The note says so in every result, so an agent that passed points sees it.
- **The page's shape decides**: a page whose scale across and down differ
  by more than 1 % (a reader that did not apply the EXIF turn) is refused,
  never guessed. PyMuPDF 1.27 applies the turn (checked with a JPEG made
  with an EXIF orientation tag).
- **A view sent at full resolution or larger may anchor a mark, however
  wide**: the core's own precision line already says such a box is "good to
  a few of the image's pixels". A view of 300 px or less is never refused
  for its width (as on a page), and the refusals keep the writer's index
  order (the check maps written rows by it).
- **kept_readings keeps pdf_page 1 for an image** (the number every tool
  takes for an image's one page, and the way to have the zoom back) and
  adds the `frame` beside it, rather than a new way to fetch.
- **A copy of another image goes to `_2`, not over the first copy**: the
  first image's marks are kept, as `_destination` already does for a file
  that is not this host's copy.

## Settings

None added. Every part is a repair, on.

## What moved for the model

- The DEFAULT build: nothing (`test_review_fingerprints.py`: only
  `review_minimal` differs).
- The minimal build: its system prompt (`/calls[0]/system_prompt`, 1,637 to
  1,753 characters). Not re-recorded here, for the lead.
- Results on an image only: `measure` (`frame`, the location note in px,
  the converted positions), `log_grid` (`frame`), `kept_readings` (`frame`
  on an image's row), the minimal arm's `page_count`, and with the new
  planlens `annotate_document`'s decisions and reasons on image views and
  a copy of another image.

## Tests

Core and Document Review, through the runner, with planlens 0.16.0 and
with this branch's planlens:

- `packages/aec-agent-core/tests/test_vis26_image_files.py`: `measure` with
  a pixel bbox on a PNG (192 dpi declared, 0.375 pt a pixel) and on a JPEG
  stored turned with EXIF orientation 6 reads the right depth; a zoom's
  view + image_box in pixels snaps, its pad converted and its note in
  pixels; a page of another shape is refused; `log_grid` on the PNG reads
  every layer and says its frame; the same PDF calls with the image path
  taken out give the same bytes (measure and log_grid); a build's host
  sets `sent_scale` (with the new planlens) or leaves the host as it was
  (0.16.0); a mark on a wide screenshot view is placed (new planlens) or
  refused as before (0.16.0); a kept image zoom is listed with its frame
  and a PDF zoom's row keys are unchanged.
- `packages/aec-document-review/tests/test_vis26_minimal_images.py`: the
  prompt sentence; `page_count` on a turned JPEG (shown size) and a PNG;
  on a PDF byte for byte as before; on neither still an error.
- Each new test fails on the code before this branch (checked with the
  code stashed); the two PDF guards fail there only because the name they
  patch (`_image_frame`) did not exist yet.
- Also run: `funhouse_agent/deep/tests/test_measure_tool_offline.py`,
  `test_kept_readings_offline.py`, `test_review_binding_offline.py`,
  `test_document_tools_offline.py`, `test_markup_check_offline.py`,
  `webapp/tests/test_page_fingerprints.py`, `test_core_move_11c.py`,
  `packages/aec-agent-core/tests/test_core_imports.py`, Document Review's
  `test_review_minimal_offline.py`, `test_review_binding_offline.py`,
  `test_review_harness_offline.py`, `test_review_fingerprints.py` (only
  `review_minimal` fails, as above).

## Noted, not fixed

- Other planlens tools' boxes on an image (`search_document`,
  `read_document`, the page map) are still in its page points while
  `open_document` says pixels: the same redesign as measure's output.
- A different image of the SAME size under the same name is still added to
  its copy (only the size can be told from the copy).
- The app's own copy of the minimal arm (`funhouse_agent/deep/minimal_agent.py`)
  is quiet in this repository and keeps the old sentence.
- The core's image precision line does not yet say "a mark may be anchored
  on this view" for a view sent at full resolution; with the new planlens
  that is now true.

## For the lead: the live check

The offline parts are tested with made-up images; the item's live check
(LIVE_TEST_QUEUE check 9's extension) is what only a model can show:
1. A phone photo JPEG with EXIF rotation, one ring asked for: confirmed by
   the check first time, the marked PDF's page the photo's pixels upright;
   now also a ring from the whole photo's view when it went whole.
2. `find_like` on a pasted screenshot with a repeated tag: every copy, in
   its pixels.
3. New with this item: `measure` on a scanned log saved as PNG or JPEG
   from a look's view + image_box gives the same depth as on the PDF of it.
On the hosted platform, compare the rollout model on the same photo with
planlens 0.16.0 and with this planlens: how many ring calls are refused
for the view's width, and whether the placed rings are confirmed.
