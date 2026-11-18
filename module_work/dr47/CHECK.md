# DR-47 and DR-45 CHECK: Word out with photographs, Word in with marks (branch `claude/dr-47-45-word`)

Both items live in one file, `packages/aec-agent-core/aec_agent_core/docx_renderer.py`, so their notes are kept here together. Both are repairs of a defect: on by default, no setting added.

## DR-47: photographs go into a .docx as JPEG at print size

**Asked.** `write_docx` embedded each picture as its file. Photo crops arrive as PNG at 200 to 300 dpi, about 1 MB each; a 20-photo report passed a common 20 MB mail limit and the agent dropped evidence photos to get under it. Store photographs as JPEG at about quality 85, no wider than the page at about 200 dpi; leave figures and drawing crops as PNG.

**Built.** `_picture` asks `_photo_as_jpeg(path, ctx)` first; `None` means "place the file as before". New constants `PHOTO_JPEG_QUALITY = 85` and `PHOTO_MAX_DPI = 200`.
- `_is_photograph(rgb)`: the rule, from the pixels alone, on a nearest-pixel sample of at most 256 pixels a side (`_PHOTO_SAMPLE`), so the sampling makes up no colour. Both must hold: (1) continuous tone, at least one distinct colour per five sampled pixels; (2) no paper, fewer than half the sampled pixels within 32 levels (every channel) of the most common tone.
  - The most common tone is the centre (mean) of the fullest cell on eight grids of cells 32 levels wide, each channel's cells either as they are or shifted by 16 levels. The largest share within 32 levels of such a centre is the one judged. On one of the eight grids any paper tone lies at least 8 levels inside its cell in every channel, so scanner noise of a few levels keeps the paper in one cell (checked up to a standard deviation of 5).
  - Why (review fix): on the single grid first built, a noisy paper tone at a cell edge (about 224 in each channel) split across up to eight cells. A flat coloured zone over a sixth of a scan then outnumbered each fragment and was taken for the paper, so a colour scan that is 80 % paper was converted.
- `_photo_as_jpeg`: only RGB, or RGBA with every pixel opaque; no EXIF rotation; a photograph by the rule. **The size rule as built: the printed size at 200 pixels per inch.** The printed size is python-docx's own reading of the file, fitted to the text width and height as `_picture` already does. A 600 x 400 px photograph printed 2 in wide is stored at 400 x 267 px; a large photograph printed at the full text width is stored at 200 px per inch of that width. A JPEG already within that size is left alone. Resize with Lanczos, save at quality 85 and **full colour resolution (4:4:4, `subsampling=0`)**, with the ICC profile kept and the JPEG's dpi set to its printed size. Used only if smaller than the source file. The picture is placed at the size the original file would have had, so the page looks the same.

**Calls made.**
1. *The photograph rule* (the lead left it to the builder): two cheap pixel measures, thresholds 1/5 and 1/2. Checked on a probe outside the repository:
   - The matplotlib line, filled-contour, image, 3-D surface and dense alpha-scatter charts probed reach at most 0.11 distinct colours per pixel. That holds for charts drawn in flat colours or through a stepped colour map, not for every chart (see the four kinds below). A rendered PDF drawing with text reaches 0.01.
   - Noisy grey and yellowed-paper scans pass test 1 (0.27 to 0.30) and fail test 2 (paper covers about 0.9).
   - Real photographs (stock wallpapers, matplotlib's sample photo) pass both, except smooth wallpapers that are mostly sky, which stay as they are. A sample takes 10 to 35 ms on a 9-megapixel picture.
   - **Photograph-like in their pixels: stored as JPEG at full colour resolution.** A smoothly shaded heat map with labelled contours; a shaded-relief site map with labels; lines and lettering drawn over a photograph (a boring plan on an aerial base); a scan under strongly uneven light. These pass both tests and convert. No rule on pixels tells them from photographs, and none was fitted to them (the lead's ruling); full colour resolution keeps their coloured lettering and symbols from bleeding.
2. *Erring towards leaving a picture alone*: grey pictures (scans of drawings and text are grey; JPEG blurs lettering), palette and CMYK pictures, any transparency, an EXIF rotation (the re-encode would drop it), a JPEG within the size (re-encoding only loses quality), a result no smaller than the file, and any exception.
3. *The size rule is the printed size at 200 pixels per inch, not the page width*: a picture printed narrower than the text keeps 200 px per inch of its own printed width, never more (the 600 x 400 px example above). That is the item's "no wider than the page at about 200 dpi" or less.
4. *The JPEG's dpi equals its printed size*, so `docx_to_markdown` then `write_docx` prints it at the same size again (tested).
5. *No warning line* for a re-encoded picture: `warnings` lists what the reader will not see, and a re-encoded photograph is seen.
6. *Eight grids, not the two the reviewer checked* (review fix): the reviewer's fix used two grids 16 levels apart, all channels shifted together. The eight grids shift each channel on its own, so the guarantee holds for a paper tone near a cell edge in one channel and near the shifted edge in another, such as (224, 208, 224). The eight include the original grid and both of the reviewer's, and the largest share is kept, so a picture that stays as it is under either of those rules stays under this one. On the probe photographs the verdict was the same under one, two and eight grids.
7. *Full colour resolution* (the lead's ruling): `subsampling=0`. On the probe's stock photographs at 1200 x 800 it added 20 to 55 % to the bytes (the lead's estimate was 20 to 34 %).
8. *Report size, stated as it holds*: realistic photographs at 1200 x 800 and quality 85 come to about 180 to 430 KB each, more at full colour resolution. A 20-photograph report is therefore about 4 to 13 MB, not "2 to 3 MB": the aim that holds is well under a 20 MB mail limit with every photograph kept. The made-up photographs in the tests are smoother and store far smaller, so the test's bound is theirs.

**Tests** (`packages/aec-agent-core/tests/test_dr47_photos_in_word.py`, made-up images). The module imports `PHOTO_MAX_DPI`, so on master the whole file errors.
- `test_photo_is_stored_as_jpeg_no_wider_than_the_page_at_200_dpi`: JPEG part, width ≤ text width × 200 px, printed at the full text width with its aspect, source file unchanged, smaller than the source. Fails on master (PNG part).
- `test_photo_with_its_own_dpi_prints_at_the_same_size`: a 300 dpi photo prints 5 in wide, stored 1000 px wide, and still prints 5 in after read-back and rewrite. Fails on master.
- `test_chart_drawing_and_doubtful_pictures_are_left_as_they_are`: a matplotlib chart, a line drawing with lettering, a grey photo, a photo with one transparent pixel and a JPEG within the size are stored byte for byte. Passes on master: it pins that these are untouched.
- `test_order_kept_and_an_unreadable_picture_is_noted_as_before`: photo, unreadable file, chart, photo give JPEG, PNG, JPEG in order, and the unreadable file gets today's warning and note. Fails on master.
- `test_a_twenty_photo_report_keeps_every_photo_and_shrinks`: 20 distinct made-up 1400 × 933 photos, all kept, all JPEG, file under 5 MB. Its comment says the bound holds only for the smooth made-up photographs and gives the range for real ones (call 8). Fails on master (tens of MB).
- `test_photo_is_stored_at_full_colour_resolution` (review fix): reads the stored JPEG's subsampling back, 4:4:4. Fails on the branch before the fix (4:2:0) and on master.
- `test_colour_scans_stay_and_photographs_convert_whatever_the_paper_tone` (review fix): sweeps 30 made-up colour scans: five paper tones at and across the cell edges ((224, 224, 224), (225, 223, 222), (223, 226, 224), (240, 238, 236), (224, 208, 224)), noise of 2, 3 and 5 levels, and a flat zone over 15 or 25 % of the width in a tone at the middle of a cell, with line work, lettering and a scanner's blur. Every scan stays as it is. Five made-up photographs, two of them with a smooth sky over 45 %, all convert. One of the scans goes through `write_docx` and is stored as PNG byte for byte. On the branch before the fix, 9 of the 30 scans converted.

**Noted, not fixed.**
- A grey photograph, or one that is mostly sky, stays PNG (call 2).
- The four kinds of chart and crop that are photograph-like in their pixels convert (call 1), at full colour resolution.
- `write_docx`'s description still says nothing about picture size. Left alone: no fingerprint moves, and with this repair a photo-heavy report no longer comes near a mail limit.

**For the lead, on the hosted platform.** Build one report of about 20 `snip_region` photo crops plus a chart and a drawing crop with `write_docx`, before and after. Compare the .docx size, check that every photo is present, open it in Word, and check that the photos look right at print size and the chart and drawing crop are as sharp as before.

## DR-45: Word in says what it did not read, reads Symbol characters, gives tracked deletions

**Asked.** `read_office` on a Word file silently dropped equations in Word's own format (`m:oMath`) and `w:sym` characters (a Symbol-font phi), and read tracked changes as accepted with nothing said and the deletions gone.

**Built** (`_DocxReader`, `docx_to_markdown`, `office_text._docx`):
- `segments`: `m:oMathPara` (one per `m:oMath` inside it) and a bare `m:oMath` each count one "equation (Word's own equation format)" in the existing not-read section. A run's `w:sym` becomes its character through `_sym`.
- `_sym` and `_SYMBOL_FONT`: Symbol-font codes (the code, or 0xF000 plus it) become their character. Any other font, or a Symbol code with no character in the table, counts one "symbol character in the <font> font" as not read and adds nothing to the text.
- `_symbol_run` and `_run_font` (added at the lead's request, the item's own case of a phi typed in the Symbol font). The run's font comes from its own `w:rFonts` (`w:ascii`, else `w:hAnsi`; `w:cs` first for a complex-script run), else from its character style's.
  - A run set in Symbol has its text turned into the characters printed through the same table (`_symbol_char`: the code or 0xF000 plus it; white space, a non-breaking space included, stays as it is; a character beyond Latin-1 outside 0xF0xx is already Unicode text and stays). This happens only when every character of the run maps.
  - Otherwise, and for a run set in a font from `_SYMBOL_FONTS_NO_TABLE` (Wingdings 1 to 3, Webdings, Zapf Dingbats, Marlett, MT Extra), the text stays as it is and the run counts once as "run set in a symbol font (the letters shown are not the characters printed)".
  - A run with no such font is read as before.
- `_deleted`: a `w:del` stays out of the body as before. Its text (`w:delText`/`w:t`, tabs and breaks as spaces) is kept with its author, date and the paragraph's text before and after it. Neighbouring `w:del`s by one author at one place are joined (Word splits a deletion per run). `w:ins`, `w:moveTo` and `w:moveFrom` are noted as tracked changes; moved text is still read at its new place only.
- `tracked_changes_text()`: when the file holds tracked changes, it builds a section headed "Tracked changes in the Word file" (`TRACKED_CHANGES_HEADING`). The section says "The file holds tracked changes. The text above reads them as accepted: insertions in, deletions out", plus a clause for moves, then one line per deletion: `- J. Doe, 2026-05-01: deleted "soft" after "The footing bears on"`. The anchor is "before ..." when the deletion starts its paragraph, and "(its paragraph has no other text)" when the paragraph has nothing else.
- `docx_to_markdown` returns that section under a new key, `tracked_changes` (`""` when the file holds none). `markdown` stays the body alone, the writer's inverse, exactly as on master.
- `office_text._docx` (the `read_office` path the model reads) puts the section after everything else in the text it returns, and adds its heading to the sections its note names, with the instruction to leave them out when writing back.

**Calls made.**
1. *Source of the Symbol table*: the Unicode Consortium's published "Adobe Symbol Encoding to Unicode" (`unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt`, table version 1.0, 2011-07-12), whose code order the Symbol font follows. It is written into the code as `code:unicode` pairs, generated from that file, not typed.
   - Where the table gives a code two values, the Greek letter is taken (0x6D gives mu, not the micro sign; 0x44 gives Delta; 0x57 gives Omega), otherwise the first listed (0x20 gives space, 0xA4 gives fraction slash).
   - Its 29 private-use-only codes (bracket pieces, extenders, serif and sans-serif marks) and its unassigned codes are counted as not read.
   - No other font has a table here, so Wingdings, Webdings and the like are counted as not read, never guessed.
2. *Deletions apart from the body, not inline* (the lead's ruling): `docx_to_markdown(...)["markdown"]` stays the writer's inverse ("accept all"), so no path that writes that Markdown back can bring deleted text back as current text. Only the model's text (`read_office`) carries the section, last, under its own heading, each line saying "deleted". Symbol characters are the author's text, so they stay in the body. The equation and unknown-symbol counts stay in DR-41's not-read section.
3. *Equations counted, not transcribed*: as charts and embedded objects are (the item's words); the text around them is unchanged.
4. *The section is bounded* (review fix, the lead's ruling): each deleted passage used to be given whole, so a 1,500-word deletion made a 13 KB section for the model.
   - A deleted passage longer than 300 characters (`_DELETION_MAX_CHARS`) is given by its start and its end, about 150 characters each, cut at word breaks. Between them, inside the quotes, it says `[... <count> characters not shown ...]`, and the count is exact. Its author, date and place (before and after context) are kept as before. 300 characters is enough to recognise a passage and find it in the file.
   - The deletion lines stop at 4,000 characters in all (`_DELETIONS_MAX_CHARS`, about a thousand tokens); the first line is always given. One closing line in brackets then says how many more deletions are not shown: `(12 more deletions are not shown here: this list stops at about 4,000 characters.)`. So the count of deletions stays exact.
   - Nothing reads as current text: the clip mark is inside the quoted deleted text, and the closing line is a note in brackets, not a list item.
   - A file whose deletion lines fit within both bounds reads exactly as before the bounds.
5. *White space in a Symbol run is white space* (review fix): a non-breaking space typed in a Symbol-font run read as the table's character at 0xA0 (a euro sign between "f" and "="). `_symbol_char` now keeps any white-space character as it is, as it already kept the plain space. A character at 0xF0A0 (the private-use form) still reads through the table.

**Tests** (`packages/aec-agent-core/tests/test_dr45_word_in_marks.py`, hand-built parts, made-up text):
- `test_symbol_font_characters_are_read`: phi, the degree sign (0xF0B0) and gamma (written as 0x0067) come out in place, with no not-read section. Fails on master (they were dropped).
- `test_text_set_in_the_symbol_font_reads_as_printed`: a typed "f" in a Symbol run reads as phi, 0xF067 as gamma and a run styled with a Symbol character style as sigma; an "f" in Times New Roman stays "f"; there is no not-read section. One assertion added by the review fix: a Symbol run of "f", a non-breaking space and "=" reads as phi, the non-breaking space and "=" (it read with a euro sign in the middle before the fix). Fails on master.
- `test_a_few_short_deletions_read_exactly_as_before_the_bounds` (review fix): three short deletions give the whole section as an exact string. It passed on the branch before the bounds and passes after them; it fails on master, which has no `tracked_changes` key.
- `test_a_long_deletion_is_clipped_with_its_place_kept` (review fix): a deletion of 1,500 words (15,389 characters) is given by its start and its end, with its author, date and place. The count said between them plus the characters shown equals the deletion's length, and the line stays under the bound. The body holds none of it. Fails before the fix.
- `test_many_deletions_are_bounded_and_the_rest_counted` (review fix): of 200 deletions, the lines given total at most 4,000 characters, in order, each exactly as before with its place. The closing line's count plus the lines given equals 200. Fails before the fix.
- `test_symbol_font_runs_without_a_table_are_kept_and_counted`: a Symbol run with a code that has only a private-use value, and a Wingdings run, keep their text, and the section says "2 x run set in a symbol font (the letters shown are not the characters printed)". Fails on master.
- `test_unknown_symbols_and_equations_are_counted_as_not_read`: 3 equations (one inline, two in one `m:oMathPara`), one Wingdings and one private-use Symbol code are counted; the text around them is kept; `read_office`'s note names the section. Fails on master.
- `test_tracked_changes_are_said_and_deletions_given_apart`: on the `read_office` path, the body reads "The footing bears on stiff clay at 2 m." with no deleted text, and the section after it says the file holds tracked changes and gives two deletions: one joined from two runs, with its place, and one whole deleted paragraph. The note names the section. `docx_to_markdown`'s `markdown` has no deleted text and no section; the section is its `tracked_changes`. Fails on master.
- `test_insertions_only_are_said_and_the_body_is_as_before`: `markdown` equals that of the same file with the insertion as plain text. `tracked_changes` says the file holds tracked changes and lists no deletion. `read_office` returns exactly the body plus that section. Fails on master (no section).
- `test_a_file_with_none_of_these_reads_exactly_as_before`: a memo with a heading, bold, a list and a table reads to an exact string with no extras and an empty `tracked_changes`, and `read_office` gives the same text. The empty-key check fails on master (the key is new); the rest pins today's output.

**Who consumes the reader's Markdown (checked at the lead's request).** Nothing in this repository or in planlens reads a Word file to Markdown and then writes a Word file from it on its own. `docx_to_markdown` has one caller, `office_text.read_office`, and that has two:
- `vision_tools` `read_text_file`, whose JSON tool result goes to the model;
- `document_tools._read_office`, which is the review host's `read_office`. planlens' `review/conversation.py` `_read_office` hands its lines to the model through `open_document`.

There are no callers in `calc_package/`, `webapp/`, `funhouse_agent/`, `report_ingest/` or `tinyapps/`, and planlens never calls the writer. The Markdown reaches `write_docx` only when the model writes it.

The lead's ruling still keeps `docx_to_markdown(...)["markdown"]` the writer's inverse, with the tracked-changes section under its own key, added only on the `read_office` path. So a round trip of a file with tracked changes writes exactly what it writes today.

A file with no tracked change, equation or `w:sym` takes none of the new branches, so on both paths it yields exactly what it yields today.

**Existing tests edited: none.** `calc_package/tests/test_docx_roundtrip.py::test_a_word_file_not_written_here_reads_too` (`"gone" not in md` for a file with a tracked deletion) passes untouched.

**Noted, not fixed.**
- A run's font is taken from its own `w:rFonts` and its character style only. A Symbol font set through a paragraph style, the document defaults, a style's base style or a theme font is not followed, and such text reads as before.
- Symbol fonts with no table are known by name (`_SYMBOL_FONTS_NO_TABLE`). A symbol font outside that list is read as plain text, as before. `fontTable.xml`'s symbol charset (02) would be a more general sign.
- Text in a Symbol-font run inside a tracked deletion is given as stored, not turned into the characters printed.
- Deletions in footnotes, comments or headers are listed without naming that part.
- Tracked formatting-only changes (`w:rPrChange`, `w:pPrChange`) are not mentioned, since no text is lost.
- A deleted table row still shows as an empty row, as before.
- Equations' own text is not transcribed (call 3).

**For the lead, on the hosted platform.** Read a real report that has a Symbol-font phi, a Word equation and tracked changes. Check that phi reads in place, that the equation is counted, and that the tracked-changes section gives the deletions apart from the body. Then rewrite the report with `write_docx` and confirm the deleted text does not come back.

## Runs (through the runner; exit codes read directly)
- Root: `test_dr45_word_in_marks.py`, `test_dr47_photos_in_word.py`, `test_dr41_office_files.py`, `test_saved_files_and_settings.py`, `calc_package/tests/test_docx_renderer.py`, `test_docx_roundtrip.py`, `funhouse_agent/tests/test_read_office_files.py`, `funhouse_agent/deep/tests/test_deep_tools_offline.py`, `webapp/tests/test_public_text.py`: 110 passed, with the roundtrip test untouched.
- Root: `test_core_suite_and_records.py`, `test_core_imports.py`, `funhouse_agent/deep/tests/test_deep_tools_offline.py`, `test_review_binding_offline.py`, `webapp/tests/test_core_move_11c.py`: 108 passed.
- Document Review: `tests/test_page_turn.py`, `test_review_fingerprints.py` pass. `test_review_binding_offline.py` has one failure, `test_the_process_host_is_the_apps` (`own_mark_words`); it fails the same way alone and on master (DR-41's CHECK records it), and it reaches no Word code.

After the review fixes (through the runner, each file below and every existing test file that names `docx_renderer`, `write_docx`, `docx_to_markdown` or `office_text`; no file outside the two packages edited):
- Root, exit 0 (214 passed): the core's `test_dr47_photos_in_word.py`, `test_dr45_word_in_marks.py`, `test_dr41_office_files.py`, `test_saved_files_and_settings.py`, `test_app_details.py`, `test_core_imports.py`, `test_core_suite_and_records.py`; `calc_package/tests/test_docx_renderer.py`, `test_docx_roundtrip.py`; `funhouse_agent/tests/test_cited_documents.py`, `test_read_office_files.py`, `test_writer_output_paths.py`.
- Root, exit 0 (150 passed, 5 skipped): `funhouse_agent/deep/tests/test_deep_tools_offline.py`, `test_review_binding_offline.py`, `test_review_harness_offline.py`, `test_review_minimal_offline.py`; `funhouse_agent/review_eval/tests/test_review_eval_delivered_offline.py`; `webapp/tests/test_app_details.py`, `test_core_move_11c.py`, `test_review_boundary.py`, `test_tinyapps_entry.py`, `test_wheels.py`.
- Document Review, exit 0 (29 passed): `tests/test_dr43_attachment_note.py`, `test_page_turn.py`, `test_review_binding_offline.py` (its earlier failure no longer shows), `test_review_eval_delivered_offline.py`, `test_review_minimal_offline.py`, `test_review_fingerprints.py`.
- `webapp/tests/test_public_text.py` before the notes commit.

## Fingerprints
Not moved: no tool name, description, schema or prompt touched; `test_review_fingerprints.py` passes. Not re-recorded.
