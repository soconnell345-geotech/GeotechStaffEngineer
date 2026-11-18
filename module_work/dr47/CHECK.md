# DR-47 and DR-45 CHECK: Word out with photographs, Word in with marks (branch `claude/dr-47-45-word`)

Both items live in one file, `packages/aec-agent-core/aec_agent_core/docx_renderer.py`, so their notes are kept here together. Both are repairs of a defect: on by default, no setting added.

## DR-47: photographs go into a .docx as JPEG at print size

**Asked.** `write_docx` embedded each picture as its file. Photo crops arrive as PNG at 200 to 300 dpi, about 1 MB each; a 20-photo report passed a common 20 MB mail limit and the agent dropped evidence photos to get under it. Store photographs as JPEG at about quality 85, no wider than the page at about 200 dpi; leave figures and drawing crops as PNG.

**Built.** `_picture` asks `_photo_as_jpeg(path, ctx)` first; `None` means "place the file as before". New constants `PHOTO_JPEG_QUALITY = 85` and `PHOTO_MAX_DPI = 200`.
- `_is_photograph(rgb)`: the rule, from the pixels alone, on a nearest-pixel sample of at most 256 pixels a side (`_PHOTO_SAMPLE`), so the sampling makes up no colour. Both must hold: (1) continuous tone, at least one distinct colour per five sampled pixels; (2) no paper, fewer than half the sampled pixels within 32 levels (every channel) of the most common tone.
- `_photo_as_jpeg`: only RGB, or RGBA with every pixel opaque; no EXIF rotation; a photograph by the rule. The size cap is the size the picture is printed at (python-docx's own reading of the file, fitted to the text width and height as `_picture` already does) times 200 px per inch. A JPEG already within that size is left alone. Resize with Lanczos, save at quality 85 with the ICC profile kept and the JPEG's dpi set to its printed size. Used only if smaller than the source file. The picture is placed at the size the original file would have had, so the page looks the same.

**Calls made.**
1. *The photograph rule* (the lead left it to the builder): two cheap pixel measures, thresholds 1/5 and 1/2. Checked on a probe outside the repository: matplotlib line, filled-contour, image, 3-D surface and dense alpha-scatter charts reach at most 0.11 distinct colours per pixel. A rendered PDF drawing with text reaches 0.01. Noisy grey and yellowed-paper scans pass test 1 (0.27 to 0.30) and fail test 2 (paper covers about 0.9). Real photographs (stock wallpapers, matplotlib's sample photo) pass both, except two smooth wallpapers that are mostly sky, which stay as they are. A sample takes 10 to 35 ms on a 9-megapixel picture.
2. *Erring towards leaving a picture alone*: grey pictures (scans of drawings and text are grey; JPEG blurs lettering), palette and CMYK pictures, any transparency, an EXIF rotation (the re-encode would drop it), a JPEG within the size (re-encoding only loses quality), a result no smaller than the file, and any exception.
3. *Cap on printed size, not on page width*: a picture printed narrower than the text keeps 200 px per inch of its own printed width, never more. That is the item's "no wider than the page at about 200 dpi" or less.
4. *The JPEG's dpi equals its printed size*, so `docx_to_markdown` then `write_docx` prints it at the same size again (tested).
5. *No warning line* for a re-encoded picture: `warnings` lists what the reader will not see, and a re-encoded photograph is seen.

**Tests** (`packages/aec-agent-core/tests/test_dr47_photos_in_word.py`, made-up images). The module imports `PHOTO_MAX_DPI`, so on master the whole file errors.
- `test_photo_is_stored_as_jpeg_no_wider_than_the_page_at_200_dpi`: JPEG part, width ≤ text width × 200 px, printed at the full text width with its aspect, source file unchanged, smaller than the source. Fails on master (PNG part).
- `test_photo_with_its_own_dpi_prints_at_the_same_size`: a 300 dpi photo prints 5 in wide, stored 1000 px wide, and still prints 5 in after read-back and rewrite. Fails on master.
- `test_chart_drawing_and_doubtful_pictures_are_left_as_they_are`: a matplotlib chart, a line drawing with lettering, a grey photo, a photo with one transparent pixel and a JPEG within the size are stored byte for byte. Passes on master: it pins that these are untouched.
- `test_order_kept_and_an_unreadable_picture_is_noted_as_before`: photo, unreadable file, chart, photo give JPEG, PNG, JPEG in order, and the unreadable file gets today's warning and note. Fails on master.
- `test_a_twenty_photo_report_stays_under_3_mb`: 20 distinct 1400 × 933 photos, all kept, all JPEG, file under 3 MB. Fails on master (tens of MB).

**Noted, not fixed.**
- A grey photograph, or one that is mostly sky, stays PNG (call 2).
- `write_docx`'s description still says nothing about picture size. Left alone: no fingerprint moves, and with this repair a photo-heavy report no longer comes near a mail limit.

**For the lead, on the hosted platform.** Build one report of about 20 `snip_region` photo crops plus a chart and a drawing crop with `write_docx`, before and after. Compare the .docx size, check that every photo is present, open it in Word, and check that the photos look right at print size and the chart and drawing crop are as sharp as before.

## DR-45: Word in says what it did not read, reads Symbol characters, gives tracked deletions

**Asked.** `read_office` on a Word file silently dropped equations in Word's own format (`m:oMath`) and `w:sym` characters (a Symbol-font phi), and read tracked changes as accepted with nothing said and the deletions gone.

**Built** (`_DocxReader`, `docx_to_markdown`, `office_text._docx`):
- `segments`: `m:oMathPara` (one per `m:oMath` inside it) and a bare `m:oMath` each count one "equation (Word's own equation format)" in the existing not-read section. A run's `w:sym` becomes its character through `_sym`.
- `_sym` and `_SYMBOL_FONT`: Symbol-font codes (the code, or 0xF000 plus it) become their character. Any other font, or a Symbol code with no character in the table, counts one "symbol character in the <font> font" as not read and adds nothing to the text.
- `_symbol_run` and `_run_font` (added at the lead's request, the item's own case of a phi typed in the Symbol font). The run's font comes from its own `w:rFonts` (`w:ascii`, else `w:hAnsi`; `w:cs` first for a complex-script run), else from its character style's.
  - A run set in Symbol has its text turned into the characters printed through the same table (`_symbol_char`: the code or 0xF000 plus it; a space stays a space; a character beyond Latin-1 outside 0xF0xx is already Unicode text and stays). This happens only when every character of the run maps.
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
4. *No cap on the number of deletion lines*: the comments section has none either.

**Tests** (`packages/aec-agent-core/tests/test_dr45_word_in_marks.py`, hand-built parts, made-up text):
- `test_symbol_font_characters_are_read`: phi, the degree sign (0xF0B0) and gamma (written as 0x0067) come out in place, with no not-read section. Fails on master (they were dropped).
- `test_text_set_in_the_symbol_font_reads_as_printed`: a typed "f" in a Symbol run reads as phi, 0xF067 as gamma and a run styled with a Symbol character style as sigma; an "f" in Times New Roman stays "f"; there is no not-read section. Fails on master.
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

## Fingerprints
Not moved: no tool name, description, schema or prompt touched; `test_review_fingerprints.py` passes. Not re-recorded.
