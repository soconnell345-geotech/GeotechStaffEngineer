# Notes on the blind attempt

I worked only from the six PDFs and questions.md in the blind folder, using PyMuPDF and the standard library. Scripts and images are in `blind\work\`. I looked at nothing outside the folder and needed no outside context.

## 1. How each question was worked, and how sure I am

The method was the same throughout. I extracted every text span with its position (`work\*.spans.tsv`, plus a compact per-sheet dump in `*.compact.txt`), then paired callout bubbles by position: the view number directly above the sheet number, centres within 8 pt. I checked the results on zoomed renders of the sheets. The text layer is clean: every label on every rendered image also appears in the text.

- **coord-sheet-index: text, plus a check of every sheet's title block. Very sure.**
  - A-202 is missing from the set.
  - S-201 is titled wrongly in the index.
- **coord-callouts: text pairing by position, with images of A-101, A-102, A-301 and A-401. Very sure.**
  - 7/A-501 points to a detail that doesn't exist.
  - 2/A-302 points to a sheet that doesn't exist.
- **coord-schedules: text, plus images of A-101, A-102 and S-101. Very sure.**
  - Doors: 109 is on the plans but not scheduled; 204 is scheduled but not on the plans.
  - Footings: F4 is on the plan but not scheduled. Its outline measures 2600 mm square from the vector rectangle at 1:50.
- **coord-levels-dimensions: text, plus scaled vector and text positions. Very sure.**
  - A-301 gives the second floor as 104.50, against 104.20 elsewhere. Its level line sits at the same height as A-201's, which is 4.20 m at 1:100.
  - A-401 dimensions grid B to C as 7500 while the grid lines are drawn 816 pt apart, which is 7200 mm at 1:25.
- **coord-drawings-vs-manual: text. Very sure.**
  - Footing concrete is 25 MPa on the drawings against 30 MPa in the manual.
  - The reference to 07 84 00 does not resolve.
- **coord-drawings-vs-geotech: text, plus elevation arithmetic. Very sure of the conflicts, fairly sure the omissions are worth listing.**
  - Conflicts: 200 against 150 kPa, and 1.2 against 1.5 m.
  - The omissions (fill removal, bearing stratum, inspection, shoring) are differences of omission. The question asked for "every difference", so I listed them separately from the conflicts.
- **coord-revisions: automated diff of all text spans and all vector drawings between issue 0 and issue 1, sheet by sheet, with the clouds and deltas checked on images. Very sure the list of changes is complete for text and linework.**
- **coord-submittal: text and arithmetic. Very sure.**
  - Firm findings: w/cm 0.48 against 0.45, fly ash 30 % against 25 %, 12 tests against 30, aggregate reports missing, and "deviations: none".
  - The 10-day gap in the strength record is put as a query, not a finding.
- **coord-full-review:** the union of the 15-sheet findings above, limited to its three documents. It does not rely on issue 0, so it makes no claim that S-001 changed without a revision. Very sure.
- **coord-long-callouts: same method as coord-callouts. Very sure.** The extra error in the 40-sheet set is 9/A-502 on A-114.
- **coord-long-schedules: text and images (A-104, M-103, E-104). Very sure.** Its extra errors are doors 407 and 509, FCU-3-4 and LP-4B.
- **coord-long-review:** the 40-sheet union. The extra items are A-301's fifth floor at 116.50 against 116.80, plus the long-set schedule and callout errors. I also diffed every 15-sheet page against its 40-sheet counterpart, to confirm the shared sheets carry the same errors. Very sure.

## 2. Wording I had to interpret

- **"Callouts":** I counted section, detail and enlarged-plan bubbles that point elsewhere. View-title bubbles are the targets, not callouts; I checked only that each carries its own sheet number. I judged "points to a view that exists" on existence only, not on whether the detail suits the location (see section 3).
- **"Every schedule" (long set):** I took this as door, footing, mechanical equipment and panel schedules. The sheet index is also a schedule, but it lists sheets, not plan marks, so it is covered in the reviews. Hardware sets are checked against the specification, not the plans.
- **Floor levels:** I treated "T.O. SLAB" on the S-sheets as equal to the architectural floor level, since they match everywhere except A-301.
- **Revisions:** I counted the G-001 index REV-column updates as a change on G-001, and noted the issue-line change on every sheet as administrative.
- **"Every difference" (geotech):** I split this into contradictions and memo requirements not carried onto the drawings, and also listed what agrees.
- **Comment logs:** one row per problem. Where one problem shows on two sheets (door 109 on A-101 and A-401), I gave it one row naming both sheets.

## 3. Things that looked like inconsistencies but that I chose not to report

- **Callouts that exist but suit their location oddly.**
  - On A-401, 1/A-501 "SILL AT STOREFRONT" is called on the interior core wall.
  - On A-301, 2/A-501 "PARAPET AT ROOF EDGE" is called near the second-floor line.
  - In the 40-sheet A-301, 6/A-501 "ROOF DRAIN" sits near the third floor.
  - On A-103, 1/A-501 (storefront sill) is called on a third-floor plan; on A-104, 2/A-501 (parapet) is called on a fourth-floor plan.
  - I didn't report these because the questions ask about existence, and every detail on A-501, A-502 and S-501 carries the same generic placeholder notes (vapour retarder, grout frame solid and so on), so the detail content can't be judged.
- **Boring locations.** On C-101 at 1:200, B-1 to B-4 sit about 4 to 7 m outside the building corners. The memo says the borings were drilled "at the corners of the building". This is a scaled measurement on a "do not scale" set and is probably just schematic placement, so I'm not confident it's real.
- **The manual is incomplete.** Its table of contents lists 7 sections, but only 01 33 00, 03 30 00 and 08 71 00 are bound in. I mentioned this as a scope limit in the manual answer, not as a finding.
- **Work with no specification section.** Structural steel, metal deck, insulation, gypsum board, curtain wall and storefront, acoustic ceilings, earthwork, mechanical and electrical appear on the drawings with no section in the manual. The table of contents says unlisted sections are not part of the manual. This is neither a conflict nor a reference, so I didn't report it.
- **Electrical rooms (40-sheet set).** E-601 locates the panels in a "FIRST/SECOND/... FLOOR ELECTRICAL ROOM", but no electrical room exists on the architectural plans. The panels are drawn inside the core, where A-401 shows Stair 1 and the toilet. AHU-1 sits in STORAGE on M-101. This is real but may be schematic; I left it out to avoid a soft comment.
- **Footing sizes in the 5-storey set.** The 40-sheet set has the same footing schedule as the 2-storey set: F3 is 2.2 m square. A rough tributary-load estimate gives about 1300 kN at interior column B/2: 35.6 m2 per level, four floors at about 8 kPa plus the roof. F3 carries only about 730 kN at 150 kPa, or 970 kN even at 200 kPa. No loads are given on the drawings and this is a design check, not coordination, so I left it out of the log. **A structural engineer should look at it.**
- **Missing doors and elevations.** The core has no door on floors 2 to 5 and the toilet has none on any floor, and A-201 shows only the south and east elevations. These look like schematic simplifications, not coordination errors.
- **Submittal strength margin.** The record averages 40.9 MPa for a 35 MPa mix, which would be thin against a usual required average strength for a small record. The specification sets no required average, so I didn't report it. The concrete yield doesn't sum to exactly 1 m3 with assumed specific gravities; that depends on the assumptions, so I didn't report it either.
- **Door 107 delta position (A-601).** The delta sits next to row 106, just above the cloud around 107's HW-2. That is normal placement, not an error.

## 4. Anything that got in the way

Nothing significant. The PDFs have a complete, clean text layer, no images and no annotations. The clouds are short vector line segments, so a vector diff plus zoomed images settled every revision question. The only care needed was pairing callout numbers with sheet numbers by position, since the text is in no useful order.
