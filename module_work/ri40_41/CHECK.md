# RI-41 and RI-40: DIGGS 2.5.a on pydiggs 1.x, and the vertical datum

Branch `claude/ri-40-41-diggs`, 2026-11-05. Commits: RI-41 `f7b6a4e`,
RI-40 `94fa8ea`.

## RI-41: what pydiggs 1.x ships

CI installs pydiggs with no pin (`pip install pydiggs || true` in
`tests.yml` and `nightly.yml`), so it resolves the newest release: **1.4.1**
today, 1.1.1 when the item was filed. I looked inside every 1.x wheel on PyPI
(1.0.0, 1.1.1, 1.2.0, 1.3.0, 1.4.0, 1.4.1) and in 0.1.5:

| pydiggs | `schemas/diggs-schema-2.5.a/Complete.xsd` | `dictionaries/properties.xml` | `diggs-schema-2.5.a/Example_2-DIGGS.XML` |
|---|---|---|---|
| 0.1.5 | yes | yes | yes |
| 1.0.0 to 1.4.1 | yes | yes | **no** |

So **the item was wrong about one thing**: pydiggs 1.x DOES ship the 2.5.a
schema. The only thing 1.x dropped is the **sample file** `Example_2-DIGGS.XML`,
which 0.1.x kept in the 2.5.a folder. That sample is what the CI test read at
63c3331 (`test_the_samples_own_form_reads_longitude_first_with_its_height`),
and it already skips when the file is missing.

What I checked on 1.1.1 and 1.4.1:
- `get_schema_path("2.5.a")` and `get_dictionary_path()` both find their
  files. The dictionary path the code uses (`pydiggs/dictionaries/properties.xml`)
  is right for 1.x.
- `validate_diggs_schema(..., schema_version="2.5.a")` runs pydiggs'
  `validator(instance_path=, schema_path=, output_log=)` and `schema_check()`
  as before, and the check is real: a made-up minimal 2.5.a document passes,
  a 2.6 document fails the 2.5.a schema, and a bare `<Diggs/>` fails with
  libxml2's errors.
- `validate_diggs_dictionary` passes on the writer's output (1.4.1).

## RI-41: the decision

**2.5.a stays an option.** It can be validated with what pydiggs 1.x ships.
The only change is in the message: if a pydiggs build is ever missing the
2.5.a schema or the dictionary, the code now says what is missing, from which
pydiggs version, and that the 2.6 check is bundled and needs no pydiggs. Before,
it raised a bare "Schema file not found: <path>". The 2.6 path is unchanged.

## RI-40: the list and what changed

DIGGS publishes its coordinate-system dictionary in the public repository
**DIGGSml/def**, served at `https://diggsml.org/def` (pydiggs' own
`dictionaries/NOTICE` names that repository as the source of its
dictionaries). The compound list is
`docs/crs/DIGGS/0.1/compoundCRS.xml` (commit `7bd4c574d874`, 2026-11-02):
2,896 compounds over 700 horizontal systems. Only four are over WGS 84 (4326):

| id | name in the dictionary | height unit (EPSG) |
|---|---|---|
| `4326_5703` | WGS 84 + North American Vertical Datum of 1988 | metre |
| `4326_5713` | WGS 84 + Canadian Vertical Datum of 1928 | metre |
| `4326_5714` | WGS 84 + mean sea level height | metre |
| `4326_5702` | WGS 84 + National Geodetic Vertical Datum of 1929 | **US survey foot** |

The form of the name: DIGGS' own CRS check (`docs/validation/modules/crs-validation.xsl`,
`diggs-functions.xsl`) resolves an srsName as a whitelisted URL
(`https://diggsml.org` is on the whitelist) with the entry's `gml:id` as the
fragment. The old `urn:diggs:def:crs:DIGGS:0.1:4326_5702` form in pydiggs'
0.1.x sample would not pass that check. The writer therefore writes, for example:

    https://diggsml.org/def/crs/DIGGS/0.1/compoundCRS.xml#diggs-crs-4326_5703

**What changed** (`report_ingest/diggs_writer.py`):
- A point **with a height**, whose horizontal system is plain WGS 84 (the
  CRS84 case) and whose record prints **NAVD 88, CGVD 28 or mean sea level**,
  is written `srsDimension="3"`, `longitude latitude height`, under that
  compound. The reference point's description quotes the printed datum and
  the dictionary's name.
- Longitude first, as DIGGS' own example writes these names and as the reader
  already read them since RI-30.
- Only the three metre-height names are committed (`DIGGS_COMPOUNDS`), with
  the source URL and commit.
- **The 2-D form is kept** (CRS84, the height in the centre line, the datum
  in the description) when:
  - no datum is printed;
  - the datum is a local benchmark;
  - two datums are named at once (e.g. a conversion note);
  - the system is projected;
  - there is no height;
  - the datum is **NGVD 29**. Its compound exists, but its height is in US
    survey feet and the writer's heights are metres. Naming it would need the
    reader to convert too.
- A named datum wins over "MSL" beside it ("NAVD88 (approximately MSL)" ->
  NAVD 88).
- `subsurface_characterization/plots_plan.py` labels the new form as
  geographic. The parser comment names the form. The reader needed no change:
  it takes the height from a 3-value position and does not swap these axes.

Noted, not fixed: NGVD 29 stays two-dimensional (see above). The DIGGSml/def
README says MIT, but the repository has no LICENSE file (pydiggs' NOTICE says
the same). Only three short names and a URL are copied.

## Tests

Run here on Python 3.11 with pydiggs 1.4.1, and the DIGGS tests again with
1.1.1 on `PYTHONPATH`. Each result is pytest's own exit code.

- `subsurface_characterization -k diggs`: 103 passed (1.4.1); 103 passed (1.1.1).
- New `TestSchema25aOnPydiggs1x`, in `subsurface_characterization/tests/test_formats_diggs_validation.py`:
  - the schema and dictionary are found;
  - a valid 2.5.a document passes;
  - a 2.6 document fails the 2.5.a check, and the reverse;
  - a build without the files is refused with the new messages, while 2.6
    still validates.
- The existing skip for the missing `Example_2-DIGGS.XML` is unchanged and
  fires on 1.x.
- `report_ingest/tests/test_diggs_writer.py`:
  - new `TestAPrintedVerticalDatum` (six datum spellings to three compounds,
    each through the bundled 2.6 schema, the pydiggs dictionary check, the
    reader and the round-trip gate; feet converted to metres; no height,
    projected);
  - the RI-30 "never claims a datum" test now runs over: none, a benchmark,
    NGVD 29, and two datums.
- `report_ingest` + `subsurface_characterization`, whole: 4,644 passed, 60
  skipped, and 1 collection error, `test_library_agent.py`. That error is
  `langchain_core` not installed in this environment; the test has nothing to
  do with DIGGS.
- `funhouse_agent` 2.5.a and subsurface adapter tests: 21 passed.
- `webapp/tests/test_public_text.py`: 4 passed, before each push.

## Release note (three lines)

- DIGGS 2.5.a validation works with every pydiggs 1.x (they all ship its schema). A pydiggs build without the 2.5.a schema or the dictionary is now refused with a message saying what is missing.
- Report ingest's DIGGS files: a boring with a height over WGS 84, from a report that prints NAVD 88, CGVD 28 or mean sea level, is written in 3-D. Its coordinate system is DIGGS' compound for that datum (from DIGGS' CRS dictionary), so **the reference point's srsName and dimension change** for those reports.
- Every other case (no datum, a local benchmark, NGVD 29 (in feet), a projected system) keeps the 2-D point with the height in the centre line.
