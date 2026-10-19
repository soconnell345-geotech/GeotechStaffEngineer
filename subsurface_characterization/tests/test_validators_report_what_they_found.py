"""GM-160 (found by the method probes on eval_samples/): the AGS4 and DIGGS
dictionary checks report what their libraries found. python-ags4 counts as
a tuple, which was read as no errors (every file valid); pydiggs logs a
pass too, which was read as a failure."""

from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "funhouse_agent" / \
    "eval_samples"


def test_an_ags4_file_with_errors_is_not_valid():
    pytest.importorskip("python_ags4")
    from subsurface_characterization.formats.ags4 import validate_ags4
    r = validate_ags4(str(SAMPLES / "sample_borehole.ags"))
    assert r.n_errors > 0 and not r.is_valid
    # lines ending in LF, not CR LF (AGS Format Rule 2a), made here: a
    # checkout may turn the sample's LF into CR LF (core.autocrlf)
    text = (SAMPLES / "sample_borehole.ags").read_bytes().decode("utf-8")
    r = validate_ags4(content=text.replace("\r\n", "\n"))
    assert "AGS Format Rule 2a" in r.errors and not r.is_valid


def test_a_diggs_file_that_passes_the_dictionary_is_valid():
    pytest.importorskip("pydiggs")
    from subsurface_characterization.formats.diggs_validation import (
        validate_diggs_dictionary)
    r = validate_diggs_dictionary(str(SAMPLES / "sample_site_diggs.xml"))
    assert r.is_valid and r.n_errors == 0 and r.errors == []
