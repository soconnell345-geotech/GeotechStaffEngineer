"""The DIGGS 2.6 schema check runs on every host, without pydiggs.

Until 2026-10-08 the schema check needed the optional pydiggs package,
which is deliberately not installed on Databricks (5.10.1: its dependencies
replaced the notebook's Pygments and the kernel was killed). So on the
cluster nothing could check a DIGGS file, and in the 2026-10-06 field
session a hand-typed "DIGGS" file went unchecked. The schema now ships with
the package (``formats/schemas/diggs-schema-2.6.zip``) and is checked with
lxml. Every test here hides pydiggs.
"""

import os
import sys
import zipfile

import pytest

pytest.importorskip("lxml")

from subsurface_characterization.formats import diggs_validation as V

XS = "{http://www.w3.org/2001/XMLSchema}"


@pytest.fixture
def no_pydiggs(monkeypatch, tmp_path):
    """pydiggs cannot be imported, and the schema unpacks into a fresh
    folder (so the unpack itself is exercised)."""
    monkeypatch.setitem(sys.modules, "pydiggs", None)
    monkeypatch.setenv(V.SCHEMA_CACHE_ENV, str(tmp_path / "cache"))
    monkeypatch.setattr(V, "_unpacked", None)
    assert V.has_pydiggs() is False
    return tmp_path


def _writer_output():
    pytest.importorskip("pydantic")
    from report_ingest.diggs_writer import write_diggs
    from report_ingest.model import Investigation
    inv = Investigation.model_validate({
        "investigation_id": "B-1", "kind": "boring", "depth_unit": "m",
        "total_depth": {"value": 10.0, "unit": "m"},
        "layers": [{"top": {"value": 0.0, "unit": "m"},
                    "bottom": {"value": 10.0, "unit": "m"},
                    "description": "Brown poorly graded SAND", "uscs": "SP"}],
        "spt": [{"depth_top": {"value": 1.0, "unit": "m"},
                 "depth_bottom": {"value": 1.45, "unit": "m"},
                 "blows": [2, 1, 2], "n": 3}],
    })
    return write_diggs([inv])


class TestTheBundle:
    def test_the_zip_and_its_licence_ship_with_the_package(self):
        here = os.path.dirname(V.SCHEMA_ZIP)
        assert os.path.isfile(V.SCHEMA_ZIP)
        licence = open(os.path.join(here, "LICENSE-DIGGS-SCHEMA"),
                       encoding="utf-8").read()
        assert "Mozilla Public License Version 2.0" in licence

    def test_every_import_is_inside_the_zip_and_none_is_remote(self):
        from lxml import etree
        with zipfile.ZipFile(V.SCHEMA_ZIP) as zf:
            names = set(zf.namelist())
            assert "Diggs.xsd" in names
            for name in names:
                root = etree.fromstring(zf.read(name))
                for el in root:
                    if el.tag not in (XS + "import", XS + "include",
                                      XS + "redefine"):
                        continue
                    loc = el.get("schemaLocation")
                    if not loc:
                        continue
                    assert not loc.startswith(("http:", "https:")), (name, loc)
                    target = os.path.normpath(os.path.join(
                        os.path.dirname(name), loc)).replace("\\", "/")
                    assert target in names, (name, loc)

    def test_pyproject_ships_the_schema(self):
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        text = open(os.path.join(root, "pyproject.toml"), encoding="utf-8").read()
        assert '"formats/schemas/*.zip"' in text
        assert '"formats/schemas/LICENSE-DIGGS-SCHEMA"' in text


class TestTheCheckWithoutPydiggs:
    def test_26_is_checkable_and_25a_is_not(self, no_pydiggs):
        assert V.has_schema_check("2.6") is True
        assert V.has_schema_check("2.5.a") is False
        with pytest.raises(ImportError):
            V.validate_diggs_schema(content="<x/>", schema_version="2.5.a")

    def test_the_writers_own_file_passes(self, no_pydiggs):
        res = V.validate_diggs_schema(content=_writer_output())
        assert res.is_valid, res.errors
        assert res.schema_version == "2.6"

    def test_the_writer_gate_now_checks_it(self, no_pydiggs):
        from report_ingest.diggs_writer import diggs_schema_gate
        ok, errors = diggs_schema_gate(_writer_output())
        assert ok is True and errors == []

    def test_a_hand_typed_root_fails_at_the_root(self, no_pydiggs):
        # The field session's shape: an invented root under a wrong namespace.
        xml = ('<?xml version="1.0"?><DIGGS '
               'xmlns="http://diggsml.org/schemas/2.6.a"><project/></DIGGS>')
        res = V.validate_diggs_schema(content=xml)
        assert res.is_valid is False
        assert "No matching global declaration" in " ".join(res.errors)

    def test_xml_that_is_not_well_formed_is_said_so(self, no_pydiggs):
        res = V.validate_diggs_schema(content="<Diggs><unclosed></Diggs>")
        assert res.is_valid is False
        assert res.errors[0].startswith("not well-formed XML")

    def test_a_file_on_disk_is_checked_too(self, no_pydiggs):
        path = no_pydiggs / "site.xml"
        path.write_text(_writer_output(), encoding="utf-8")
        res = V.validate_diggs_schema(filepath=str(path))
        assert res.is_valid, res.errors
        assert res.source == "site.xml"

    def test_the_unpack_is_reused(self, no_pydiggs):
        first = V.bundled_schema_path()
        assert first.startswith(str(no_pydiggs / "cache"))
        assert V.bundled_schema_path() == first

    def test_the_agent_method_answers_without_pydiggs(self, no_pydiggs):
        from funhouse_agent.dispatch import call_agent
        res = call_agent("subsurface", "validate_diggs_schema",
                         {"content": _writer_output()})
        res = res if isinstance(res, dict) else __import__("json").loads(res)
        assert res.get("is_valid") is True, res
        old = call_agent("subsurface", "validate_diggs_schema",
                         {"content": "<x/>", "schema_version": "2.5.a"})
        old = old if isinstance(old, dict) else __import__("json").loads(old)
        assert "pydiggs" in old.get("error", "")
