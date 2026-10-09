"""Live smoke wave 3 (module_work/live_smoke/runs/w3-confirm-sonnet/
REVIEW.md), the web-app side.

* F8  ``downloads.json`` -- mirrored to the conversation's SharePoint folder
      -- stores each download by its name in the conversation folder, never
      the server's absolute path; the ledger still reads back absolute, and
      a conversation restored to another place still finds its downloads.
"""

import json
import os

import pytest

import webapp.core as core


@pytest.fixture(autouse=True)
def _tmp_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))


def _ledger(conv) -> list:
    with open(os.path.join(str(conv), core.DOWNLOADS_LEDGER),
              encoding="utf-8") as fh:
        return json.load(fh)


def test_the_ledger_stores_conversation_relative_names(tmp_path):
    conv = tmp_path / "server" / "conversations" / "abc123"
    files = conv / "files"
    files.mkdir(parents=True)
    (files / "Report Vol I.pdf").write_bytes(b"%PDF")
    core.record_download(str(conv), "Shared Documents/X/Report Vol I.pdf",
                         str(files / "Report Vol I.pdf"), 4)
    (stored,) = _ledger(conv)
    assert stored["local"] == "files/Report Vol I.pdf"
    raw = (conv / core.DOWNLOADS_LEDGER).read_text(encoding="utf-8")
    assert str(tmp_path) not in raw and "server" not in raw
    # ... and reads back as the file it is, for every reader of the ledger
    (entry,) = core.load_downloads(str(conv))
    assert entry["local"] == os.path.abspath(str(files / "Report Vol I.pdf"))
    assert entry["remote"] == "Shared Documents/X/Report Vol I.pdf"


def test_a_restored_conversation_finds_its_downloads(tmp_path):
    old = tmp_path / "host_a" / "abc123"
    (old / "files").mkdir(parents=True)
    core.record_download(str(old), "Shared/r.pdf",
                         str(old / "files" / "r.pdf"), 1)
    new = tmp_path / "host_b" / "elsewhere" / "abc123"
    new.parent.mkdir(parents=True)
    os.replace(str(old), str(new))
    (entry,) = core.load_downloads(str(new))
    assert entry["local"] == os.path.abspath(str(new / "files" / "r.pdf"))


def test_an_older_ledgers_absolute_paths_are_rewritten_on_the_next_download(
        tmp_path):
    conv = tmp_path / "conv"
    (conv / "files").mkdir(parents=True)
    old_abs = os.path.abspath(str(conv / "files" / "a.pdf"))
    (conv / core.DOWNLOADS_LEDGER).write_text(json.dumps(
        [{"remote": "Shared/a.pdf", "local": old_abs, "bytes": 1,
          "ts": 0.0}]), encoding="utf-8")
    # an older ledger still reads
    (entry,) = core.load_downloads(str(conv))
    assert entry["local"] == old_abs
    core.record_download(str(conv), "Shared/b.pdf",
                         str(conv / "files" / "b.pdf"), 2)
    assert sorted(e["local"] for e in _ledger(conv)) == [
        "files/a.pdf", "files/b.pdf"]
    assert sorted(os.path.basename(e["local"])
                  for e in core.load_downloads(str(conv))) == ["a.pdf",
                                                              "b.pdf"]
