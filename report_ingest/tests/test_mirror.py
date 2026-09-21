"""The durable mirror offline: both backends, the manifest, and the failures.

A fake file manager stands in for the Funhouse SharePoint object, holding
its remote tree in a dict, exactly as ``webapp/tests/test_sharepoint_store.py``
does for the app's conversation mirror. Nothing here touches a network, a
credential or a cluster.
"""

from __future__ import annotations

import json
import os

import pytest

from report_ingest import mirror as m


class FakeFM:
    """A Funhouse-shaped file manager over a ``{remote path: bytes}`` dict."""

    def __init__(self, fail_names=()):
        self.tree = {}
        self.folders = []
        self.uploads = []
        self.downloads = []
        self.fail_names = set(fail_names)

    def create_folder(self, path):
        self.folders.append(path)
        return True

    def upload_file(self, local, remote, overwrite=False):
        if os.path.basename(remote) in self.fail_names:
            raise RuntimeError("the library said no")
        with open(local, "rb") as fh:
            self.tree[remote] = fh.read()
        self.uploads.append((local, remote, overwrite))
        return True

    def put(self, remote, data: bytes):
        """Seed the remote as if an earlier run had written it."""
        self.tree[remote] = data

    def ls(self, path):
        path = str(path).rstrip("/")
        seen = {}
        for remote in self.tree:
            if not remote.startswith(path + "/"):
                continue
            rest = remote[len(path) + 1:]
            head, _, tail = rest.partition("/")
            if tail:
                seen.setdefault(head, {"name": head, "type": "folder",
                                       "path": f"{path}/{head}"})
            else:
                seen[head] = {"name": head, "type": "file", "path": remote,
                              "size": len(self.tree[remote])}
        return list(seen.values())

    def download_file(self, path, local_path=None, return_bytes=True,
                      overwrite=False):
        if path not in self.tree:
            raise FileNotFoundError(path)
        self.downloads.append(path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(self.tree[path])
        return True


class BrokenFM(FakeFM):
    """Every upload fails, the way an expired token does."""

    def upload_file(self, local, remote, overwrite=False):
        raise RuntimeError("401 the token expired")


def _run_dir(tmp_path, name="521_sheet"):
    """A folder shaped like one of this stage's ``out_dir``s."""
    out = tmp_path / name
    (out / "runs").mkdir(parents=True)
    (out / "vision").mkdir()
    (out / "runs" / "R36.json").write_text('{"id": "R36"}', encoding="utf-8")
    (out / "vision" / "R36.json").write_text('{"id": "R36"}',
                                             encoding="utf-8")
    (out / "RESULTS.md").write_text("# results\n", encoding="utf-8")
    return out


# -- what the caller may pass ------------------------------------------------

class TestWhatItAccepts:

    def test_a_file_manager_is_used_as_it_is(self):
        fm = FakeFM()
        assert m.file_manager_of(fm) is fm

    def test_an_fh_sp_client_shaped_object_is_unwrapped(self):
        fm = FakeFM()
        client = type("Client", (), {"file_manager": fm})()
        assert m.file_manager_of(client) is fm

    def test_a_store_whose_file_manager_is_a_METHOD_is_unwrapped(self):
        """The app's SharePointStore exposes it as a method, not a property."""
        fm = FakeFM()

        class Store:
            def file_manager(self):
                return fm

        assert m.file_manager_of(Store()) is fm

    def test_nothing_passed_is_no_backend_at_all(self, tmp_path):
        mirror = m.Mirror()
        assert not mirror.active
        summary = mirror.mirror_dir(_run_dir(tmp_path), "anywhere")
        assert summary == {"uploaded": 0, "skipped": 0, "errors": [],
                           "remote": "anywhere",
                           "duration_s": summary["duration_s"]}

    def test_the_remote_folder_is_the_out_dir_s_own_name(self):
        mirror = m.Mirror(sharepoint=FakeFM())
        assert mirror.remote_for("521_sheet") == (
            "GeotechStaffEngineer/report_ingest/521_sheet")


# -- the SharePoint backend --------------------------------------------------

class TestTheSharePointBackend:

    def test_every_file_lands_under_the_run_s_remote_folder(self, tmp_path):
        fm = FakeFM()
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        remote = mirror.remote_for(out.name)
        summary = mirror.mirror_dir(out, remote)
        assert summary["uploaded"] == 3 and summary["errors"] == []
        assert sorted(fm.tree) == [
            f"{remote}/RESULTS.md",
            f"{remote}/runs/R36.json",
            f"{remote}/vision/R36.json",
        ]

    def test_the_manifest_skips_a_file_that_has_not_changed(self, tmp_path):
        fm = FakeFM()
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        remote = mirror.remote_for(out.name)
        mirror.mirror_dir(out, remote)
        again = mirror.mirror_dir(out, remote)
        assert again["uploaded"] == 0
        assert again["skipped"] == 3
        assert len(fm.uploads) == 3, "nothing was sent twice"

    def test_a_changed_file_is_sent_again(self, tmp_path):
        fm = FakeFM()
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        remote = mirror.remote_for(out.name)
        mirror.mirror_dir(out, remote)
        (out / "RESULTS.md").write_text("# results, now longer\n",
                                        encoding="utf-8")
        again = mirror.mirror_dir(out, remote)
        assert again["uploaded"] == 1 and again["skipped"] == 2
        assert b"longer" in fm.tree[f"{remote}/RESULTS.md"]

    def test_the_manifest_is_never_itself_uploaded(self, tmp_path):
        fm = FakeFM()
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        mirror.mirror_dir(out, mirror.remote_for(out.name))
        assert (out / m.MANIFEST_NAME).is_file()
        assert not any(m.MANIFEST_NAME in remote for remote in fm.tree)

    def test_a_failing_manager_reports_and_does_not_raise(self, tmp_path):
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=BrokenFM())
        summary = mirror.mirror_dir(out, mirror.remote_for(out.name))
        assert summary["uploaded"] == 0
        assert len(summary["errors"]) == 3
        assert "401" in summary["errors"][0]

    def test_one_bad_file_does_not_stop_the_others(self, tmp_path):
        fm = FakeFM(fail_names={"RESULTS.md"})
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        summary = mirror.mirror_dir(out, mirror.remote_for(out.name))
        assert summary["uploaded"] == 2 and len(summary["errors"]) == 1
        assert any(r.endswith("runs/R36.json") for r in fm.tree)

    def test_a_file_that_failed_is_retried_next_time(self, tmp_path):
        fm = FakeFM(fail_names={"RESULTS.md"})
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm)
        remote = mirror.remote_for(out.name)
        mirror.mirror_dir(out, remote)
        fm.fail_names.clear()                 # the token came back
        again = mirror.mirror_dir(out, remote)
        assert again["uploaded"] == 1
        assert f"{remote}/RESULTS.md" in fm.tree


# -- restoring ---------------------------------------------------------------

class TestRestoring:

    def test_what_the_remote_holds_and_the_local_does_not_comes_back(
            self, tmp_path):
        fm = FakeFM()
        remote = "GeotechStaffEngineer/report_ingest/521_sheet"
        fm.put(f"{remote}/runs/R36.json", b'{"id": "R36"}')
        fm.put(f"{remote}/vision/R31.json", b'{"id": "R31"}')
        out = tmp_path / "521_sheet"
        mirror = m.Mirror(sharepoint=fm)
        summary = mirror.restore_dir(remote, out)
        assert summary["downloaded"] == 2 and summary["errors"] == []
        assert json.loads((out / "runs" / "R36.json").read_text())["id"] \
            == "R36"
        assert (out / "vision" / "R31.json").is_file()

    def test_a_local_file_is_left_alone_unless_overwrite_is_asked_for(
            self, tmp_path):
        fm = FakeFM()
        remote = "GeotechStaffEngineer/report_ingest/521_sheet"
        fm.put(f"{remote}/runs/R36.json", b'{"id": "remote"}')
        out = tmp_path / "521_sheet"
        (out / "runs").mkdir(parents=True)
        (out / "runs" / "R36.json").write_text('{"id": "local"}',
                                               encoding="utf-8")
        mirror = m.Mirror(sharepoint=fm)
        summary = mirror.restore_dir(remote, out)
        assert summary["downloaded"] == 0 and summary["skipped"] == 1
        assert '"local"' in (out / "runs" / "R36.json").read_text()

        summary = mirror.restore_dir(remote, out, overwrite=True)
        assert summary["downloaded"] == 1
        assert '"remote"' in (out / "runs" / "R36.json").read_text()

    def test_a_restore_seeds_the_manifest_so_nothing_goes_straight_back(
            self, tmp_path):
        fm = FakeFM()
        remote = "GeotechStaffEngineer/report_ingest/521_sheet"
        fm.put(f"{remote}/runs/R36.json", b'{"id": "R36"}')
        out = tmp_path / "521_sheet"
        mirror = m.Mirror(sharepoint=fm)
        mirror.restore_dir(remote, out)
        summary = mirror.mirror_dir(out, remote)
        assert summary["uploaded"] == 0 and summary["skipped"] == 1
        assert fm.uploads == []

    def test_an_empty_remote_restores_nothing_and_says_nothing_wrong(
            self, tmp_path):
        mirror = m.Mirror(sharepoint=FakeFM())
        summary = mirror.restore_dir("nowhere/at/all", tmp_path / "out")
        assert summary == {"downloaded": 0, "skipped": 0, "errors": [],
                           "remote": "nowhere/at/all",
                           "duration_s": summary["duration_s"]}


# -- the filesystem backend --------------------------------------------------

class TestTheFilesystemBackend:

    def test_it_round_trips_a_whole_run(self, tmp_path):
        durable = tmp_path / "durable"
        out = _run_dir(tmp_path)
        mirror = m.Mirror(durable_dir=durable)
        remote = mirror.remote_for(out.name)
        summary = mirror.mirror_dir(out, remote)
        assert summary["uploaded"] == 3 and summary["errors"] == []
        # The base folder is stripped: a folder does not need the library's
        # tree, only the run's own name.
        assert (durable / "521_sheet" / "runs" / "R36.json").is_file()

        wiped = tmp_path / "after_the_restart"
        back = m.Mirror(durable_dir=durable).restore_dir(remote, wiped)
        assert back["downloaded"] == 3
        assert (wiped / "vision" / "R36.json").read_text() == '{"id": "R36"}'

    def test_it_skips_what_it_has_already_copied(self, tmp_path):
        durable = tmp_path / "durable"
        out = _run_dir(tmp_path)
        mirror = m.Mirror(durable_dir=durable)
        remote = mirror.remote_for(out.name)
        mirror.mirror_dir(out, remote)
        again = mirror.mirror_dir(out, remote)
        assert again["uploaded"] == 0 and again["skipped"] == 3

    def test_a_remote_path_outside_the_base_folder_is_used_whole(self,
                                                                 tmp_path):
        durable = tmp_path / "durable"
        out = _run_dir(tmp_path)
        mirror = m.Mirror(durable_dir=durable, base_folder="")
        mirror.mirror_dir(out, "some/other/place")
        assert (durable / "some" / "other" / "place" / "RESULTS.md").is_file()


# -- both at once ------------------------------------------------------------

class TestBothBackends:

    def test_a_run_can_be_mirrored_to_sharepoint_and_a_folder_at_once(
            self, tmp_path):
        fm = FakeFM()
        durable = tmp_path / "durable"
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm, durable_dir=durable)
        remote = mirror.remote_for(out.name)
        summary = mirror.mirror_dir(out, remote)
        assert summary["uploaded"] == 6, "three files, two backends"
        assert f"{remote}/RESULTS.md" in fm.tree
        assert (durable / "521_sheet" / "RESULTS.md").is_file()

    def test_each_backend_keeps_its_own_manifest(self, tmp_path):
        fm = BrokenFM()
        durable = tmp_path / "durable"
        out = _run_dir(tmp_path)
        mirror = m.Mirror(sharepoint=fm, durable_dir=durable)
        remote = mirror.remote_for(out.name)
        summary = mirror.mirror_dir(out, remote)
        # The folder took all three; SharePoint took none and must not be
        # credited with the folder's uploads on the next pass.
        assert summary["uploaded"] == 3 and len(summary["errors"]) == 3
        again = mirror.mirror_dir(out, remote)
        assert again["skipped"] == 3, "the folder's three, and no more"
        assert len(again["errors"]) == 3, "SharePoint is tried again"

    def test_the_description_names_both(self, tmp_path):
        mirror = m.Mirror(sharepoint=FakeFM(), durable_dir=tmp_path / "d")
        text = mirror.describe(mirror.remote_for("521_sheet"))
        assert "SharePoint" in text and "521_sheet" in text

    def test_with_no_backend_the_description_says_so(self):
        assert "nowhere" in m.Mirror().describe("anything")


# -- the manifest ------------------------------------------------------------

def test_a_manifest_for_a_different_remote_is_discarded(tmp_path):
    fm = FakeFM()
    out = _run_dir(tmp_path)
    mirror = m.Mirror(sharepoint=fm)
    mirror.mirror_dir(out, "somewhere/521_sheet")
    # The owner renamed the folder. Everything has to go to the new one.
    summary = mirror.mirror_dir(out, "elsewhere/521_sheet")
    assert summary["uploaded"] == 3 and summary["skipped"] == 0
    manifest = json.loads((out / m.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["remote"] == "elsewhere/521_sheet"


def test_an_unreadable_manifest_is_treated_as_no_manifest(tmp_path):
    fm = FakeFM()
    out = _run_dir(tmp_path)
    (out / m.MANIFEST_NAME).write_text("this is not json", encoding="utf-8")
    mirror = m.Mirror(sharepoint=fm)
    summary = mirror.mirror_dir(out, "somewhere/521_sheet")
    assert summary["uploaded"] == 3 and summary["errors"] == []


@pytest.mark.parametrize("missing", ["gone", "also_gone"])
def test_mirroring_a_folder_that_is_not_there_does_nothing(tmp_path, missing):
    mirror = m.Mirror(sharepoint=FakeFM())
    summary = mirror.mirror_dir(tmp_path / missing, "somewhere/x")
    assert summary["uploaded"] == 0 and summary["errors"] == []
