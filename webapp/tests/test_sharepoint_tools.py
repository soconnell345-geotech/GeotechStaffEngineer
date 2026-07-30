"""Offline tests for the agent-facing SharePoint tools (fake file manager)."""

import os

import pytest

pytest.importorskip("langchain_core")

import webapp.sharepoint_store as sp
import webapp.sharepoint_tools as spt


class FakeFM:
    def __init__(self):
        self.entries = [
            {"name": "boring_logs.pdf", "type": "file", "size": 1024,
             "path": "/sites/X/Shared Documents/General/GSE_app/boring_logs.pdf"},
            {"name": "reports", "type": "folder",
             "path": "/sites/X/Shared Documents/General/GSE_app/reports"},
        ]
        self.uploads = []
        self.folders = []
        self.download_content = b"PDF-bytes"
        self.existing_names = set()

    def ls(self, path):
        self.last_ls = path
        return self.entries

    def download_file(self, path, local_path=None, return_bytes=True,
                      overwrite=False):
        if "missing" in path:
            raise FileNotFoundError(path)
        self.last_download = path
        with open(local_path, "wb") as fh:
            fh.write(self.download_content)
        return True

    def upload_file(self, local, remote, overwrite=False):
        if os.path.basename(remote) in self.existing_names and not overwrite:
            return False
        self.uploads.append((local, remote))
        return True

    def create_folder(self, path):
        self.folders.append(path)
        return True

    def get_web_url(self, path):
        return f"https://sp.example/{path}"

    def search_filenames(self, query, path=None):
        self.last_search = (query, path)
        return [e for e in self.entries if query in e["name"]]


@pytest.fixture
def fake_sp(monkeypatch, tmp_path):
    """Configured store with a fake fm; working folder -> tmp."""
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    monkeypatch.setenv(sp.ENV_ROOT, "Shared Documents/General/GSE_app")
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    fm = FakeFM()
    store = sp.SharePointStore(file_manager=fm)
    monkeypatch.setattr(sp, "_STORE", store)
    return fm


@pytest.fixture
def unconfigured(monkeypatch):
    for e in (sp.ENV_SITE, sp.ENV_TOKEN, sp.ENV_TOKEN_FILE,
              sp.ENV_CLIENT_ID, sp.ENV_CLIENT_SECRET):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setattr(sp, "_STORE", None)


# ---------------------------------------------------------------------------

def test_unconfigured_everything_says_so(unconfigured):
    assert "not configured" in spt.sharepoint_list_files.invoke({"path": ""})
    assert "not configured" in spt.sharepoint_download_file.invoke(
        {"path": "a.pdf"})
    assert spt.tools_if_configured() == ([], "")


def test_tools_if_configured_returns_four_tools(fake_sp):
    tools, prompt = spt.tools_if_configured()
    assert len(tools) == 4 and "SHAREPOINT" in prompt
    names = {t.name for t in tools}
    assert names == {"sharepoint_list_files", "sharepoint_download_file",
                     "sharepoint_upload_file", "sharepoint_search_files"}


def test_list_relative_path_resolves_under_root(fake_sp):
    out = spt.sharepoint_list_files.invoke({"path": "reports"})
    assert fake_sp.last_ls == "Shared Documents/General/GSE_app/reports"
    assert "boring_logs.pdf" in out and "[folder] reports" in out


def test_list_absolute_paths_pass_through(fake_sp):
    spt.sharepoint_list_files.invoke(
        {"path": "/sites/Other/Shared Documents/x"})
    assert fake_sp.last_ls == "/sites/Other/Shared Documents/x"
    spt.sharepoint_list_files.invoke({"path": "Shared Documents/Elsewhere"})
    assert fake_sp.last_ls == "Shared Documents/Elsewhere"
    spt.sharepoint_list_files.invoke({"path": "https://x.sharepoint.com/f"})
    assert fake_sp.last_ls == "https://x.sharepoint.com/f"


def test_download_lands_in_working_folder(fake_sp, tmp_path):
    out = spt.sharepoint_download_file.invoke({"path": "boring_logs.pdf"})
    dest = tmp_path / "work" / "boring_logs.pdf"
    assert dest.exists() and dest.read_bytes() == b"PDF-bytes"
    assert "Downloaded" in out and str(dest) in out
    assert fake_sp.last_download == \
        "Shared Documents/General/GSE_app/boring_logs.pdf"


def test_download_missing_gives_guidance(fake_sp):
    out = spt.sharepoint_download_file.invoke({"path": "missing.pdf"})
    assert "not found" in out and "sharepoint_search_files" in out


def test_upload_creates_folder_and_reports_link(fake_sp, tmp_path):
    local = tmp_path / "calc_package.pdf"
    local.write_bytes(b"x")
    out = spt.sharepoint_upload_file.invoke(
        {"local_path": str(local), "dest_folder": "deliverables"})
    assert fake_sp.uploads[-1][1] == \
        "Shared Documents/General/GSE_app/deliverables/calc_package.pdf"
    assert "Shared Documents/General/GSE_app/deliverables" in fake_sp.folders
    assert "Link: https://sp.example/" in out


def test_upload_name_collision_gets_timestamped_name(fake_sp, tmp_path):
    fake_sp.existing_names.add("calc_package.pdf")
    local = tmp_path / "calc_package.pdf"
    local.write_bytes(b"x")
    out = spt.sharepoint_upload_file.invoke({"local_path": str(local)})
    assert "Uploaded" in out
    assert "calc_package_" in fake_sp.uploads[-1][1]     # timestamp suffix


def test_upload_missing_local_file(fake_sp):
    out = spt.sharepoint_upload_file.invoke({"local_path": "/nope/x.pdf"})
    assert "Local file not found" in out


def test_search_scopes_to_root_and_formats(fake_sp):
    out = spt.sharepoint_search_files.invoke({"query": "boring"})
    assert fake_sp.last_search == ("boring",
                                   "Shared Documents/General/GSE_app")
    assert "boring_logs.pdf" in out
    out2 = spt.sharepoint_search_files.invoke({"query": "zzz"})
    assert "No files matching" in out2


def test_errors_never_raise(fake_sp, monkeypatch):
    def boom(path):
        raise RuntimeError("proxy down")
    monkeypatch.setattr(fake_sp, "ls", boom)
    out = spt.sharepoint_list_files.invoke({"path": ""})
    assert "SharePoint list error" in out and "proxy down" in out
