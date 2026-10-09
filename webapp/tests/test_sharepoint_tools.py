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
    # Named by its place in the working folder, never by the server path
    # (live smoke 1, A6).
    assert "Downloaded" in out and "-> 'boring_logs.pdf'" in out
    assert str(tmp_path) not in out
    assert fake_sp.last_download == \
        "Shared Documents/General/GSE_app/boring_logs.pdf"


def test_download_missing_gives_guidance(fake_sp):
    out = spt.sharepoint_download_file.invoke({"path": "missing.pdf"})
    assert "not found" in out and "sharepoint_search_files" in out


def _work_file(tmp_path, name, data=b"x"):
    """A file in the conversation's working folder (the fixture binds it):
    only those may be uploaded once a working folder is bound (A2)."""
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    (work / name).write_bytes(data)
    return work / name


def test_upload_creates_folder_and_reports_link(fake_sp, tmp_path):
    local = _work_file(tmp_path, "calc_package.pdf")
    out = spt.sharepoint_upload_file.invoke(
        {"local_path": str(local), "dest_folder": "deliverables"})
    assert fake_sp.uploads[-1][1] == \
        "Shared Documents/General/GSE_app/deliverables/calc_package.pdf"
    assert "Shared Documents/General/GSE_app/deliverables" in fake_sp.folders
    assert "Link: https://sp.example/" in out


def test_upload_name_collision_gets_timestamped_name(fake_sp, tmp_path):
    fake_sp.existing_names.add("calc_package.pdf")
    local = _work_file(tmp_path, "calc_package.pdf")
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


class TestBaseFolderNotDoubled:
    """Naming the folder you can see must not double it (live 2026-09-09/10).

    The root ends in the base folder, so "GSE_app/uploaded references/x.pdf"
    used to resolve under ".../GSE_app/GSE_app/..." and 404. The agent burned
    4-5 SharePoint round trips per turn rediscovering that, twice.
    """

    def test_leading_base_segment_is_dropped(self, fake_sp):
        assert spt._resolve("GSE_app/uploaded references/x.pdf") == \
            "Shared Documents/General/GSE_app/uploaded references/x.pdf"

    def test_case_insensitive(self, fake_sp):
        assert spt._resolve("gse_app/uploaded references") == \
            "Shared Documents/General/GSE_app/uploaded references"

    def test_plain_relative_path_is_untouched(self, fake_sp):
        assert spt._resolve("uploaded references/x.pdf") == \
            "Shared Documents/General/GSE_app/uploaded references/x.pdf"

    def test_only_the_first_segment_is_considered(self, fake_sp):
        """A nested folder that happens to share the name still resolves."""
        assert spt._resolve("uploaded references/GSE_app/x.pdf") == \
            "Shared Documents/General/GSE_app/uploaded references/GSE_app/x.pdf"

    def test_bare_base_name_alone_is_not_stripped(self, fake_sp):
        """"GSE_app" on its own is a real (if odd) child request — keep it.

        Stripping it would silently turn a request for a child folder into a
        request for the root, which is a different answer, not a fixed one.
        """
        assert spt._resolve("GSE_app") == \
            "Shared Documents/General/GSE_app/GSE_app"

    def test_absolute_paths_still_pass_through(self, fake_sp):
        for p in ("Shared Documents/General/GSE_app/x.pdf",
                  "/sites/CSEGeotechGroup/Shared Documents/x.pdf",
                  "https://t.sharepoint.com/sites/x/y.pdf"):
            assert spt._resolve(p) == p

    def test_empty_still_returns_the_root(self, fake_sp):
        assert spt._resolve("") == "Shared Documents/General/GSE_app"
        assert spt._resolve("/") == "Shared Documents/General/GSE_app"


# ---------------------------------------------------------------------------
# Field feedback 2026-09-15 (Nairobi SOE re-run): N5 and N8
# ---------------------------------------------------------------------------

class TestConversationFolder:
    """Asked to save a report to SharePoint, the agent used "uploaded
    references", then "General/GSE_app/conversations/<thread id>", which
    doubled two base segments; the real folder is named by title and date."""

    FOLDER = "Shared Documents/General/GSE_app/conversations/Review_2026-09-15"

    def test_two_repeated_root_segments_are_dropped(self, fake_sp):
        assert spt._resolve("General/GSE_app/conversations/b9ef") == \
            "Shared Documents/General/GSE_app/conversations/b9ef"

    def test_upload_without_dest_goes_to_this_conversation(
            self, fake_sp, tmp_path, monkeypatch):
        monkeypatch.setattr(sp.SharePointStore, "session_folder",
                            lambda self, tid, root=None: self_folder(tid))
        tools, prompt = spt.tools_if_configured(thread_id="t1")
        up = next(t for t in tools if t.name == "sharepoint_upload_file")
        local = _work_file(tmp_path, "SOE_sensitivity_review_embedded.pdf")
        out = up.invoke({"local_path": str(local)})
        assert fake_sp.uploads[-1][1] == \
            f"{self.FOLDER}/files/SOE_sensitivity_review_embedded.pdf"
        assert "this conversation's SharePoint folder" in out
        assert "uploaded references" in prompt
        assert "WITHOUT dest_folder" in prompt

    def test_explicit_dest_is_still_honoured(self, fake_sp, tmp_path,
                                             monkeypatch):
        monkeypatch.setattr(sp.SharePointStore, "session_folder",
                            lambda self, tid, root=None: self_folder(tid))
        up = next(t for t in spt.tools_if_configured(thread_id="t1")[0]
                  if t.name == "sharepoint_upload_file")
        local = _work_file(tmp_path, "a.pdf")
        up.invoke({"local_path": str(local), "dest_folder": "deliverables"})
        assert fake_sp.uploads[-1][1] == \
            "Shared Documents/General/GSE_app/deliverables/a.pdf"


def self_folder(tid):
    return TestConversationFolder.FOLDER


class TestDownloadReuse:
    """The same 23 MB submittal was downloaded four times under two names."""

    def test_second_download_reuses_the_copy(self, fake_sp, monkeypatch):
        calls = []
        original = fake_sp.download_file

        def counting(path, **kw):
            calls.append(path)
            return original(path, **kw)

        monkeypatch.setattr(fake_sp, "download_file", counting)
        first = spt.sharepoint_download_file.invoke(
            {"path": "uploaded references/sub.pdf", "save_as": "sub.pdf"})
        again = spt.sharepoint_download_file.invoke(
            {"path": "uploaded references/sub.pdf", "save_as": "renamed.pdf"})
        assert len(calls) == 1
        assert "reusing" in again and "input to read" in again
        assert "input to read" in first
        spt.sharepoint_download_file.invoke(
            {"path": "uploaded references/sub.pdf", "refresh": True})
        assert len(calls) == 2


def test_build_agent_binds_the_upload_tool_to_the_conversation(
        fake_sp, tmp_path, monkeypatch):
    import webapp.core as core
    import funhouse_agent.deep.agent as deep_agent
    seen = {}
    real = spt.tools_if_configured

    def spy(thread_id=None, record_dir=None):
        seen["thread_id"] = thread_id
        seen["record_dir"] = record_dir
        return real(thread_id=thread_id, record_dir=record_dir)

    monkeypatch.setattr(spt, "tools_if_configured", spy)
    monkeypatch.setattr(deep_agent, "build_deep_agent",
                        lambda model, **kw: object())
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    files = os.path.join(core.conversation_dir("conv42"), "files")
    os.makedirs(files)
    core.build_agent(object(), {}, files, [])
    assert seen["thread_id"] == "conv42"
    # the download tool keeps its ledger in the conversation's own folder
    assert seen["record_dir"] == os.path.dirname(os.path.abspath(files))


# -- addresses copied from a browser (field session 2026-10-01) --------------

SITE = "https://contoso.sharepoint.com/sites/TeamSite"


class TestBrowserUrls:
    def _site(self, monkeypatch):
        from webapp import sharepoint_store
        monkeypatch.setenv(sharepoint_store.ENV_SITE, SITE)

    def test_the_address_bar_of_an_open_folder(self, monkeypatch):
        self._site(monkeypatch)
        url = ("https://contoso.sharepoint.com/sites/TeamSite/Shared%20Documents"
               "/Forms/AllItems.aspx?id=%2Fsites%2FTeamSite%2FShared%20Documents"
               "%2FGeneral%2FGSE%5Fapp%2Fconversations%2Fdocument%5Freview%2F"
               "I%5Fuploaded%5Fa%5Fdoc%E2%80%A6%5F2026%2D10%2D01"
               "&sortField=Modified&isAscending=true&viewid=db0f8117")
        assert spt.browser_url_to_path(url) == (
            "Shared Documents/General/GSE_app/conversations/document_review/"
            "I_uploaded_a_doc\u2026_2026-10-01")

    def test_a_copied_r_link_carries_its_path(self, monkeypatch):
        self._site(monkeypatch)
        url = ("https://contoso.sharepoint.com/:f:/r/sites/TeamSite/"
               "Shared%20Documents/General/GSE_app/conversations?d=w446e25"
               "&csf=1&web=1&e=kV9eOC")
        assert spt.browser_url_to_path(url) == \
            "Shared Documents/General/GSE_app/conversations"
        # and _resolve hands that path on rather than the URL
        assert spt._resolve(url) == \
            "Shared Documents/General/GSE_app/conversations"

    def test_another_site_keeps_its_site(self, monkeypatch):
        self._site(monkeypatch)
        url = ("https://contoso.sharepoint.com/sites/Other/Shared%20Documents/"
               "Forms/AllItems.aspx?id=%2Fsites%2FOther%2FShared%20Documents"
               "%2FReports")
        assert spt.browser_url_to_path(url) == \
            "/sites/Other/Shared Documents/Reports"

    def test_a_plain_file_address_is_converted_too(self, monkeypatch):
        """Since 2026-10-07 a plain address becomes its path like the other
        forms: the Tiny Apps client takes no URLs, and a path is what the
        name fallback needs when the file is not where the address says."""
        self._site(monkeypatch)
        url = ("https://contoso.sharepoint.com/sites/TeamSite/Shared%20Documents"
               "/General/borings.pdf?web=1")
        assert spt.browser_url_to_path(url) == \
            "Shared Documents/General/borings.pdf"
        assert spt._resolve(url) == "Shared Documents/General/borings.pdf"

    def test_a_token_link_has_no_path_and_says_what_to_paste(self, monkeypatch):
        self._site(monkeypatch)
        url = "https://contoso.sharepoint.com/:f:/s/TeamSite/EaBcD123xyz?e=abc"
        assert spt.browser_url_to_path(url) is None
        assert spt._resolve(url) == url          # left for the SDK to try
        assert "AllItems.aspx?id=" in spt.TOKEN_LINK_HINT

    def test_not_a_url_is_left_alone(self):
        assert spt.browser_url_to_path("Shared Documents/x") is None
        assert spt.browser_url_to_path("") is None
