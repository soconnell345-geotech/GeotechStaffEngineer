"""SharePoint links, honest failures, finding a file by name, one local copy.

Field session 2026-10-06 (geotech page, 5.32.0). In order:

* a pasted ``/:b:/r/sites/...pdf?d=...`` link downloaded the report in turns
  4 and 5 -- that link form was handled;
* in turn 6 the SAME link, a folder listing, the parent folder and a search
  all came back within eight seconds as "SharePoint file not found", "(empty
  or missing folder)" and "No files matching" -- the shape of a refused
  request the SDK turns into empty answers (``ls`` -> ``[]`` and
  ``FileNotFoundError("... Download failed: 401 ...")``). Sixteen minutes
  later the same paths worked. The agent concluded the file did not exist,
  and then that its earlier work had never been done;
* the exact file name, searched in the folder it was listed in, found
  nothing three times (the search index misses such names);
* the report was saved twice in the conversation, under two names.

The fake below behaves like the SDK on each of those points. Every file and
folder name in it is synthetic.
"""

import os

import pytest

pytest.importorskip("langchain_core")

import webapp.core as core
import webapp.sharepoint_store as sp
import webapp.sharepoint_tools as spt

SITE = "https://contoso.sharepoint.com/sites/TeamSite"
ROOT = "Shared Documents/General/APP"
REPORT = "Site _Volume I Ground Report_01022026.pdf"


class TreeFM:
    """A SharePoint file tree, answering the way the Funhouse SDK does.

    ``refused`` makes every call behave as the SDK does on a 401: ``ls`` and
    ``search_filenames`` return ``[]``, ``get_folder_details`` ``{}``, and
    ``download_file`` raises ``FileNotFoundError`` with the status in its
    text. The search index matches single words only (it never finds a long
    name with underscores and spaces as one query)."""

    def __init__(self, files):
        self.files = {k: v for k, v in files.items()}
        self.refused = False
        self.downloads = []
        self.searches = []
        self.listed = []

    # paths come in as "Shared Documents/..." here
    def _norm(self, path):
        p = path.strip().strip("/")
        if p.lower().startswith("sites/teamsite/"):
            p = p[len("sites/TeamSite/"):]
        return p

    def _children(self, folder):
        folder = self._norm(folder).rstrip("/")
        out, seen = [], set()
        for path, data in self.files.items():
            if not path.startswith(folder + "/"):
                continue
            rest = path[len(folder) + 1:]
            head = rest.split("/", 1)[0]
            if "/" in rest:
                if head not in seen:
                    seen.add(head)
                    out.append({"name": head, "type": "folder",
                                "path": f"/sites/TeamSite/{folder}/{head}"})
            else:
                out.append({"name": head, "type": "file", "size": len(data),
                            "path": f"/sites/TeamSite/{path}"})
        return out

    def ls(self, path):
        self.listed.append(path)
        if self.refused:
            return []
        return self._children(path)

    def get_folder_details(self, path):
        if self.refused:
            return {}
        folder = self._norm(path).rstrip("/")
        if any(p.startswith(folder + "/") for p in self.files) or \
                folder == ROOT:
            return {"id": "x", "folder": {"childCount": 1}}
        return {}

    def download_file(self, path, local_path=None, return_bytes=True,
                      overwrite=False):
        self.downloads.append(path)
        if self.refused:
            raise FileNotFoundError(
                f"Failed to download file from SharePoint: {path}. Download "
                "failed: 401 - {\"error\": {\"code\": "
                "\"InvalidAuthenticationToken\"}}")
        key = self._norm(path)
        if path.lower().startswith("http"):
            key = self.shared_links.get(path, "")
        if key not in self.files:
            raise FileNotFoundError(
                f"Failed to download file from SharePoint: {path}. Download "
                "failed: 404 - {\"error\": {\"code\": \"itemNotFound\"}}")
        with open(local_path, "wb") as fh:
            fh.write(self.files[key])
        return True

    shared_links: dict = {}

    def search_filenames(self, query, path=None):
        self.searches.append(query)
        if self.refused or " " in query.strip():
            return []
        q = query.lower()
        return [{"name": p.rsplit("/", 1)[-1], "type": "file",
                 "path": f"/sites/TeamSite/{p}"}
                for p in self.files if q in p.rsplit("/", 1)[-1].lower()]


@pytest.fixture
def tree(monkeypatch, tmp_path):
    monkeypatch.setenv(sp.ENV_SITE, SITE)
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    monkeypatch.setenv(sp.ENV_ROOT, ROOT)
    work = tmp_path / "conv" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    fm = TreeFM({
        f"{ROOT}/uploaded references/{REPORT}": b"%PDF report bytes",
        f"{ROOT}/uploaded references/Other notes.pdf": b"%PDF other",
        f"{ROOT}/projects/archive/Borings 2011.pdf": b"%PDF borings",
        f"{ROOT}/conversations/Old_chat_2026-09-01/files/x.pdf": b"%PDF x",
    })
    monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(file_manager=fm))
    spt._DOWNLOADS.clear()
    fm.work = str(work)
    fm.conv = str(tmp_path / "conv")
    return fm


def _r_link(path):
    """A "copy link" address for a file at ``path`` under the site."""
    from urllib.parse import quote
    return (f"https://contoso.sharepoint.com/:b:/r/sites/TeamSite/"
            f"{quote(path)}?d=w0123abcd&csf=1&web=1&e=AbCdEf")


def _download(fm, **kw):
    tool = spt.make_download_tool(fm.conv)
    return tool.invoke(kw)


# ---------------------------------------------------------------------------
# Link forms
# ---------------------------------------------------------------------------

class TestLinkForms:
    @pytest.fixture(autouse=True)
    def _site(self, monkeypatch):
        monkeypatch.setenv(sp.ENV_SITE, SITE)
        monkeypatch.setenv(sp.ENV_ROOT, ROOT)

    def test_a_pdf_copy_link_with_its_query(self):
        url = _r_link(f"{ROOT}/uploaded references/{REPORT}")
        assert spt.browser_url_to_path(url) == \
            f"{ROOT}/uploaded references/{REPORT}"
        assert spt.link_target_name(url) == REPORT

    @pytest.mark.parametrize("kind", ["b", "w", "x", "p", "f", "u"])
    def test_every_r_link_kind(self, kind):
        url = (f"https://contoso.sharepoint.com/:{kind}:/r/sites/TeamSite/"
               "Shared%20Documents/General/APP/a%20b.docx?d=w1&e=2")
        assert spt.browser_url_to_path(url) == \
            "Shared Documents/General/APP/a b.docx"

    def test_token_links_carry_no_path(self):
        for url in ("https://contoso.sharepoint.com/:b:/s/TeamSite/EaBc12?e=x",
                    "https://contoso.sharepoint.com/:f:/g/personal/EaBc12",
                    "https://contoso.sharepoint.com/:x:/s/TeamSite/AbC?e=1"):
            assert spt.browser_url_to_path(url) is None
            assert spt.link_target_name(url) == ""
            assert spt._resolve(url) == url        # left for the SDK

    def test_office_viewer_link_is_left_for_the_sdk_but_names_the_file(self):
        url = ("https://contoso.sharepoint.com/:x:/r/sites/TeamSite/_layouts/"
               "15/Doc.aspx?sourcedoc=%7BAB12-CD34%7D&file=Lab%20summary.xlsx"
               "&action=default&mobileredirect=true")
        assert spt.browser_url_to_path(url) is None
        assert spt.link_target_name(url) == "Lab summary.xlsx"
        plain = ("https://contoso.sharepoint.com/sites/TeamSite/_layouts/15/"
                 "Doc.aspx?sourcedoc={AB12}&file=Memo.docx")
        assert spt.browser_url_to_path(plain) is None
        assert spt.link_target_name(plain) == "Memo.docx"

    def test_address_bar_of_a_file_preview_in_the_library(self):
        url = ("https://contoso.sharepoint.com/sites/TeamSite/Shared%20Documents"
               "/Forms/AllItems.aspx?id=%2Fsites%2FTeamSite%2FShared%20Documents"
               "%2FGeneral%2FAPP%2Freport%2Epdf&parent=%2Fsites%2FTeamSite")
        assert spt.browser_url_to_path(url) == \
            "Shared Documents/General/APP/report.pdf"

    def test_library_view_without_an_id_is_the_library(self):
        url = ("https://contoso.sharepoint.com/sites/TeamSite/Shared%20Documents"
               "/Forms/AllItems.aspx")
        assert spt.browser_url_to_path(url) == "Shared Documents"

    def test_teams_and_other_sites_keep_their_prefix(self):
        url = ("https://contoso.sharepoint.com/teams/Design/Shared%20Documents/"
               "x.pdf?web=1")
        assert spt.browser_url_to_path(url) == \
            "/teams/Design/Shared Documents/x.pdf"
        assert spt._resolve(url) == "/teams/Design/Shared Documents/x.pdf"

    def test_another_library_of_this_site_stays_absolute(self):
        """A library other than Shared Documents must not be read as a
        folder under the base folder."""
        url = ("https://contoso.sharepoint.com/sites/TeamSite/Project%20Docs/"
               "x.pdf?web=1")
        assert spt.browser_url_to_path(url) == \
            "/sites/TeamSite/Project Docs/x.pdf"
        assert spt._resolve(url) == "/sites/TeamSite/Project Docs/x.pdf"
        # a file at the site root names no library: left for the SDK
        root_file = "https://contoso.sharepoint.com/sites/TeamSite/y.pdf"
        assert spt.browser_url_to_path(root_file) is None

    def test_a_site_page_is_not_a_path(self):
        url = "https://contoso.sharepoint.com/sites/TeamSite/SitePages/Home.aspx"
        assert spt.browser_url_to_path(url) is None

    def test_the_breadcrumb_spelling_of_the_library(self, monkeypatch):
        """The page shows "Documents > General > APP"; the path says
        "Shared Documents/General/APP"."""
        monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(
            file_manager=object()))
        assert spt._resolve("Documents/General/APP/uploaded references") == \
            f"{ROOT}/uploaded references"
        # a folder that really is called Documents under the base is kept
        assert spt._resolve("Documents/x.pdf") == f"{ROOT}/Documents/x.pdf"


# ---------------------------------------------------------------------------
# A failure is never reported as an answer
# ---------------------------------------------------------------------------

class TestRefusedIsNotMissing:
    def test_download_refused_says_so_and_searches_nothing(self, tree):
        tree.refused = True
        out = _download(tree, path=_r_link(f"{ROOT}/uploaded references/"
                                           f"{REPORT}"))
        assert "REFUSED" in out and "401" in out
        assert "not a missing file" in out
        assert "file not found" not in out.lower()
        assert tree.searches == []            # no hunt on a refusal
        assert os.listdir(tree.work) == []    # no part file left behind

    def test_listing_that_comes_back_empty_everywhere(self, tree):
        tree.refused = True
        out = spt.sharepoint_list_files.invoke({"path": "uploaded references"})
        assert "not answering" in out and "ACCESS problem" in out
        assert "empty or missing" not in out

    def test_search_that_finds_nothing_while_refused(self, tree):
        tree.refused = True
        out = spt.sharepoint_search_files.invoke({"query": "Ground Report"})
        assert "not answering" in out
        assert "No files matching" not in out

    def test_a_real_empty_folder_is_still_empty(self, tree):
        tree.files[f"{ROOT}/empty/.keep"] = b""
        tree.ls = lambda path: [] if "empty" in path else \
            TreeFM._children(tree, path)
        out = spt.sharepoint_list_files.invoke({"path": "empty"})
        assert out == f"(empty folder: {ROOT}/empty)"

    def test_a_missing_folder_with_sharepoint_answering(self, tree):
        out = spt.sharepoint_list_files.invoke({"path": "no such folder"})
        assert out.startswith("(empty or missing folder:")

    def test_status_and_kind_are_read_from_the_error(self):
        e401 = FileNotFoundError("x. Download failed: 401 - {}")
        assert spt._status(e401) == 401 and spt._classify(e401) == "refused"
        e404 = FileNotFoundError("x. Download failed: 404 - itemNotFound")
        assert spt._classify(e404) == "missing"
        from webapp.graph_sharepoint import GraphError
        g = GraphError("GET https://graph/x -> HTTP 403: denied", status=403)
        assert spt._classify(g) == "refused"
        assert spt._classify(FileNotFoundError("a/b.pdf")) == "missing"
        assert spt._classify(RuntimeError("HTTP 429: slow down")) == \
            "throttled"
        assert spt._classify(RuntimeError("proxy down")) == "error"


# ---------------------------------------------------------------------------
# A link that does not open: find the file by its name
# ---------------------------------------------------------------------------

class TestFindByName:
    def test_the_field_session_link_downloads_directly(self, tree):
        out = _download(tree, path=_r_link(f"{ROOT}/uploaded references/"
                                           f"{REPORT}"),
                        save_as="report_full.pdf")
        assert out.startswith("Downloaded ")
        assert tree.downloads == [f"{ROOT}/uploaded references/{REPORT}"]
        assert open(os.path.join(tree.work, "report_full.pdf"), "rb").read() \
            == b"%PDF report bytes"

    def test_moved_file_is_found_by_name_and_said_plainly(self, tree):
        # the address names a folder the file is no longer in
        out = _download(tree, path=_r_link(f"{ROOT}/projects/Borings 2011.pdf"))
        assert "did not open directly" in out
        assert "Borings 2011.pdf" in out and "Tell the user" in out
        assert f"/sites/TeamSite/{ROOT}/projects/archive/Borings 2011.pdf" in \
            out
        assert os.path.isfile(os.path.join(tree.work, "Borings 2011.pdf"))
        # the "Downloaded X -> path (N bytes)" line the app reads inputs from
        from webapp.output_capture import paths_in
        assert paths_in(out)[1] == [os.path.join(tree.work, "Borings 2011.pdf")]

    def test_a_name_the_index_misses_is_found_by_a_shorter_search(self, tree):
        """The field session: the exact name, listed in the folder, found
        nothing in the index three times."""
        out = spt.sharepoint_search_files.invoke({"query": REPORT})
        assert REPORT in out and "Matches for" in out
        assert "shorter search" in out
        assert "No files matching" not in out

    def test_with_no_index_at_all_the_folders_are_walked(self, tree):
        tree.search_filenames = lambda query, path=None: []
        out = spt.sharepoint_search_files.invoke({"query": "ground report"})
        assert REPORT in out and "looking through" in out

    def test_search_still_says_no_when_answering_and_absent(self, tree):
        out = spt.sharepoint_search_files.invoke({"query": "Kinshasa"})
        assert out.startswith("No files matching 'Kinshasa'")
        assert "folders were looked through" in out

    def test_two_files_of_that_name_are_listed_not_guessed(self, tree):
        tree.files[f"{ROOT}/projects/old/Borings 2011.pdf"] = b"%PDF a"
        tree.files[f"{ROOT}/other/Borings 2011.pdf"] = b"%PDF b"
        out = _download(tree, path="projects/gone/Borings 2011.pdf")
        assert "nothing was downloaded" in out
        assert out.count("- [file] Borings 2011.pdf") == 3
        assert not any(n.endswith(".pdf") for n in os.listdir(tree.work))

    def test_absent_everywhere(self, tree):
        out = _download(tree, path="uploaded references/Nowhere.pdf")
        assert "not found" in out and "sharepoint_search_files" in out
        assert "SharePoint is answering" in out

    def test_the_apps_own_mirror_is_walked_last(self, tree):
        files, listed = spt._walk(tree, ROOT, budget=3)
        assert not any("/conversations/" in f["path"] for f in files)


# ---------------------------------------------------------------------------
# One local copy per SharePoint file, remembered by the conversation
# ---------------------------------------------------------------------------

class TestOneCopy:
    def test_a_second_name_and_a_refresh_keep_one_copy(self, tree):
        link = _r_link(f"{ROOT}/uploaded references/{REPORT}")
        _download(tree, path=link, save_as="report_full.pdf", refresh=True)
        again = _download(tree, path=f"APP/uploaded references/{REPORT}",
                          save_as="Report_renamed.pdf", refresh=True)
        assert "refreshed 'report_full.pdf' in place" in again
        assert sorted(os.listdir(tree.work)) == ["report_full.pdf"]
        reuse = _download(tree, path=link)
        assert "reusing that copy" in reuse

    def test_the_ledger_outlives_the_process(self, tree):
        _download(tree, path=f"uploaded references/{REPORT}")
        spt._DOWNLOADS.clear()                 # an app restart
        n = len(tree.downloads)
        out = _download(tree, path=f"uploaded references/{REPORT}")
        assert "reusing" in out and len(tree.downloads) == n
        led = core.load_downloads(tree.conv)
        assert led[0]["remote"] == f"{ROOT}/uploaded references/{REPORT}"
        assert led[0]["local"] == os.path.join(tree.work, REPORT)

    def test_a_different_file_never_overwrites_a_same_named_one(self, tree):
        with open(os.path.join(tree.work, "Other notes.pdf"), "wb") as fh:
            fh.write(b"the user's upload")
        out = _download(tree, path="uploaded references/Other notes.pdf")
        assert "Other notes_1.pdf" in out
        assert open(os.path.join(tree.work, "Other notes.pdf"), "rb").read() \
            == b"the user's upload"

    def test_identical_bytes_are_the_same_file(self, tree):
        with open(os.path.join(tree.work, "Other notes.pdf"), "wb") as fh:
            fh.write(b"%PDF other")
        _download(tree, path="uploaded references/Other notes.pdf")
        assert sorted(os.listdir(tree.work)) == ["Other notes.pdf"]
