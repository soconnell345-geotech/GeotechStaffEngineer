"""Tests for the list_files extended tool — read-only real-directory listing.

list_files is the discovery tool: the agent browses the user's REAL folders
(the scratch filesystem cannot see real paths) before reading a report or
choosing a save destination. It must list entries with type/size/mtime, sort
dirs first, cap output, recurse only when asked (bounded), and return a clear
error for a missing/unreadable path — never raise.
"""

import json

from funhouse_agent.vision_tools import (
    EXTENDED_TOOLS,
    _dispatch_list_files,
    dispatch_extended_tool,
)


def _list(path, **kw):
    args = {"path": str(path)}
    args.update(kw)
    return json.loads(_dispatch_list_files(args))


class TestListFilesBasic:
    def test_in_extended_tool_set(self):
        assert "list_files" in EXTENDED_TOOLS

    def test_lists_immediate_children_with_metadata(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "sub").mkdir()
        res = _list(tmp_path)
        assert res["n_entries"] == 2
        by_name = {e["name"]: e for e in res["entries"]}
        assert by_name["a.txt"]["type"] == "file"
        assert by_name["a.txt"]["size_bytes"] == 5
        assert by_name["a.txt"]["modified"]  # a formatted timestamp
        assert by_name["sub"]["type"] == "dir"
        assert by_name["sub"]["size_bytes"] is None

    def test_dirs_sorted_first(self, tmp_path):
        (tmp_path / "b_file.txt").write_text("x")
        (tmp_path / "a_file.txt").write_text("x")
        (tmp_path / "z_dir").mkdir()
        res = _list(tmp_path)
        types = [e["type"] for e in res["entries"]]
        assert types[0] == "dir"  # z_dir first despite the 'z' name
        # files that follow are alphabetical
        file_names = [e["name"] for e in res["entries"] if e["type"] == "file"]
        assert file_names == ["a_file.txt", "b_file.txt"]

    def test_default_is_not_recursive(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_text("x")
        res = _list(tmp_path)
        names = {e["name"] for e in res["entries"]}
        assert names == {"sub"}  # deep.txt NOT included at depth 0

    def test_depth_one_recurses_one_level(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "deep.txt").write_text("x")
        (tmp_path / "sub" / "sub2").mkdir()
        (tmp_path / "sub" / "sub2" / "deeper.txt").write_text("x")
        res = _list(tmp_path, depth=1)
        names = {e["name"] for e in res["entries"]}
        assert "sub/deep.txt" in names
        assert "sub/sub2" in names
        # depth 1 does NOT reach two levels down
        assert "sub/sub2/deeper.txt" not in names

    def test_depth_capped_at_two(self, tmp_path):
        # A 4-deep tree; depth=99 is clamped to 2.
        p = tmp_path
        for d in ("l1", "l2", "l3", "l4"):
            p = p / d
            p.mkdir()
            (p / "f.txt").write_text("x")
        res = _list(tmp_path, depth=99)
        names = {e["name"] for e in res["entries"]}
        assert "l1/l2/l3" in names          # reached (2 levels below root)
        assert "l1/l2/l3/l4" not in names   # 3 levels below root — not reached


class TestListFilesErrors:
    def test_missing_path_is_error(self, tmp_path):
        res = _list(tmp_path / "does_not_exist")
        assert "error" in res
        assert "not found" in res["error"].lower()

    def test_file_path_reports_file_not_dir(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 ...")
        res = _list(f)
        assert res.get("is_file") is True
        assert res["size_bytes"] == len(b"%PDF-1.4 ...")
        assert "note" in res

    def test_entry_for_handles_stat_error(self):
        """A child whose stat() fails is recorded with an error, not raised."""
        from funhouse_agent.vision_tools import _list_entry_for

        class FakeEntry:
            name = "x"

            def is_dir(self, follow_symlinks=True):
                return False

            def stat(self, follow_symlinks=True):
                raise OSError("permission denied")

        entry = _list_entry_for(FakeEntry(), "x")
        assert entry["type"] == "file"
        assert "error" in entry


class TestListFilesCaps:
    def test_max_entries_truncates_with_nudge(self, tmp_path):
        for i in range(20):
            (tmp_path / f"f{i:02d}.txt").write_text("x")
        res = _list(tmp_path, max_entries=5)
        assert res["n_entries"] == 5
        assert res.get("truncated") is True
        assert "narrow" in res["truncated_note"].lower()

    def test_char_budget_keeps_result_small(self, tmp_path):
        # Many long-named files: output stays bounded (valid JSON, not a
        # mid-structure string cut) and flags truncation.
        for i in range(400):
            (tmp_path / (f"file_{i:03d}_" + "n" * 60 + ".txt")).write_text("x")
        raw = _dispatch_list_files({"path": str(tmp_path), "max_entries": 400})
        res = json.loads(raw)  # must parse — never truncated into invalid JSON
        assert res.get("truncated") is True
        assert len(raw) < 16000

    def test_dispatch_via_extended_tool_entry(self, tmp_path):
        (tmp_path / "a.txt").write_text("x")
        raw = dispatch_extended_tool(
            "list_files", {"path": str(tmp_path)},
            engine=None, attachments=None, save_fn=None,
        )
        assert json.loads(raw)["n_entries"] == 1
