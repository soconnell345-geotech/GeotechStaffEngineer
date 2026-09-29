"""Cache folders in the working folder never become download cards."""

import os

from webapp import core


def test_digest_cache_is_not_an_artifact(tmp_path):
    work = tmp_path / "files"
    (work / "digest" / "abc123").mkdir(parents=True)
    before = core.snapshot_dir(str(work))
    (work / "digest" / "abc123" / "index.sqlite").write_bytes(b"x")
    (work / "review_memo.docx").write_bytes(b"y")
    nested = work / "sub" / "digest"
    nested.mkdir(parents=True)
    (nested / "kept.txt").write_text("a folder named digest deeper down is not the cache")
    new = core.new_artifacts(str(work), before, [])
    names = {os.path.relpath(p, work) for p in new}
    assert "review_memo.docx" in names
    assert os.path.join("sub", "digest", "kept.txt") in names
    assert not any(n.startswith("digest") for n in names)
