"""This repository's Document Review files are frozen (BACKLOG INF-33, the
app split's step 13; plan ``module_work/APP_SPLIT_PLAN.md`` Step 13, "From
this day").

Since step 13 Document Review is its own package,
``packages/aec-document-review`` (``aec_document_review``), published to
PyPI as ``aec-document-review``. Its page, its review files and its tasks
were copied there; the copies here stay only so GeotechStaffEngineer's
two-page app runs unchanged until step 15 takes them out. So a change to the
review page goes to the PACKAGE, not here, and this test fails if one of the
files below changes: the review files (plan 2.2), the review profile's file
and the Tiny Apps entry, and the review tasks.

If it fails:

- you meant to change the Document Review page: make the change in
  ``packages/aec-document-review/aec_document_review/`` (the same file, by
  the names in ``module_work/inf33/copy_step13.py``) and undo it here;
- the change is to GeotechStaffEngineer's own four geotech-page tasks in
  ``funhouse_agent/review_eval/tasks.py``, or a fix the two-page app needs
  before step 15: make it, say why in the commit, and update the hash below
  in the same commit.
"""

import hashlib
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

#: path -> sha256 of the file with LF line ends, at the copy (step 13).
FROZEN = {
    "funhouse_agent/deep/review_agent.py":
        "cb08918acb890c0bd3c0c3ac435062e2965f112def2e7e36fad28c5b2ca44ee7",
    "funhouse_agent/deep/minimal_agent.py":
        "2981f42104e7d7d5b3bb9dcb7f9a4c14ba125160b781c414bf3933a980c4c4b9",
    "funhouse_agent/review_prompt.py":
        "b8a6ed16820e4a5801cb0a6d64841d920fa33de756ef5ffbf4529f110b83f131",
    "funhouse_agent/review_build.py":
        "bba9877c10fa21176e56a8c70869fd5d6312f7da80fb0c749a381a9289d3e7dc",
    "webapp/review_page.py":
        "c1cfb9f80abd470329feb40d8a49826f81ad50c1c1fdc620540c14746243b8ec",
    "webapp/tinyapps_entry.py":
        "cc11eb037758dfbb27ce51ed21ead1c0e713975bcde4d7f83b11e1eadc26cc24",
    "funhouse_agent/review_eval/tasks.py":
        "fb90ed5d0a6254a02f613ec72613afa0f5ade5d0baeb5454bfe9bf7a603d4bb2",
}


def _sha(path: str) -> str:
    with open(os.path.join(REPO, path), "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


@pytest.mark.parametrize("path", sorted(FROZEN))
def test_the_review_file_is_unchanged(path):
    assert _sha(path) == FROZEN[path], (
        f"{path} changed, but this repository's Document Review files are "
        "frozen since the app split's step 13 (BACKLOG INF-33). Review-page "
        "changes go to the package: packages/aec-document-review/"
        "aec_document_review/ (published as aec-document-review). If this "
        "is GSE's own change (its geotech-page tasks, or a fix the two-page "
        "app needs before step 15), say why in the commit and update the "
        "hash in webapp/tests/test_review_freeze.py. See this test's "
        "docstring.")


def test_each_frozen_file_has_its_copy_in_the_package():
    pkg = os.path.join(REPO, "packages", "aec-document-review",
                       "aec_document_review")
    copies = {
        "funhouse_agent/deep/review_agent.py": "review_agent.py",
        "funhouse_agent/deep/minimal_agent.py": "minimal_agent.py",
        "funhouse_agent/review_prompt.py": "review_prompt.py",
        "funhouse_agent/review_build.py": "review_build.py",
        "webapp/review_page.py": "review_page.py",
        "webapp/tinyapps_entry.py": "tinyapps_entry.py",
        "funhouse_agent/review_eval/tasks.py": "review_eval/tasks.py",
    }
    assert sorted(copies) == sorted(FROZEN)
    missing = [c for c in copies.values()
               if not os.path.isfile(os.path.join(pkg, c))]
    assert not missing, missing
