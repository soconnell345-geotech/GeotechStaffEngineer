"""A review method for a drawing set or a specification (BACKLOG DR-06).

A DRAFT, to be measured before it is anyone's default. The page's prompt is
thorough on how to READ a document and says nothing on how to REVIEW one;
these two files are a method for a set review, written as deployment
instructions (``aec_agent_core.deep.instructions``):

- ``main.md``, for the page's agent: the set before the sheets, the checks
  on one sheet, the checks that need two places, what a comment may rest on
  (the document, a document the user supplied, or general practice named as
  such; never a clause number from memory), and the comment log;
- ``general-purpose.md``, for the reading helper: what to bring back from a
  sheet so the reviewer can compare sheets.

Nothing loads them unless ``GEOTECH_INSTRUCTIONS`` names this folder
(:data:`FOLDER`). Then each role's text goes last in its prompt, under the
built-in rules, which it cannot switch off. The review suite measures it as
an arm (:func:`aec_document_review.review_eval.method_arm`).

It names no discipline's checklist, cites no code or standard, and ranks
no comment by severity: the page reviews architectural, civil, structural
and MEP sets alike, has no reference library, and leaves the judgement to
its reader.
"""

import os

#: The folder to name in ``GEOTECH_INSTRUCTIONS``.
FOLDER = os.path.dirname(os.path.abspath(__file__))

#: The roles the method has a file for.
ROLES = ("main", "general-purpose")


def text(role: str) -> str:
    """The method's text for ``role`` (``main`` or ``general-purpose``)."""
    if role not in ROLES:
        raise KeyError(role)
    with open(os.path.join(FOLDER, role + ".md"), encoding="utf-8") as fh:
        return fh.read()


__all__ = ["FOLDER", "ROLES", "text"]
