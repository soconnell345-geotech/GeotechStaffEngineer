"""Document Review's professional-use notice, shown at the top of the page.

The wording is a PLACEHOLDER until the owner decides the page's own notice
(BACKLOG OWN-78): GeotechStaffEngineer's notice speaks of geotechnical
analysis and SI units, which are not this page's; this one keeps its points
(an aid, not a deliverable; the judgement stays with the professional; no
engagement; no warranty) for documents of any discipline.
"""

FULL_NOTICE = """\
Document Review -- Professional-use notice
==========================================

Document Review is a REVIEW AID -- a multiplier for a qualified reviewer's
judgement, not a replacement for it. It reads drawings, specifications,
reports and calculations with an LLM agent that looks at the pages, and it
drafts memos and marked-up copies; it does not know your project and cannot
exercise professional judgement.

What it is / is not
  - An agent that reads, looks at and summarises documents, and drafts Word
    memos and marked-up PDFs for a reviewer to check.
  - NOT a professional opinion, a stamped review, or a substitute for the
    review the documents require.
  - Using this software creates NO professional engagement with its authors
    or contributors.

Your responsibilities
  - Check every finding, quote, count and citation against the document
    before relying on it. Its answers are candidates to be checked.
  - Treat its readings of drawings and scanned pages with particular care:
    it can misread small lettering, miss a sheet, or mis-measure.
  - Every markup it writes is signed as an AI draft; review it before the
    copy goes to anyone else.

No warranty
  - Provided under the MIT License "AS IS", WITHOUT WARRANTY OF ANY KIND. The
    authors are not liable for any claim, damages, or other liability arising
    from its use. Responsibility for any review remains with the professional
    who adopts it. See the package's LICENSE for the full terms.
"""


def notice_text() -> str:
    """The notice, as the page shows it."""
    return FULL_NOTICE.strip()
