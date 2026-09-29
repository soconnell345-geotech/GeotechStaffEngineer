"""Per-document digests for large reviews - the FREE layer (code only).

WHY. A review of hundreds of pages, or of several PDFs, cannot be carried by
one agent loop: every page it reads stays in its context. The digest is a
per-document, discipline-neutral store the reviewing agent holds SUMMARIES of
and pulls DETAIL from (plan of record ``module_work/REVIEW_ARCHITECTURE.md``,
shape 2, Stage A). This package is its free layer: pure code over planlens,
no model call anywhere, built in seconds the first time a document is asked
about, cached under the conversation's working folder and reused across
turns - keyed by the document's content, so a re-upload reuses it.

WHAT. ``build(source, name, root)`` writes ``<root>/<sha256[:16]>/``:
``inventory.json``, ``pages.jsonl``, ``index.sqlite`` (FTS5 over every text
line and markup) and ``references.json`` (sheet, detail, standard,
specification-section, table, figure and appendix references with page and
box) and returns a :class:`Digest` - ``inventory()``, ``pages(spec)``,
``search(query, pages, kinds, limit)``, ``references(target)``.
``inventory_of(sources, root)`` does several uploads at once and says which
cited sheets and standards no upload contains.

WHAT IT IS NOT. It reads the PDF's text; it never reads a picture. A page it
flags ``needs_look`` (a drawing sheet, figure, scan or form, or text that is
not what the page says) must still be looked at to know what it shows.
"""

from funhouse_agent.review_digest.build import (
    DEFAULT_SHAPE1_PAGES, DIGEST_DIRNAME, NEEDS_LOOK_KINDS, SHAPE1_PAGES_ENV,
    build, cross_references, default_root, inventory_of, root_for, sha256_of,
    shape1_pages)
from funhouse_agent.review_digest.digest import (
    FORMAT_VERSION, Digest, DigestError, clear_cache, load_digest, parse_pages)
from funhouse_agent.review_digest.references import (
    find_references, label_matches)

__all__ = ["build", "inventory_of", "cross_references", "default_root",
           "root_for", "sha256_of", "shape1_pages", "Digest", "DigestError",
           "FORMAT_VERSION", "load_digest", "clear_cache", "parse_pages",
           "find_references", "label_matches", "DIGEST_DIRNAME",
           "NEEDS_LOOK_KINDS", "SHAPE1_PAGES_ENV", "DEFAULT_SHAPE1_PAGES"]
