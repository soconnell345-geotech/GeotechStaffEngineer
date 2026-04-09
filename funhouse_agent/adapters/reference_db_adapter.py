"""Cross-reference text retrieval adapter — SQLite FTS5 over all
structured chapter text in geotech_references (DM7 + GEC + micropile).

Exposes three tools optimized for noise-reduced LLM agent retrieval:

- ``reference_search(query, reference="", chapter=0, limit=5)`` — ranked
  BM25 hits returning *summary only*.
- ``reference_get(reference, section_id)`` — full section body.
- ``reference_query(sql)`` — read-only SELECT against the backing DB.
- ``list_indexed_references()`` — inventory.

This adapter sits alongside the per-reference adapters (dm7, gec6, …)
which keep their existing tables/figures/equations/legacy text APIs
unchanged. Use this one for cross-reference search and noise-reduced
drill-in.
"""

from __future__ import annotations

from funhouse_agent.adapters._reference_common import (
    extract_method_info, make_wrapper,
)


def _build():
    from geotech_references import _retrieval_db

    def reference_search(query: str, reference: str = "",
                         chapter: int = 0, limit: int = 5) -> list:
        """Full-text search across all structured reference text.

        Returns ranked summary-only hits — call ``reference_get`` to
        fetch the full body. Search is the noise-reduction lever:
        run it first to scope, then drill in.

        Parameters
        ----------
        query : str
            FTS5 MATCH query. Plain words are AND-matched with porter
            stemming. Use quotes for phrases (``"primary consolidation"``),
            ``OR`` for alternatives, ``NEAR()`` for proximity,
            ``col:term`` to scope to one column (title/summary/body/
            key_points/applicability).
        reference : str, optional
            Reference id to scope to (e.g. ``"dm7_1"``, ``"gec_12"``).
            Empty string searches all references.
        chapter : int, optional
            Chapter number to scope to (only meaningful with
            ``reference``). ``0`` means no chapter filter.
        limit : int, optional
            Max hits (default 5, capped at 50).

        Returns
        -------
        list
            Summary-only hits, each with ``reference``, ``reference_title``,
            ``chapter``, ``chapter_title``, ``section_id``, ``title``,
            ``summary``.
        """
        ref = reference if reference else None
        ch = int(chapter) if chapter else None
        return _retrieval_db.reference_search(
            query, reference=ref, chapter=ch, limit=int(limit)
        )

    def reference_get(reference: str, section_id: str) -> dict:
        """Fetch the full body of one section by id.

        Parameters
        ----------
        reference : str
            Reference id (e.g. ``"dm7_1"``, ``"gec_12"``,
            ``"micropile"``).
        section_id : str
            Section id as it appears in the source. Examples:
            UFC hyphen-then-dot ``"4-2.1"``, ``"5-5.4"``;
            FHWA dot ``"5.7.2"``; prologue ``"P-1"``.

        Returns
        -------
        dict
            Full section dict with body, key_points, applicability,
            equations (with implemented_in), figures, tables.
        """
        return _retrieval_db.reference_get(reference, section_id)

    def reference_query(sql: str) -> list:
        """Run a read-only SELECT against the reference text database.

        Tables available:

        - ``sections`` — one row per section. Columns: ``reference``,
          ``reference_title``, ``chapter``, ``chapter_title``,
          ``section_id``, ``title``, ``summary``, ``body``,
          ``key_points``, ``applicability``, ``equations_json``,
          ``figures_json``, ``tables_json``.
        - ``sections_fts`` — FTS5 virtual table. Use
          ``sections_fts MATCH '...'`` and ``bm25(sections_fts)``.

        Only single SELECT (or ``WITH ... SELECT``) statements are
        accepted. Connection is read-only. Result set capped at 50 rows.

        Parameters
        ----------
        sql : str
            SELECT statement. Embed literal values directly. Multiple
            statements are rejected.

        Returns
        -------
        list
            Result rows as dicts, or ``[{"error": "..."}]`` on failure.
        """
        return _retrieval_db.reference_query(sql)

    def list_indexed_references() -> list:
        """Inventory of references currently in the FTS DB.

        Use this to discover what's available before searching.

        Returns
        -------
        list
            One dict per reference with ``reference``, ``reference_title``,
            ``n_chapters``, ``n_sections``.
        """
        return _retrieval_db.list_indexed_references()

    funcs = {
        "reference_search": reference_search,
        "reference_get": reference_get,
        "reference_query": reference_query,
        "list_indexed_references": list_indexed_references,
    }
    registry = {}
    info = {}
    for name, fn in funcs.items():
        registry[name] = make_wrapper(fn)
        info[name] = extract_method_info(
            fn, "Reference DB (FTS5)", "geotech_references SQLite FTS5"
        )
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
