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

from funhouse_agent.adapters import apply_aliases
from funhouse_agent.adapters._reference_common import (
    RESULT_BUDGET_CHARS, as_result, extract_method_info, make_wrapper, takes,
)

# Common guesses for the result-count parameter, mapped to ``limit``.
_SEARCH_ALIASES = {"max_results": "limit", "top_k": "limit",
                   "n_results": "limit", "k": "limit", "n": "limit"}


def _build():
    import inspect

    from geotech_references import _retrieval_db

    # geotech-references before REF-24 has no include_subsections (and no
    # subsection rows to inline); pass it only where taken.
    takes_include = "include_subsections" in inspect.signature(
        _retrieval_db.reference_get).parameters
    # ... and before REF-26 no results shaped to a size (offset, max_chars).
    shaped = takes(_retrieval_db.reference_search, "max_chars")

    def reference_search(query: str, reference: str = "",
                         chapter: int = 0, limit: int = 5,
                         offset: int = 0) -> dict:
        """Full-text search across all structured reference text, every
        section and subsection indexed with its own text; each hit's
        ``under`` names the sections above it.

        Returns ranked summary-only hits — call ``reference_get`` to
        fetch the full body. Search is the noise-reduction lever:
        run it first to scope, then drill in. The hit is the section
        that holds the words (a method's own subsection, e.g. GEC-12
        7.2.1.3.1 for "Nordlund", not only the chapter section above
        it). As many hits as fit one result: when they would not all
        fit, summaries are cut to 300 characters, then the lowest-ranked
        hits are left out, and ``next_offset`` and a ``note`` say how to
        get the rest.

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
            Max hits (default 5, capped at 50; fewer when they would
            not fit).
        offset : int, optional
            Skip this many of the best hits: the ``next_offset`` of the
            last call (default 0).

        Returns
        -------
        dict
            ``result``: summary-only hits, each with ``reference``,
            ``reference_title``, ``chapter``, ``chapter_title``,
            ``section_id``, ``title``, ``parent_id``, ``under`` (e.g.
            ``"7.2 Strength Limit States > 7.2.1 Nominal Axial ..."``;
            empty at the top level), ``summary``; ``shown``; and, when
            hits were left out, cut or more match, ``left_out``,
            ``next_offset`` and ``note``.
        """
        ref = reference if reference else None
        ch = int(chapter) if chapter else None
        if not shaped:
            hits = _retrieval_db.reference_search(
                query, reference=ref, chapter=ch,
                limit=int(limit) + int(offset or 0))
            return {"result": hits[int(offset or 0):]}
        out = _retrieval_db.reference_search(
            query, reference=ref, chapter=ch, limit=int(limit),
            offset=int(offset or 0), max_chars=RESULT_BUDGET_CHARS)
        if "error" in out:
            return {"result": [out]}
        return as_result(out, "hits")

    def reference_get(reference: str, section_id: str,
                      include_subsections: bool = False) -> dict:
        """Fetch the full body of one section by id, at any level; its
        subsections are listed in ``children`` (id and title), each
        fetched by its own id, not inlined.

        A missing id's error names the nearest section above it and the
        sections numbered under it.

        Parameters
        ----------
        reference : str
            Reference id (e.g. ``"dm7_1"``, ``"gec_12"``,
            ``"micropile"``).
        section_id : str
            Section id as it appears in the source, at any level.
            Examples: UFC hyphen-then-dot ``"4-2.1"``, ``"5-5.4"``;
            FHWA dot ``"4.4"``, ``"7.2.1.3.1"``; prologue ``"P-1"``.
        include_subsections : bool, optional
            Inline the subsections' text as ``subsections`` when the
            whole fits in 12,000 characters; otherwise they stay listed
            in ``children`` with a note. Default False.

        Returns
        -------
        dict
            Full section dict with body, key_points, applicability,
            equations (with implemented_in), figures, tables, plus
            ``under`` (the sections above it), ``parent_id`` and
            ``children`` (``section_id`` and ``title`` of each section
            directly under it).
        """
        if takes_include and include_subsections:
            return _retrieval_db.reference_get(
                reference, section_id, include_subsections=True)
        return _retrieval_db.reference_get(reference, section_id)

    def reference_query(sql: str) -> dict:
        """Run a read-only SELECT against the reference text database.

        Tables available:

        - ``sections`` — one row per section, at every level. Columns:
          ``rowid``, ``reference``, ``reference_title``, ``chapter``,
          ``chapter_title``, ``section_id``, ``title``, ``summary``,
          ``body``, ``key_points``, ``applicability``, ``equations_json``,
          ``figures_json``, ``tables_json``, ``parent_rowid`` (the row
          of the section above; NULL at the top level), ``depth``,
          ``parents_json``.
        - ``sections_fts`` — FTS5 virtual table. Use
          ``sections_fts MATCH '...'`` and ``bm25(sections_fts)``.

        Only single SELECT (or ``WITH ... SELECT``) statements are
        accepted. Connection is read-only. Result set capped at 50 rows,
        and at the whole rows that fit one result: select the columns you
        need (``substr(body, 1, 500)`` rather than ``body``; a section's
        whole text is ``reference_get``) and page with ``LIMIT`` /
        ``OFFSET``.

        Parameters
        ----------
        sql : str
            SELECT statement. Embed literal values directly. Multiple
            statements are rejected.

        Returns
        -------
        dict
            ``result``: the rows as dicts, or ``[{"error": "..."}]`` on
            failure; when rows were left out or a value cut,
            ``left_out`` and a ``note``.
        """
        if not shaped:
            return {"result": _retrieval_db.reference_query(sql)}
        out = _retrieval_db.reference_query(sql,
                                            max_chars=RESULT_BUDGET_CHARS)
        if "error" in out:
            return {"result": [out]}
        return as_result(out, "rows")

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

    # reference_search: accept the commonly-guessed result-count names
    # (max_results/top_k/...) as aliases for ``limit``.
    _search_wrapped = registry["reference_search"]

    def _search_with_aliases(params):
        return _search_wrapped(apply_aliases(params, _SEARCH_ALIASES))

    _search_with_aliases.__wrapped__ = reference_search
    registry["reference_search"] = _search_with_aliases
    info["reference_search"]["parameters"]["limit"]["description"] = (
        info["reference_search"]["parameters"]["limit"].get("description", "")
        + " (aliases: max_results, top_k, n_results)"
    ).strip()
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
