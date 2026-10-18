"""Figure-catalog retrieval adapter — SQLite FTS5 over every reference's
digitized figure catalog (``<reference>/figures_catalog.json``).

Companion to ``reference_db_adapter`` (which searches chapter *text*). Use this
to locate an engineering chart/figure by meaning and resolve it to a renderable
PDF page, then read a value off it with the ``read_reference_figure`` vision
tool.

- ``figure_search(query, reference="", chapter=0, limit=5)`` — ranked figure
  hits (number, caption, page) over captions + cross-linked chapter context.
- ``figure_get(reference, figure_number)`` — full catalog row incl. the source
  ``pdf_path`` and 0-based ``pdf_page_index``.
- ``list_indexed_figures()`` — inventory of references with a figure catalog.
"""

from __future__ import annotations

from funhouse_agent.adapters import declare_aliases
from funhouse_agent.adapters._reference_common import (
    RESULT_BUDGET_CHARS, as_result, extract_method_info, make_wrapper, takes,
)

#: Names models guess for figure_search's result cap; its catch-all reads
#: these as ``limit`` and nothing else.
_LIMIT_ALIASES = ("top_k", "k", "n", "max_results", "top", "num_results")


def _build():
    from geotech_references import _figures_db

    # geotech-references before REF-26 has no paged or size-shaped figure
    # search; pass the new options only where taken.
    pages = takes(_figures_db.figure_search, "offset")
    fit_hits = None
    if takes(_figures_db.figure_get, "max_chars"):
        from geotech_references._retrieval import fit_hits

    def figure_search(query: str, reference: str = "",
                      chapter: int = 0, limit: int = 5, offset: int = 0,
                      **kwargs) -> list:
        """Full-text search the figure catalog by caption and concept.

        Captions use notation (``KA``/``KP``); the index also carries concept
        vocabulary cross-linked from the chapter text (e.g. "passive earth
        pressure"), so natural queries work. If a strict (all-words) match finds
        nothing, an OR-of-terms fallback recovers the best partial matches.

        Parameters
        ----------
        query : str
            FTS5 MATCH query. Plain words are AND-matched with porter stemming;
            ``"phrase"``, ``OR``, ``NEAR()``, ``col:term`` operators work.
        reference : str, optional
            Reference id to scope to (e.g. ``"dm7_2"``). Empty = all.
        chapter : int, optional
            Chapter number to scope to (prologue figures are chapter 0).
            ``0`` means no chapter filter.
        limit : int, optional
            Max hits (default 5, capped at 50; fewer when they would not
            fit one result: captions are cut to 300 characters first, then
            the lowest-ranked hits left out, and ``next_offset`` and a
            ``note`` say how to get the rest).
        offset : int, optional
            Skip this many of the best hits: the ``next_offset`` of the
            last call (default 0).

        Returns
        -------
        list
            Ranked hits, each with ``reference``, ``reference_title``,
            ``figure_number``, ``caption``, ``chapter``, ``pdf_page_index``,
            ``printed_page``, and a ``read_value`` next-step instruction: a
            chart value must be read off the rendered figure via
            ``read_reference_figure``, never inferred from the caption or memory.
        """
        # Models commonly guess a different name for the result cap; accept the
        # usual ones rather than hard-failing the call on an unknown kwarg.
        for alt in _LIMIT_ALIASES:
            if kwargs.get(alt):
                try:
                    limit = int(kwargs[alt])
                except (TypeError, ValueError):
                    pass
        ref = reference if reference else None
        ch = int(chapter) if chapter else None
        if pages:
            hits = _figures_db.figure_search(
                query, reference=ref, chapter=ch, limit=int(limit),
                offset=int(offset or 0))
        else:
            hits = _figures_db.figure_search(
                query, reference=ref, chapter=ch,
                limit=int(limit) + int(offset or 0))[int(offset or 0):]
        # Starve the shortcut: a number that lives in a chart must come from the
        # rendered image, not this caption or the model's training memory. The
        # search result only *identifies* the figure; it carries no values. So
        # signpost every real hit with the exact next call, making the vision
        # read-off the path of least resistance.
        for hit in hits:
            if isinstance(hit, dict) and hit.get("figure_number") and "error" not in hit:
                hit["read_value"] = (
                    "To read a numeric value off this chart, call "
                    f"read_reference_figure(reference='{hit['reference']}', "
                    f"figure_number='{hit['figure_number']}', "
                    "prompt='<value(s) needed>'). Do NOT report a chart value "
                    "from this caption or from memory."
                )
        return hits

    def figure_get(reference: str, figure_number: str) -> dict:
        """Fetch the full catalog row for one figure.

        Parameters
        ----------
        reference : str
            Reference id (e.g. ``"dm7_2"``).
        figure_number : str
            Figure id as in the source (e.g. ``"4-12"``, ``"P-1"``, ``"B-3"``).
            A leading ``"Figure "`` is tolerated.

        Returns
        -------
        dict
            Full row incl. ``caption``, ``description``, ``chapter``,
            ``pdf_path``, ``pdf_page_index`` (0-based for rendering),
            ``printed_page``, ``page_estimated``. Pass ``reference`` and
            ``figure_number`` to ``read_reference_figure`` to read a value off
            the chart with vision. A row too long for one result (a few
            catalog captions run on into the page text) has its caption cut,
            and a ``note`` says so.
        """
        if fit_hits is not None:
            return _figures_db.figure_get(reference, figure_number,
                                          max_chars=RESULT_BUDGET_CHARS)
        return _figures_db.figure_get(reference, figure_number)

    def list_indexed_figures() -> list:
        """Inventory of references that have an indexed figure catalog.

        Returns
        -------
        list
            One dict per reference with ``reference``, ``reference_title``,
            ``n_figures``, ``n_chapters``.
        """
        return _figures_db.list_indexed_figures()

    funcs = {
        "figure_search": figure_search,
        "figure_get": figure_get,
        "list_indexed_figures": list_indexed_figures,
    }
    registry = {}
    info = {}
    for name, fn in funcs.items():
        registry[name] = make_wrapper(fn)
        info[name] = extract_method_info(
            fn, "Figure DB (FTS5)", "geotech_references figure catalogs"
        )

    def _search_fitted(params):
        """figure_search as the agent gets it: the signposted hits, as many
        as fit one result (BACKLOG REF-26; 50 hits ran to 68,859
        characters), one more asked for to learn whether more match."""
        params = dict(params)
        limit = int(params.get("limit") or 5)
        for alt in _LIMIT_ALIASES:
            if params.get(alt):
                try:
                    limit = int(params.pop(alt))
                except (TypeError, ValueError):
                    params.pop(alt)
        limit = max(1, min(limit, 50))
        offset = max(0, int(params.get("offset") or 0))
        hits = figure_search(**dict(params, limit=limit + 1))
        if fit_hits is None or any("error" in h for h in hits):
            return {"result": hits[:limit]}
        more = len(hits) > limit
        if not more and pages and len(hits) == limit:
            # the library gives at most 50: ask for the one after them
            after = figure_search(**dict(params, limit=1,
                                         offset=offset + limit))
            more = bool(after) and not any("error" in h for h in after)
        return as_result(fit_hits(
            hits[:limit], RESULT_BUDGET_CHARS, text_fields=("caption",),
            offset=offset, more=more,
            max_offset=getattr(_figures_db, "_MAX_OFFSET", None),
            narrow="narrow the query (more specific words, or a 'reference')",
            fetch="figure_get returns a figure's whole catalog row"), "hits")

    _search_fitted.__wrapped__ = figure_search
    registry["figure_search"] = _search_fitted
    # figure_search's catch-all reads only the result-cap names above, so
    # dispatch checks it against its parameters and these (GM-110).
    info["figure_search"].pop("open_parameters", None)
    declare_aliases(info, {"figure_search": dict.fromkeys(_LIMIT_ALIASES,
                                                          "limit")})
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
