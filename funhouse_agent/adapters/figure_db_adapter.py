"""Figure-catalog retrieval adapter — SQLite FTS5 over every reference's
digitized figure catalog (``<reference>/figures_catalog.json``).

Companion to ``reference_db_adapter`` (which searches chapter *text*). Use this
to locate an engineering chart/figure by meaning and resolve it to a renderable
PDF page, then read a value off it with the ``read_reference_figure`` vision
tool.

- ``figure_search(query, reference="", chapter=0, limit=5)`` — ranked figure
  hits (number, caption, page) over captions + cross-linked chapter context.
- ``figure_get(reference, figure_number, pdf_page_index=None)`` — full catalog
  row incl. the source ``pdf_path`` and 0-based ``pdf_page_index`` (which
  page, for a number the catalog holds more than once).
- ``list_indexed_figures()`` — inventory of references with a figure catalog.
"""

from __future__ import annotations

import json

from funhouse_agent.adapters import declare_aliases
from funhouse_agent.adapters._reference_common import (
    RESULT_BUDGET_CHARS, as_result, extract_method_info, make_wrapper, takes,
)

#: Names models guess for figure_search's result cap; its catch-all reads
#: these as ``limit`` and nothing else.
_LIMIT_ALIASES = ("top_k", "k", "n", "max_results", "top", "num_results")

#: Room kept for each hit's ``read_value`` signpost when the library fits
#: the hits to a size (the longest signpost is about 330 characters), and
#: the least the library is given to fit them in.
_SIGNPOST_ROOM = 340
_LEAST_LIBRARY_BUDGET = 4000


def _chapter_scope(chapter):
    """A figure search's chapter as the catalogs file it: ``None`` for no
    filter (0 or empty), a number, or a label such as the AASHTO 1993
    Guide's ``"II-2"`` (the catalogs file chapters as both)."""
    if chapter in (None, "", 0, "0"):
        return None
    text = str(chapter).strip()
    return int(text) if text.isdigit() else text


def _build():
    from geotech_references import _figures_db

    # geotech-references before REF-26 has no paged or size-shaped figure
    # search; pass the new options only where taken.
    pages = takes(_figures_db.figure_search, "offset")
    fit_hits = None
    if takes(_figures_db.figure_get, "max_chars"):
        from geotech_references._retrieval import fit_hits
    # Since the 2026-11-10 review the library fits a search to a size
    # itself and says how each hit matched (``matched``, ``words_not_found``,
    # an empty result's scope), and a figure number held more than once
    # (a figure over several sheets, a number printed twice) is fetched by
    # its page.
    library_fits = takes(_figures_db.figure_search, "max_chars")
    by_page = takes(_figures_db.figure_get, "pdf_page_index")

    def _signpost(hit: dict) -> None:
        """Every real hit gets the exact next call (see figure_search)."""
        if not (isinstance(hit, dict) and hit.get("figure_number")
                and "error" not in hit):
            return
        page = ""
        if by_page and hit.get("pdf_page_index") not in (None, ""):
            page = f", pdf_page_index={hit['pdf_page_index']}"
            if hit.get("pages"):
                page += " (or another sheet's from 'pages')"
        hit["read_value"] = (
            "To read a numeric value off this chart, call "
            f"read_reference_figure(reference='{hit['reference']}', "
            f"figure_number='{hit['figure_number']}'{page}, "
            "prompt='<value(s) needed>'). Do NOT report a chart value "
            "from this caption or from memory."
        )

    def figure_search(query: str, reference: str = "",
                      chapter: int = 0, limit: int = 5, offset: int = 0,
                      **kwargs) -> list:
        """Full-text search the figure catalog by caption and concept.

        Captions use notation (``KA``/``KP``); the index also carries concept
        vocabulary cross-linked from the chapter text (e.g. "passive earth
        pressure"), so natural queries work. The figures holding every word
        come first; the places left are filled by figures holding some of
        the words, each marked ``match: "any word"`` with ``lacks`` (the
        words it does not hold): read such a hit as no match for the whole
        query.

        Parameters
        ----------
        query : str
            FTS5 MATCH query, words matched with porter stemming;
            ``"phrase"``, ``OR``, ``NEAR()``, ``col:term`` operators work.
        reference : str, optional
            Reference id to scope to (e.g. ``"dm7_2"``), another spelling
            of it or a report number its catalog carries. ``"dm7"`` is two
            catalogs: give ``"dm7_1"`` or ``"dm7_2"``. A name with no
            catalog is an error naming the catalog ids. Empty = all.
        chapter : int or str, optional
            Chapter to scope to, as the catalog files it (prologue figures
            are chapter 0; the AASHTO 1993 Guide's are ``"II-2"``). A
            chapter with no figures there is an error
            naming the chapters that have some. ``0`` means no chapter
            filter.
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
            ``reference_id`` (the report number of the figure's volume),
            ``figure_number``, ``caption``, ``chapter``, ``pdf_page_index``,
            ``printed_page``, ``pages`` (every sheet's ``pdf_page_index``
            and ``printed_page``) on a figure printed over several sheets,
            ``match`` (with ``lacks``) on a hit that does not hold every
            word, and a ``read_value`` next-step instruction: a chart value
            must be read off the rendered figure via
            ``read_reference_figure``, never inferred from the caption or
            memory. As the agent gets it the hits are under ``result``,
            with ``matched`` and ``words_not_found`` when some do not hold
            every word, ``resolved_reference`` when ``reference`` was not
            itself an id, and an empty result's ``note`` naming the scope
            searched and the words no figure there holds.
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
        ch = _chapter_scope(chapter)
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
            _signpost(hit)
        return hits

    def figure_get(reference: str, figure_number: str,
                   pdf_page_index: int = None) -> dict:
        """Fetch the full catalog row for one figure.

        Parameters
        ----------
        reference : str
            Reference id (e.g. ``"dm7_2"``).
        figure_number : str
            Figure id as in the source (e.g. ``"4-12"``, ``"P-1"``, ``"B-3"``).
            A leading ``"Figure "`` is tolerated.
        pdf_page_index : int, optional
            Which page, for a figure number the catalog holds more than
            once (a figure printed over several sheets, or a number the
            print gives two figures): a search hit's ``pdf_page_index``,
            or one from its ``pages``. Without it such a number is an
            error naming each page.

        Returns
        -------
        dict
            Full row incl. ``caption``, ``description``, ``chapter``,
            ``pdf_path``, ``pdf_page_index`` (0-based for rendering),
            ``printed_page``, ``page_estimated``. Pass ``reference``,
            ``figure_number`` (and ``pdf_page_index``) to
            ``read_reference_figure`` to read a value off the chart with
            vision. A row too long for one result (a few catalog captions
            run on into the page text) has its caption cut, and a ``note``
            says so.
        """
        kwargs = {}
        if by_page and pdf_page_index not in (None, ""):
            kwargs["pdf_page_index"] = int(pdf_page_index)
        if fit_hits is not None:
            kwargs["max_chars"] = RESULT_BUDGET_CHARS
        return _figures_db.figure_get(reference, figure_number, **kwargs)

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
        if library_fits:
            return _library_fitted(params, limit, offset)
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

    def _library_fitted(params, limit: int, offset: int):
        """The library's own fitted search (how each hit matched, the
        scope an empty result searched, the volume a report number named),
        with room kept for the signposts; its refusals (a name with no
        catalog, "dm7", a chapter with no figures) come back as errors
        with their message, never as "nothing found"."""
        budget = max(RESULT_BUDGET_CHARS - _SIGNPOST_ROOM * limit,
                     _LEAST_LIBRARY_BUDGET)
        result = trial = None
        for _ in range(4):
            out = _figures_db.figure_search(
                params.get("query", ""),
                reference=params.get("reference") or None,
                chapter=_chapter_scope(params.get("chapter")), limit=limit,
                offset=offset, max_chars=budget)
            if "error" in out:
                return {"result": [out]}
            library_size = len(json.dumps(out, default=str))
            for hit in out.get("hits") or []:
                _signpost(hit)
            trial = as_result(out, "hits")
            size = len(json.dumps(trial, default=str))
            shown = len(trial["result"])
            if size > RESULT_BUDGET_CHARS:
                if result is not None or not shown:
                    # a larger budget overshot: back off by the excess and
                    # one hit, once more
                    budget -= int(size - RESULT_BUDGET_CHARS
                                  + size / max(shown, 1))
                    if budget < _LEAST_LIBRARY_BUDGET:
                        break
                    continue
                break
            result, result_budget = trial, budget
            if not (trial.get("left_out") and shown):
                break
            # hits were left out for size with room still to spare (the
            # room kept for signposts was more than they took): give the
            # library room for as many more hits as fit with their
            # signposts
            library_per_hit = library_size / shown
            per_hit = library_per_hit + (size - library_size) / shown
            more = int((RESULT_BUDGET_CHARS - size - 300) // per_hit)
            if more < 1:
                break
            budget += int(more * library_per_hit)
        if result is None:
            result, result_budget = trial, budget
        elif isinstance(result.get("note"), str):
            # the hits the note says would not fit would not fit the whole
            # result, signposts and all
            result["note"] = result["note"].replace(
                f"{result_budget:,} characters",
                f"{RESULT_BUDGET_CHARS:,} characters")
        # the signposts are longer than the room kept only for an unusual
        # reference or figure name: leave out the lowest-ranked hits
        while (len(result["result"]) > 1 and len(json.dumps(
                result, default=str)) > RESULT_BUDGET_CHARS):
            result["result"].pop()
            result["shown"] = len(result["result"])
            result["left_out"] = int(result.get("left_out") or 0) + 1
            result["next_offset"] = offset + len(result["result"])
        return result

    _search_fitted.__wrapped__ = figure_search
    registry["figure_search"] = _search_fitted
    # figure_search's catch-all reads only the result-cap names above, so
    # dispatch checks it against its parameters and these (GM-110).
    info["figure_search"].pop("open_parameters", None)
    declare_aliases(info, {"figure_search": dict.fromkeys(_LIMIT_ALIASES,
                                                          "limit")})
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
