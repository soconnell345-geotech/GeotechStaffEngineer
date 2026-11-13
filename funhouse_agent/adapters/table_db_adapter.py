"""Table-catalog retrieval adapter: SQLite FTS5 over every reference's
printed-table catalog (``<reference>/tables_catalog.json``; the library's
``geotech_references._tables_db``, REF-86).

Companion to ``figure_db_adapter`` for printed TABLES. Each catalog row says
where a table's values are (``values``): ``"text"`` (typed in a stored
section, ``values_in``), ``"lookup"`` (a coded function returns them) or
``"page"`` (only on the printed page). Every hit and row carries
``how_to_read``, the next step in this app's own tool names:

- ``"text"``: ``reference_db.reference_get`` on the section named;
- ``"lookup"``: ``call_agent`` on the reference's own module, naming
  every method that codes the table (``values_in`` and ``implemented_in``,
  one for each part of a table coded by several);
- ``"page"``: the ``read_reference_table`` call that reads the page with
  vision, where this host can reach the reference PDF (the switch that
  offers ``read_reference_figure``, ``reference_docs.table_readable``);
  where it cannot, the plain statement that the values are only on that
  printed page and cannot be read here, naming no reading tool.

- ``table_search(query, reference="", chapter=0, limit=5)``: ranked table
  hits (number, caption, page, where the values are).
- ``table_get(reference, table_number, pdf_page_index=None)``: the whole
  catalog row.
- ``list_indexed_tables()``: the references with a table catalog and how
  many of their tables are text, lookup and page.

A library with no table catalogs (an older pin) gives an adapter with no
methods, never an import error.
"""

from __future__ import annotations

import json

from funhouse_agent.adapters import declare_aliases
from funhouse_agent.adapters._reference_common import (
    RESULT_BUDGET_CHARS, as_result, extract_method_info, make_wrapper,
)

#: Names models guess for table_search's result cap; its catch-all reads
#: these as ``limit`` and nothing else.
_LIMIT_ALIASES = ("top_k", "k", "n", "max_results", "top", "num_results")

#: Room kept for each hit's ``how_to_read`` when the library fits the hits
#: to a size (the longest next step is about 470 characters: a table coded
#: by nine methods), and the least the library is given to fit them in.
_SIGNPOST_ROOM = 480
_LEAST_LIBRARY_BUDGET = 4000

#: Room ``table_get`` keeps beside the library's fitted row for the app's
#: own ``how_to_read`` (the library's sentence is replaced).
_GET_ROOM = 600


def _chapter_scope(chapter):
    """A table search's chapter as the catalogs file it: ``None`` for no
    filter (0 or empty), a number, or a label such as ``"II-2"``."""
    if chapter in (None, "", 0, "0"):
        return None
    text = str(chapter).strip()
    return int(text) if text.isdigit() else text


def lookup_target(reference: str, values_in: str):
    """``(module, method)`` through which this app runs the coded lookup
    ``values_in`` (``"tables.table_5_3_alpha_bond"``) of ``reference``:
    the reference's own module (``dispatch.reference_module``) and the
    function's name, where that module offers a method by that name; else
    ``(None, None)``."""
    from funhouse_agent import dispatch
    method = str(values_in or "").rsplit(".", 1)[-1].strip()
    module = dispatch.reference_module(reference)
    if not method or module is None:
        return None, None
    try:
        registry = dispatch._load_adapter(module).METHOD_REGISTRY
    except Exception:  # noqa: BLE001 - a module that will not load runs nothing
        return None, None
    return (module, method) if method in registry else (None, None)


def _takes_python_functions(reference: str, path: str) -> bool:
    """Whether the library function ``path`` ("module.function") of
    ``reference`` takes Python functions as inputs (the library marks it
    ``needs_python_callables``): no JSON call can pass them."""
    import importlib
    mod, _, name = f"geotech_references.{reference}.{path}".rpartition(".")
    try:
        fn = getattr(importlib.import_module(mod), name)
    except (ImportError, AttributeError, ValueError):
        return False
    return bool(getattr(fn, "needs_python_callables", False))


def lookup_methods(row: dict):
    """``(module, [method, ...])``: every function a lookup row lists
    (``values_in``, then ``implemented_in``) that the reference's own
    module offers by its name, leaving out one that takes Python
    functions; ``(None, [])`` when there is none. A table coded by
    several functions (Nc, Nq and N-gamma, or one case each) has each
    named, so a reader asking for one value is sent to its function."""
    ref = row.get("reference") or ""
    paths = [row.get("values_in") or ""] + list(row.get("implemented_in")
                                                 or [])
    module, methods = None, []
    for path in dict.fromkeys(p.strip() for p in paths if p and p.strip()):
        mod, method = lookup_target(ref, path)
        if mod is None or method in methods \
                or _takes_python_functions(ref, path):
            continue
        module = module or mod
        if mod == module:
            methods.append(method)
    return module, methods


def how_to_read(row: dict, chart_reads=None) -> str:
    """The next step for one catalog row or search hit, in this app's tool
    names (module docstring). ``chart_reads``: whether
    ``read_reference_figure`` / ``read_reference_table`` are offered on
    this host, when the caller already knows (None: looked up)."""
    from funhouse_agent import reference_docs
    ref = row.get("reference") or ""
    number = row.get("table_number") or ""
    values = row.get("values")
    values_in = row.get("values_in") or ""
    if values == "text" and values_in:
        return (f"Its cells are typed in section {values_in} of {ref}: "
                "call_agent('reference_db', 'reference_get', "
                f"{{'reference': '{ref}', 'section_id': '{values_in}'}}).")
    if values == "lookup" and values_in:
        module, methods = lookup_methods(row)
        if len(methods) == 1:
            method = methods[0]
            return (f"A coded lookup returns its values: call_agent("
                    f"'{module}', '{method}', {{...}}); "
                    f"describe_method('{module}', '{method}') gives its "
                    "parameters.")
        if methods:
            return (f"Coded lookups return its values, {len(methods)} "
                    f"methods of '{module}', each for part of the table: "
                    + ", ".join(methods) + f". Call call_agent('{module}', "
                    "<method>, {...}) with the one for the value needed; "
                    f"describe_method('{module}', <method>) says which "
                    "values it returns and its parameters.")
    page = row.get("pdf_page_index")
    printed = row.get("printed_page")
    if not reference_docs.table_readable(ref, number, page, chart_reads):
        return reference_docs.no_table_read(number, printed, ref)
    where = (f"printed page {printed} (pdf_page_index {page})"
             if printed not in (None, "") else f"pdf_page_index {page}")
    pages = [p for p in (row.get("pages") or []) if p is not None]
    runs = ""
    if len(pages) > 1:
        from funhouse_agent.reference_vision import TABLE_PAGES_MAX
        runs = (f" It runs over pdf_page_index "
                f"{', '.join(str(p) for p in pages)}; one read sends up to "
                f"{TABLE_PAGES_MAX} of them from the page given and says "
                "which were left for a second read.")
    return (f"Its values are printed only on {where} of {ref}: "
            "read them with read_reference_table("
            f"reference='{ref}', table_number='{number}', "
            f"pdf_page_index={page}, prompt='<value(s) needed>')." + runs
            + " Do NOT give a value of this table from its caption or from "
            "memory.")


def _build():
    try:
        from geotech_references import _tables_db
    except ImportError:
        # A library before REF-86 has no table catalogs: offer nothing.
        return {}, {}

    def _host():
        """Whether the page readers are offered on this host, looked up
        once per call and only when a page step needs it."""
        seen = {}

        def chart_reads():
            if "v" not in seen:
                from funhouse_agent import reference_docs
                seen["v"] = reference_docs.chart_reads_available()
            return seen["v"]
        return chart_reads

    def _signpost(hit: dict, chart_reads) -> None:
        """Every real hit gets ``how_to_read`` (module docstring); an
        error is left as it is."""
        if not (isinstance(hit, dict) and hit.get("table_number")
                and "error" not in hit):
            return
        needs_host = hit.get("values") not in ("text", "lookup") or \
            not hit.get("values_in")
        row = hit
        if hit.get("values") == "lookup" and "implemented_in" not in hit:
            # a hit carries values_in only; the row lists every function
            # that codes the table
            try:
                full = _tables_db.table_get(
                    hit["reference"], hit["table_number"],
                    pdf_page_index=hit.get("pdf_page_index"))
                row = dict(hit, implemented_in=full.get("implemented_in")
                           or [])
            except (KeyError, ValueError, TypeError):
                pass
        hit["how_to_read"] = how_to_read(
            row, chart_reads() if needs_host else None)

    def table_search(query: str, reference: str = "", chapter: int = 0,
                     limit: int = 5, offset: int = 0, **kwargs) -> list:
        """Full-text search of the printed-table catalogs by table number,
        caption and the title of the section that lists the table.

        The tables holding every word come first; the places left are
        filled by tables holding some of the words, each marked ``match:
        "any word"`` with ``lacks`` (the words it does not hold): read such
        a hit as no match for the whole query.

        Parameters
        ----------
        query : str
            Words to find (porter stemming; ``"phrase"`` and ``OR`` work),
            or a table number (``"6-11"``).
        reference : str, optional
            Reference id to scope to (e.g. ``"dm7_2"``), another spelling
            of it or a report number its catalog carries. ``"dm7"`` is two
            catalogs: give ``"dm7_1"`` or ``"dm7_2"``. A name with no
            catalog is an error naming the catalog ids. Empty = all.
        chapter : int or str, optional
            Chapter to scope to, as the catalog files it (prologue tables
            are chapter 0; appendix tables have none). ``0`` means no
            chapter filter.
        limit : int, optional
            Max hits (default 5, capped at 50; fewer when they would not
            fit one result, and ``next_offset`` and a ``note`` say how to
            get the rest).
        offset : int, optional
            Skip this many of the best hits: the ``next_offset`` of the
            last call (default 0).

        Returns
        -------
        list
            Ranked hits, each with ``reference``, ``reference_title``,
            ``table_number``, ``caption``, ``chapter``, ``section_id``,
            ``pdf_page_index``, ``printed_page``, ``values`` (where its
            values are: ``"text"``, ``"lookup"`` or ``"page"``),
            ``values_in``, ``reference_id``, ``pages`` for a table printed
            over several pages, and ``how_to_read``: the exact next call
            that gets its values here (``reference_db.reference_get``, the
            reference module's lookup, or ``read_reference_table`` for a
            page this host can read), or, for a page it cannot read, that
            the values are only on that printed page and not available
            here. A table's values are never inferred from its caption or
            from memory. As the agent gets it the hits are under
            ``result``, with ``matched`` and ``words_not_found`` when some
            do not hold every word and an empty result's ``note`` naming
            the scope searched.
        """
        for alt in _LIMIT_ALIASES:
            if kwargs.get(alt):
                try:
                    limit = int(kwargs[alt])
                except (TypeError, ValueError):
                    pass
        hits = _tables_db.table_search(
            query, reference=reference or None,
            chapter=_chapter_scope(chapter), limit=int(limit),
            offset=int(offset or 0))
        chart_reads = _host()
        for hit in hits:
            _signpost(hit, chart_reads)
        return hits

    def table_get(reference: str, table_number: str,
                  pdf_page_index: int = None) -> dict:
        """Fetch the whole catalog row of one printed table.

        Parameters
        ----------
        reference : str
            Reference id (e.g. ``"dm7_2"``); other spellings resolve
            (``"gec10"``, ``"dm7"`` for whichever of dm7_1 / dm7_2 holds
            the table, a document's designation).
        table_number : str
            Table number as printed (``"6-11"``, ``"A.7"``, ``"C-2a"``); a
            leading ``"Table "`` is tolerated, and '.' and '-' are read
            alike when the number as given is not filed.
        pdf_page_index : int, optional
            Which table, for a number the catalog holds more than once:
            any page of the one wanted (a search hit carries it). Without
            it such a number is an error naming each row's page.

        Returns
        -------
        dict
            The row: ``table_number``, ``caption``, ``chapter``,
            ``section_id`` and ``section_title``, ``pdf_page_index`` (the
            page where it starts), ``pages`` (every page it is printed
            on), ``printed_page``, ``values`` and ``values_in``,
            ``implemented_in`` (every coded lookup of it), ``note`` (what
            a reader of the page should know, e.g. a section holding part
            of its cells) and ``how_to_read``: the exact next call that
            gets its values here, or, for a page this host cannot read,
            that the values are only on that printed page and not
            available here.
        """
        kwargs = {"max_chars": RESULT_BUDGET_CHARS - _GET_ROOM}
        if pdf_page_index not in (None, ""):
            kwargs["pdf_page_index"] = int(pdf_page_index)
        row = _tables_db.table_get(reference, table_number, **kwargs)
        row["how_to_read"] = how_to_read(row)
        return row

    def list_indexed_tables() -> list:
        """Inventory of references that have an indexed table catalog.

        Returns
        -------
        list
            One dict per reference with ``reference``,
            ``reference_title``, ``n_tables``, ``n_chapters`` and how many
            of its tables have their values typed in a section
            (``n_text``), returned by a coded lookup (``n_lookup``) or
            printed only on the page (``n_page``).
        """
        return _tables_db.list_indexed_tables()

    funcs = {
        "table_search": table_search,
        "table_get": table_get,
        "list_indexed_tables": list_indexed_tables,
    }
    registry = {}
    info = {}
    for name, fn in funcs.items():
        registry[name] = make_wrapper(fn)
        info[name] = extract_method_info(
            fn, "Table DB (FTS5)", "geotech_references table catalogs")

    def _search_fitted(params):
        """table_search as the agent gets it: the library's own fitted
        search (how each hit matched, the scope an empty result searched)
        with room kept for the next steps; its refusals (a name with no
        catalog, "dm7", a chapter with no tables) come back as errors
        with their message, never as "nothing found"."""
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
        budget = max(RESULT_BUDGET_CHARS - _SIGNPOST_ROOM * limit,
                     _LEAST_LIBRARY_BUDGET)
        chart_reads = _host()
        for attempt in range(2):
            out = _tables_db.table_search(
                params.get("query", ""),
                reference=params.get("reference") or None,
                chapter=_chapter_scope(params.get("chapter")), limit=limit,
                offset=offset, max_chars=budget)
            if "error" in out:
                return {"result": [out]}
            library_size = len(json.dumps(out, default=str))
            for hit in out.get("hits") or []:
                _signpost(hit, chart_reads)
            result = as_result(out, "hits")
            size = len(json.dumps(result, default=str))
            shown = len(result["result"])
            if attempt or not (result.get("left_out") and shown):
                break
            # hits were left out for size with room to spare (the room
            # kept for next steps was more than they took): give the
            # library room for as many more hits as fit with theirs
            per_hit = size / shown
            more = int((RESULT_BUDGET_CHARS - size - 300) // per_hit)
            if more < 1:
                break
            budget += int(more * library_size / shown)
        if isinstance(result.get("note"), str):
            # the hits the note says would not fit would not fit the
            # whole result, next steps and all
            result["note"] = result["note"].replace(
                f"{budget:,} characters", f"{RESULT_BUDGET_CHARS:,} characters")
        popped = 0
        while (len(result["result"]) > 1 and len(json.dumps(
                result, default=str)) > RESULT_BUDGET_CHARS):
            result["result"].pop()
            popped += 1
            result["shown"] = len(result["result"])
            result["left_out"] = int(result.get("left_out") or 0) + 1
            result["next_offset"] = offset + len(result["result"])
        if popped:
            last = offset + len(result["result"])
            result["note"] = (
                f"Showing hits {offset + 1}-{last}, best first (the rest "
                f"would not fit in {RESULT_BUDGET_CHARS:,} characters); more "
                f"match. Call again with offset={last} for the next ones, or "
                "narrow the query (more specific words, or a 'reference').")
        return result

    _search_fitted.__wrapped__ = table_search
    registry["table_search"] = _search_fitted
    # table_search's catch-all reads only the result-cap names above, so
    # dispatch checks it against its parameters and these (GM-110).
    info["table_search"].pop("open_parameters", None)
    declare_aliases(info, {"table_search": dict.fromkeys(_LIMIT_ALIASES,
                                                         "limit")})
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
