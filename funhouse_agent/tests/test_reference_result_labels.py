"""A reference lookup's number says what it is (2026-11-10 review, item 1).

``describe_method`` for a reference method carries the docstring's Returns
text (where the unit is), its Raises (where a chart's printed limits are),
Notes and References; a bare number, tuple or list comes back with the
method's name, what its Returns text says it is and its reference beside
``result``; an unannotated parameter takes its type from its docstring
line, else "any" (it was advertised as "float"). The functions here are
made up, with made-up values.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, extract_method_info, make_wrapper,
)


def made_up_chart_value(phi: float) -> float:
    """Made-up limiting resistance qX for a test (Figure 9-9).

    Parameters
    ----------
    phi : float
        Friction angle (degrees), 20 to 30 -- the span of the made-up
        curve.

    Returns
    -------
    float
        Made-up limiting resistance qX in tsf.  Never above 50 tsf.

    Raises
    ------
    ValueError
        If phi is outside 20-30 deg: the made-up chart stops there.

    Notes
    -----
    A made-up note on reading the curve.
    """
    if not 20 <= phi <= 30:
        raise ValueError("outside the made-up chart")
    return 1.5 * phi


def made_up_factors(a, kind, flag=False):
    """Made-up shape factors for a test.

    Parameters
    ----------
    a : float
        A made-up ratio.
    kind : str
        'one' or 'two'.
    flag
        A parameter with neither annotation nor docstring type.

    Returns
    -------
    tuple of float
        (sx, sy) made-up shape factors.

    References
    ----------
    Made-Up Manual 1, Chapter 2, Table 2-3, p. 45.
    """
    return (1.0 + a, 2.0 + a)


def made_up_named(a: float):
    """Made-up pair returned as named entries.

    Returns
    -------
    nx : float
        Made-up first factor.
    ny : float
        Made-up second factor.
    """
    return (a, 2 * a)


def made_up_type_only(a: float) -> float:
    """Made-up ratio with a type-only Returns (Table 7-7).

    Returns
    -------
    float
    """
    return a / 2.0


def made_up_table(key: str = "") -> dict:
    """Made-up table returned as a dict.

    Returns
    -------
    dict
        Made-up rows.
    """
    return {"key": key, "value_kpa": 12.5}


class TestDescribeCarriesTheDocstring:
    def test_returns_raises_and_notes(self):
        info = extract_method_info(made_up_chart_value, "Figures", "MADE-UP-1")
        assert info["returns"].startswith("float: Made-up limiting "
                                          "resistance qX in tsf.")
        assert "outside 20-30 deg" in info["raises"]
        assert info["raises"].startswith("ValueError: ")
        assert info["notes"] == "A made-up note on reading the curve."

    def test_references_section(self):
        info = extract_method_info(made_up_factors, "Tables", "MADE-UP-1")
        assert info["references"] == ("Made-Up Manual 1, Chapter 2, "
                                      "Table 2-3, p. 45.")
        assert info["returns"] == ("tuple of float: (sx, sy) made-up shape "
                                   "factors.")

    def test_long_sections_are_trimmed(self):
        def long_notes(a: float) -> float:
            """Brief.

            Returns
            -------
            float
                Value.

            Notes
            -----
            """
            return a
        long_notes.__doc__ += "    " + "word " * 400
        info = extract_method_info(long_notes, "c", "r")
        assert len(info["notes"]) <= 400
        assert info["notes"].endswith("...")


class TestParameterTypes:
    def test_unannotated_parameter_types(self):
        params = extract_method_info(made_up_factors, "c", "r")["parameters"]
        assert params["a"]["type"] == "float"      # docstring line
        assert params["kind"]["type"] == "str"     # docstring line
        # neither: its default says (review F9), never a guessed "float"
        assert params["flag"]["type"] == "bool"
        assert params["flag"]["description"].startswith("A parameter with")

    def test_a_default_gives_the_type_where_nothing_else_does(self):
        def defaults(es=29000.0, n=3, name=None, given=None):
            """Brief.

            Parameters
            ----------
            es
                A made-up modulus.
            n
                A made-up count.
            name
                A made-up name.
            given : str
                Typed by its line.
            """
            return es
        params = extract_method_info(defaults, "c", "r")["parameters"]
        assert params["es"]["type"] == "float"
        assert params["n"]["type"] == "number"
        assert params["name"]["type"] == "any"      # None says nothing
        assert params["given"]["type"] == "str"

    def test_annotation_wins(self):
        params = extract_method_info(made_up_chart_value, "c",
                                     "r")["parameters"]
        assert params["phi"]["type"] == "float"
        assert "20 to 30" in params["phi"]["description"]

    def test_names_sharing_a_line_share_its_description(self):
        def shared(b1, b2):
            """Brief.

            Parameters
            ----------
            b1, b2 : float
                Two made-up widths (m).
            """
            return b1 + b2
        params = extract_method_info(shared, "c", "r")["parameters"]
        assert params["b1"]["description"] == "Two made-up widths (m)."
        assert params["b2"]["type"] == "float"

    def test_indented_lines_under_a_parameter_stay_its_description(self):
        def nested(rows):
            """Brief.

            Parameters
            ----------
            rows : list of dict
                Each with:
                'x_m' : float -- a made-up depth.
            """
            return rows
        desc = extract_method_info(nested, "c",
                                   "r")["parameters"]["rows"]["description"]
        assert desc == "Each with: 'x_m' : float -- a made-up depth."


class TestBareResultsAreLabelled:
    def test_bare_number(self):
        out = make_wrapper(made_up_chart_value, "MADE-UP-1")({"phi": 24})
        assert out["result"] == 36.0
        assert out["method"] == "made_up_chart_value"
        # the first sentence of the Returns text, where the unit is
        assert out["returns"] == "Made-up limiting resistance qX in tsf."
        # no References section: the adapter's reference and the figure
        # the brief names
        assert out["reference"] == "MADE-UP-1, Figure 9-9"

    def test_tuple_names_its_elements(self):
        out = make_wrapper(made_up_factors, "MADE-UP-1")({"a": 0.5,
                                                          "kind": "one"})
        assert out["result"] == [1.5, 2.5]
        assert out["elements"] == ["sx", "sy"]
        assert out["reference"].startswith("Made-Up Manual 1")

    def test_named_return_entries_are_elements(self):
        out = make_wrapper(made_up_named, "R")({"a": 2.0})
        assert out["elements"] == ["nx", "ny"]
        assert out["result"] == [2.0, 4.0]

    def test_elements_only_when_the_count_matches(self):
        def three():
            """Brief.

            Returns
            -------
            list of float
                (p, q) made-up values.
            """
            return [1.0, 2.0, 3.0]
        out = make_wrapper(three, "R")({})
        assert "elements" not in out
        assert out["result"] == [1.0, 2.0, 3.0]

    def test_a_bracketed_remark_before_a_number_is_not_elements(self):
        """Review F9: "(US customary, as printed)" opening a float's
        description was read as two element names; only a tuple's or a
        list's leading bracket names elements (test_tuple_names_its_elements)."""
        def one_value():
            """Brief.

            Returns
            -------
            float
                (US customary, as printed) a made-up value in ksf.
            """
            return 1.0
        from funhouse_agent.adapters._reference_common import returns_parts
        assert returns_parts(one_value)["elements"] == []

    def test_type_only_returns_falls_back_to_the_brief(self):
        out = make_wrapper(made_up_type_only, "MADE-UP-2")({"a": 3.0})
        assert out["returns"].startswith("Made-up ratio with a type-only")
        assert out["reference"] == "MADE-UP-2, Table 7-7"

    def test_a_dict_result_is_unchanged(self):
        out = make_wrapper(made_up_table, "R")({"key": "k"})
        assert out == {"key": "k", "value_kpa": 12.5}

    def test_label_is_small(self):
        out = make_wrapper(made_up_chart_value, "MADE-UP-1")({"phi": 24})
        added = {k: v for k, v in out.items() if k != "result"}
        import json
        assert len(json.dumps(added)) < 400

    def test_registry_built_from_a_module_labels_and_describes(self):
        import types
        mod = types.ModuleType("made_up_refs")
        mod.made_up_chart_value = made_up_chart_value
        registry, info = build_lookup_registry([(mod, "Figures",
                                                 "MADE-UP-1")])
        out = registry["made_up_chart_value"]({"phi": 20})
        assert out["returns"].endswith("in tsf.")
        assert out["reference"] == "MADE-UP-1, Figure 9-9"
        assert "tsf" in info["made_up_chart_value"]["returns"]


class TestThroughDispatch:
    """A real reference module, through call_agent: a bare number comes
    back labelled, and describe_method carries the Returns text."""

    def _first_float_method(self):
        import inspect
        from funhouse_agent.dispatch import _load_adapter
        mod = _load_adapter("gec12")
        for name, fn in mod.METHOD_REGISTRY.items():
            func = getattr(fn, "__wrapped__", None)
            if func is None or mod.METHOD_INFO[name].get("alias_of"):
                continue
            sig = inspect.signature(func)
            if sig.return_annotation is float and \
                    "returns" in mod.METHOD_INFO[name]:
                return name, mod.METHOD_INFO[name]
        raise AssertionError("no float-returning gec12 method")

    def test_describe_method_has_the_returns_text(self):
        from funhouse_agent.dispatch import describe_method
        name, info = self._first_float_method()
        desc = describe_method("gec12", name)
        assert desc["returns"].startswith("float: ")
        assert len(desc["returns"]) > len("float: ")
