"""GM-110: dispatch refuses a top-level parameter the method does not take.

Many adapters never checked their top-level names, so a misspelt one
('phi_deg' for 'phi') was read by nobody and the run used the default.
``dispatch._check_parameter_names`` now checks every call against the
method's METHOD_INFO (its ``parameters``, the ``parameter_aliases`` the
adapter reads, and for ``render_figures`` the named package's parameters),
before the handler runs, with the did-you-mean and valid list of
``reject_unknown_params``. Reference wrappers (``func(**params)``) keep their
own signature check, and a method marked ``open_parameters`` checks its own.
"""

import importlib
import os

import pytest

import funhouse_agent.adapters as adapters
from funhouse_agent import dispatch as D


def _error(agent, method, params):
    out = D.call_agent(agent, method, params)
    assert isinstance(out, dict) and "error" in out, out
    return out["error"]


def _all_methods():
    for agent in sorted(D.MODULE_REGISTRY):
        mod = D._load_adapter(agent)
        for method, info in mod.METHOD_INFO.items():
            yield agent, mod, method, info


# ---------------------------------------------------------------------------
# A misspelt top-level key is refused with the did-you-mean, per adapter
# group named in GM-110 (none of these checked before)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent, method, params, unknown, suggestion", [
    ("opensees", "pm4sand_cyclic_dss",
     {"Dr": 0.5, "G0": 600, "hpo": 0.5, "Den": 1.7, "sigma_v_kpa": 100},
     "sigma_v_kpa", "sigma_v"),
    ("pystrata", "eql_site_response",
     {"layers": [], "motion": "synthetic_pulse", "max_iteration": 10},
     "max_iteration", "max_iterations"),
    ("liquepy", "cpt_liquefaction",
     {"depth": [1, 2], "q_c": [1, 2], "f_s": [1, 2], "gwl_m": 1.0},
     "gwl_m", "gwl"),
    ("seismic_signals", "response_spectrum",
     {"motion": "synthetic_pulse", "dampng": 0.05},
     "dampng", "damping"),
    ("liquefaction", "liquefaction_analysis",
     {"depth": [3], "N160": [10], "FC": [5], "gamma": [18],
      "amax_g": 0.3, "gwt_dpth": 1.0},
     "gwt_dpth", "gwt_depth"),
    ("pystra", "form_analysis",
     {"variables": {}, "limit_state": "x", "corelation": []},
     "corelation", "correlation"),
    ("calc_package", "bearing_capacity_package",
     {"width": 2, "unit_weight": 18, "frictionangle": 30},
     "frictionangle", "friction_angle"),
    ("calc_package", "wave_equation_package",
     {"hammer_name": "x", "pile_length": 10, "pile_area": 0.1,
      "pile_E": 2e8, "quake_sides": 0.003},
     "quake_sides", "quake_side"),
    ("subsurface", "plot_plan_view",
     {"site_key": "x", "label_feild": "id"},
     "label_feild", "label_field"),
    ("subsurface", "parse_cpt", {"file_path": "x.gef", "indx": 0},
     "indx", "index"),
    ("gec9", "search_sections", {"querry": "p-y"}, "querry", "query"),
])
def test_misspelt_top_level_key_refused(agent, method, params, unknown,
                                        suggestion):
    err = _error(agent, method, params)
    assert f"{method}: unknown parameter(s) ['{unknown}']" in err
    assert f"{unknown} -> {suggestion}" in err
    assert "Valid parameters:" in err


def test_refused_before_the_handler_runs(monkeypatch):
    mod = D._load_adapter("seismic_signals")
    ran = []
    monkeypatch.setitem(mod.METHOD_REGISTRY, "response_spectrum",
                        lambda p: ran.append(p) or {"ok": True})
    _error("seismic_signals", "response_spectrum", {"dampng": 0.05})
    assert ran == []
    assert D.call_agent("seismic_signals", "response_spectrum",
                        {"damping": 0.05}) == {"ok": True}


def test_nothing_is_renamed(monkeypatch):
    mod = D._load_adapter("liquefaction")
    seen = []
    monkeypatch.setitem(mod.METHOD_REGISTRY, "liquefaction_analysis",
                        lambda p: seen.append(dict(p)) or {})
    D.call_agent("liquefaction", "liquefaction_analysis",
                 {"depths": [3], "qc": [1, 2]})
    assert seen == [{"depths": [3], "qc": [1, 2]}]


# ---------------------------------------------------------------------------
# What is accepted
# ---------------------------------------------------------------------------

def test_every_declared_parameter_and_alias_is_accepted():
    """Every METHOD_INFO parameter and parameter_aliases name passes the
    check, in every module (reference wrappers included)."""
    for agent, mod, method, info in _all_methods():
        names = set(info.get("parameters") or {})
        names |= set(info.get("parameter_aliases") or {})
        params = dict.fromkeys(names)
        of = info.get("parameters_of")
        if of:
            params[of] = next(m for m in mod.METHOD_INFO
                              if m.endswith("_package")
                              and not mod.METHOD_INFO[m].get(
                                  "open_parameters"))
        D._check_parameter_names(mod, method, params)


def test_every_alias_names_a_parameter_of_its_method():
    for agent, mod, method, info in _all_methods():
        params = info.get("parameters") or {}
        for alias, canonical in (info.get("parameter_aliases") or {}).items():
            assert canonical in params, (agent, method, alias, canonical)
            assert alias not in params, (agent, method, alias)


def test_metadata_keys_are_known_shapes():
    for agent, mod, method, info in _all_methods():
        assert isinstance(info.get("parameters"), dict), (agent, method)
        of = info.get("parameters_of")
        if of:
            assert of in info["parameters"], (agent, method, of)


def test_attachment_key_always_accepted():
    mod = D._load_adapter("seismic_signals")
    D._check_parameter_names(mod, "response_spectrum",
                             {"attachment_key": "x.csv"})


def test_method_aliases_inject_declared_parameters():
    """The selector values dispatch's own method aliases inject must be
    parameters of the method they route to."""
    for (agent, guess), entry in D._METHOD_ALIASES.items():
        real, inject = (entry, {}) if isinstance(entry, str) else entry
        mod = D._load_adapter(agent)
        info = mod.METHOD_INFO[real]
        accepted = set(info["parameters"]) | set(
            info.get("parameter_aliases") or {})
        assert set(inject or {}) <= accepted, (agent, guess, real, inject)


def test_method_alias_still_resolves_and_is_checked():
    out = D.call_agent("bearing_capacity", "vesic",
                       {"width": 2, "unit_weight": 18, "friction_angle": 30})
    assert "error" not in out and out["q_ultimate_kPa"] > 0
    err = _error("bearing_capacity", "vesic",
                 {"width": 2, "unit_weight": 18, "phi_deg": 30})
    assert "unknown parameter(s) ['phi_deg']" in err


def test_adapter_aliases_still_work():
    out = D.call_agent("settlement", "elastic_settlement",
                       {"pressure": 100, "B": 2, "Es": 20000})
    assert "error" not in out, out
    out = D.call_agent("drilled_shaft", "get_resistance_factors", {})
    assert "error" not in out


def test_adapter_own_checks_agree_with_method_info(monkeypatch, tmp_path):
    """Whatever an adapter's own reject_unknown_params accepts, METHOD_INFO
    declares (as a parameter or a parameter alias); otherwise dispatch would
    refuse a name the adapter reads. Captured by calling each handler with
    no parameters and a reject_unknown_params that records and stops."""
    monkeypatch.chdir(tmp_path)

    class _Stop(Exception):
        pass

    seen = {}

    def _capture(params, valid, *, method, aliases=()):
        seen["valid"] = set(valid)
        raise _Stop

    problems, checked = [], 0
    for agent in sorted(D.MODULE_REGISTRY):
        mod = D._load_adapter(agent)
        if not hasattr(mod, "reject_unknown_params"):
            continue
        monkeypatch.setattr(mod, "reject_unknown_params", _capture)
        for method, handler in mod.METHOD_REGISTRY.items():
            info = mod.METHOD_INFO.get(method) or {}
            if info.get("open_parameters"):
                continue
            seen.clear()
            try:
                handler({})
            except _Stop:
                pass
            except Exception:           # noqa: BLE001 - not reached
                continue
            if "valid" not in seen:
                continue
            checked += 1
            accepted = set(info.get("parameters") or {}) | set(
                info.get("parameter_aliases") or {})
            extra = sorted(seen["valid"] - accepted)
            if extra:
                problems.append((agent, method, extra))
    assert checked > 100
    assert problems == []


# ---------------------------------------------------------------------------
# render_figures, open methods, references
# ---------------------------------------------------------------------------

class TestRenderFigures:
    def test_takes_the_named_packages_parameters(self):
        mod = D._load_adapter("calc_package")
        D._check_parameter_names(mod, "render_figures", {
            "package": "bearing_capacity_package", "width": 2,
            "unit_weight": 18, "output_dir": "x", "format": "html"})
        D._check_parameter_names(mod, "render_figures", {
            "method": "settlement_package", "q_applied": 100})

    def test_refuses_what_the_package_does_not_take(self):
        err = _error("calc_package", "render_figures", {
            "package": "slope_stability_package", "ru": 0.2,
            "surface_points": [[0, 0]]})
        assert "render_figures: unknown parameter(s) ['ru']" in err

    def test_no_package_is_left_to_the_handler(self):
        err = _error("calc_package", "render_figures", {"width": 2})
        assert "Provide 'package'" in err


def test_pavement_package_checks_its_tools_names():
    # The package takes the parameters of the pavement_design tools
    # (GM-141): dispatch checks a name against them all, the run against
    # the tool design_type names.
    mod = D._load_adapter("calc_package")
    assert not mod.METHOD_INFO["pavement_design_package"].get(
        "open_parameters")
    D._check_parameter_names(mod, "pavement_design_package",
                             {"w18": 1e6, "esals": 1, "sc_psi": 650})
    err = _error("calc_package", "pavement_design_package",
                 {"w18": 1e6, "layers": [], "w_19": 1})
    assert "unknown parameter(s) ['w_19']" in err
    err = _error("calc_package", "pavement_design_package",
                 {"w18": 1e6, "layers": [], "sc_psi": 650})
    assert "unknown parameter(s) ['sc_psi']" in err
    assert "design_type='rigid' runs rigid_pavement_design" in err


def test_reference_wrapper_keeps_its_signature_check():
    mod = D._load_adapter("gec9")
    method = "table_4_1_lateral_resistance_factor"
    assert getattr(mod.METHOD_REGISTRY[method], "__wrapped__", None)
    out = D.call_agent("gec9", method, {"conditon": "x"})
    assert "'conditon' is not a parameter" in out["error"]
    assert "required_parameters" in out


class TestReferenceCatchAll:
    """A reference function with a **kwargs catch-all accepted any name:
    DM7's plot_figure_5_16(tv=0.5) drew no query point and said nothing.
    One whose catch-all is never read is checked against its named
    parameters; one whose catch-all is read is open until its adapter
    declares what it reads."""

    @pytest.mark.parametrize("method, given, meant", [
        ("plot_figure_5_16", "tv", "Tv"),
        ("plot_figure_5_6_mu0", "d_over_b", "D_over_B"),
        ("plot_figure_8_46", "pi", "PI"),
        ("plot_figure_5_18a", "h_over_b", "H_over_B"),
    ])
    def test_dm7_plot_catch_all_refuses_a_misspelt_name(self, method, given,
                                                        meant):
        err = _error("dm7", method, {given: 0.5})
        assert f"{method}: unknown parameter(s) ['{given}']" in err
        assert f"{given} -> {meant}" in err

    def test_dm7_plot_named_parameters_pass(self):
        mod = D._load_adapter("dm7")
        D._check_parameter_names(mod, "plot_figure_5_16",
                                 {"Tv": 0.5, "show": False})

    def test_figure_search_reads_only_its_cap_aliases(self):
        mod = D._load_adapter("figure_db")
        info = mod.METHOD_INFO["figure_search"]
        assert not info.get("open_parameters")
        D._check_parameter_names(mod, "figure_search",
                                 {"query": "x", "top_k": 3})
        err = _error("figure_db", "figure_search",
                     {"query": "x", "refrence": "dm7_2"})
        assert "refrence -> reference" in err

    def test_catch_all_is_read(self):
        from funhouse_agent.adapters._reference_common import (
            catch_all_is_read, extract_method_info,
        )

        def unused(a, **kwargs):
            """Doc."""
            return a

        def used(a, **kw):
            return kw.get("b", a)

        def none(a):
            return a

        assert catch_all_is_read(unused) is False
        assert catch_all_is_read(used) is True
        assert catch_all_is_read(none) is False
        assert "open_parameters" not in extract_method_info(unused, "c", "r")
        assert extract_method_info(used, "c", "r")["open_parameters"] is True

    def test_every_read_catch_all_is_open_or_declared(self):
        """No reference wrapper whose catch-all is read is checked with
        nothing declared for what it reads."""
        from funhouse_agent.adapters._reference_common import (
            catch_all_is_read,
        )
        found = 0
        for agent, mod, method, info in _all_methods():
            fn = getattr(mod.METHOD_REGISTRY[method], "__wrapped__", None)
            if fn is None or not D._takes_catch_all(fn):
                continue
            found += 1
            if catch_all_is_read(fn):
                assert (info.get("open_parameters")
                        or info.get("parameter_aliases")), (agent, method)
        assert found >= 8


# ---------------------------------------------------------------------------
# Names a method deliberately does not take: refused with the reason
# (refused_parameters), never with a near-miss guess
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent, method, params, refused, reason", [
    ("pile_group", "pile_group_simple",
     {"n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000, "Vx": 50},
     "Vx", "need pile_group_6dof"),
    ("pile_group", "pile_group_simple",
     {"n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000,
      "axial_stiffness": 1e5}, "axial_stiffness", "pile_group_6dof"),
    ("retaining_walls", "mse_lrfd_external_stability",
     {"wall_height": 6, "gamma_backfill": 19, "phi_backfill": 34,
      "reinforcement_spacing": 0.6}, "reinforcement_spacing",
     "external stability does not depend on it"),
    ("soe", "cantilever_excavation",
     {"excavation_depth": 3, "layers": [], "excavation_width": 10},
     "excavation_width", "check_basal_heave"),
    ("soe", "cantilever_excavation",
     {"excavation_depth": 3, "layers": [], "supports": []},
     "supports", "braced_excavation"),
    ("soe", "braced_excavation",
     {"excavation_depth": 3, "layers": [], "excavation_width": 10},
     "excavation_width", "does not use the excavation width"),
])
def test_refused_with_its_reason(agent, method, params, refused, reason):
    err = _error(agent, method, params)
    assert f"{method} does not take ['{refused}']: {refused}:" in err
    assert reason in err
    assert "Did you mean" not in err          # no guess at another quantity
    info = D._load_adapter(agent).METHOD_INFO[method]
    assert refused not in info["parameters"]  # nor advertised
    if agent != "calc_package":
        # the adapter refuses it too, for a caller that skips dispatch
        mod = D._load_adapter(agent)
        with pytest.raises(ValueError, match=rf"\['{refused}'\]"):
            mod.METHOD_REGISTRY[method](dict(params))


def test_refused_and_misspelt_reported_together():
    err = _error("pile_group", "pile_group_simple",
                 {"n_rows": 2, "n_cols": 2, "spacng": 1.5, "Vy": 5})
    assert "does not take ['Vy']" in err
    assert "Also:" in err and "spacng -> spacing" in err


def test_refused_names_are_never_declared():
    for agent, mod, method, info in _all_methods():
        for name, spec in (info.get("refused_parameters") or {}).items():
            assert name not in (info.get("parameters") or {}), (agent, method)
            assert name not in (info.get("parameter_aliases") or {})
            assert adapters.refused_reason(spec), (agent, method, name)


class TestUnusedValuesPass:
    """A refused name's "not used" value passes silently: a lateral load of
    0 is none. Any other value is refused with the reason; the stiffnesses
    take none. (The MSE backfill slope was refused this way until GM-112
    built the sloping backfill; a slope of 0 is still the level result.)"""

    _MSE = {"wall_height": 6, "gamma_backfill": 19, "phi_backfill": 34}
    _GROUP = {"n_rows": 2, "n_cols": 2, "spacing": 1.5, "Vz": 1000}

    @pytest.mark.parametrize("method, extra", [
        ("mse_wall", {"reinforcement_name": "geogrid_ux1600"}),
        ("mse_lrfd_external_stability", {})])
    def test_level_backfill_slope_runs(self, method, extra):
        given = {**self._MSE, **extra}
        level = D.call_agent("retaining_walls", method, given)
        zero = D.call_agent("retaining_walls", method,
                            {**given, "backfill_slope": 0.0})
        assert "error" not in zero, zero
        # The same result; only defaults_used (GM-115) says that one call
        # left backfill_slope out.
        level_used, zero_used = level.pop("defaults_used"), \
            zero.pop("defaults_used")
        assert level_used["backfill_slope"] == 0.0
        assert "backfill_slope" not in zero_used
        assert {k: v for k, v in level_used.items()
                if k != "backfill_slope"} == zero_used
        assert zero == level
        sloped = D.call_agent("retaining_walls", method,
                              {**given, "backfill_slope": 15})
        assert "error" not in sloped, sloped
        assert sloped["backslope"]["backfill_slope_deg"] == 15
        assert "backslope" not in level

    def test_zero_lateral_load_runs_on_the_simple_method(self):
        base = D.call_agent("pile_group", "pile_group_simple", self._GROUP)
        zero = D.call_agent("pile_group", "pile_group_simple",
                            {**self._GROUP, "Vx": 0, "Vy": 0.0, "Mz": 0})
        assert "error" not in zero, zero
        assert zero == base
        for key in ("Vx", "Vy", "Mz"):
            err = _error("pile_group", "pile_group_simple",
                         {**self._GROUP, key: -10})
            assert f"does not take ['{key}']" in err

    def test_zero_stiffness_still_refused(self):
        err = _error("pile_group", "pile_group_simple",
                     {**self._GROUP, "axial_stiffness": 0})
        assert "does not take ['axial_stiffness']" in err

    def test_accepted_as_unused(self):
        spec = {"reason": "r", "accepts": [0]}
        assert adapters.accepted_as_unused(spec, 0)
        assert adapters.accepted_as_unused(spec, 0.0)
        assert not adapters.accepted_as_unused(spec, 1e-9)
        assert not adapters.accepted_as_unused(spec, False)
        assert not adapters.accepted_as_unused(spec, [0])
        assert not adapters.accepted_as_unused("reason only", 0)


def test_mse_methods_declare_the_slope_they_compute():
    """GM-112: both MSE methods take a sloping backfill and say how."""
    from funhouse_agent.adapters.retaining_walls import METHOD_INFO
    for method in ("mse_wall", "mse_lrfd_external_stability"):
        info = METHOD_INFO[method]
        spec = info["parameters"]["backfill_slope"]
        assert "degrees from horizontal" in spec["description"]
        assert "GEC-11" in spec["description"]
        assert "broken_backslope_height" in info["parameters"]
        assert "backfill_slope" not in (info.get("refused_parameters") or {})


def test_braced_excavation_no_longer_advertises_a_width():
    from funhouse_agent.adapters.soe import METHOD_INFO
    assert "excavation_width" not in METHOD_INFO["braced_excavation"][
        "parameters"]


class TestPileGroupPackageLateral:
    _BASE = {"layout": "rectangular", "n_rows": 2, "n_cols": 2,
             "spacing_x": 1.5, "spacing_y": 1.5, "Vz": 2000.0}

    def test_lateral_load_without_6dof_refused_with_reason(self, tmp_path):
        from funhouse_agent.adapters.calc_package import (
            _generate_pile_group_package)
        for key in ("Vx", "Vy", "Mz"):
            with pytest.raises(ValueError, match=rf"\['{key}'\].*"
                               r"method_6dof=True runs pile_group_6dof"):
                _generate_pile_group_package({
                    **self._BASE, key: 50.0,
                    "output_path": str(tmp_path / "pg.html")})

    def test_zero_lateral_load_runs(self, tmp_path):
        from funhouse_agent.adapters.calc_package import (
            _generate_pile_group_package)
        res = _generate_pile_group_package({
            **self._BASE, "Vx": 0.0, "Mz": 0.0,
            "output_path": str(tmp_path / "pg.html")})
        assert res["status"] == "success"


def test_render_figures_lists_no_placeholder_name():
    mod = D._load_adapter("calc_package")
    assert "..." not in mod.METHOD_INFO["render_figures"]["parameters"]
    err = _error("calc_package", "render_figures", {
        "package": "settlement_package", "q_aplied": 100})
    assert "'...'" not in err and "q_aplied -> q_applied" in err


def test_declare_aliases_keeps_only_aliases_of_parameters():
    info = {"m": {"parameters": {"a": {}, "b": {}}}}
    adapters.declare_aliases(info, {"m": {"x": "a", "y": "nope", "b": "a",
                                          "a": "a"}})
    assert info["m"]["parameter_aliases"] == {"x": "a"}
