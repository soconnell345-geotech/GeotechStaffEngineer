"""fem2d movements per construction stage through the agent adapter
(owner direction 2026-10-09): the reference-stage option reaches the
analysis, the per-stage keys are documented, and a default reference
choice that changes the numbers comes back as a ``judgment`` record."""

import numpy as np

from funhouse_agent.dispatch import call_agent, _load_adapter

JUDGMENT_KEYS = {"question", "options", "used", "why"}
OPTION_KEYS = {"name", "source", "assumptions", "applies", "result"}


def _returns(method):
    return set(_load_adapter("fem2d").METHOD_INFO[method]["returns"])


def _con1(**extra):
    params = {
        "width": 2.0, "depth": 20.0,
        "soil_layers": [{"bottom_elevation": -20.0, "E": 529412,
                         "nu": 0.3235, "gamma": 18.0}],
        "k": 1e-10, "load_q": 100.0, "time_points": [1e7, 1e8],
        "gwt": 20.0, "consolidation_scheme": "monolithic", "theta": 0.5,
        "n_w": 4e6, "nx": 3, "ny": 20}
    params.update(extra)
    return call_agent("fem2d", "fem2d_consolidation", params)


def _check_judgment(j):
    assert set(j) == JUDGMENT_KEYS
    assert j["question"] == "Which stage are movements measured from?"
    assert len(j["options"]) == 2
    for o in j["options"]:
        assert set(o) == OPTION_KEYS
        assert "mm" in o["result"]
    assert j["used"] in [o["name"] for o in j["options"]]


class TestConsolidationAdapter:
    def test_default_carries_judgment_and_stages(self):
        r = _con1()
        assert "error" not in r, r
        assert _returns("fem2d_consolidation") <= set(r)
        _check_judgment(r["judgment"])
        assert [s["name"] for s in r["stages"]] == ["initial", "load"]
        assert r["displacement_reference"]["chosen_by"] == "default"
        # Reported settlement is the load stage alone.
        assert r["surface_settlement_m_by_time"] == \
            r["stages"][1]["surface_settlement_m_by_time"]

    def test_reference_option_reaches_the_analysis(self):
        r = _con1(reset_displacements_after="start")
        assert "error" not in r, r
        assert "judgment" not in r
        assert r["displacement_reference"]["chosen_by"] == "user"
        assert r["surface_settlement_m_by_time"] == \
            r["stages"][1]["cumulative_surface_settlement_m_by_time"]


def _staged_params(**extra):
    from fem2d.mesh import generate_rect_mesh
    nodes, elements = generate_rect_mesh(0, 8, -4, 0, 8, 4)
    top = np.where(np.abs(nodes[:, 1]) < 1e-9)[0]
    top = top[np.argsort(nodes[top, 0])]
    load = [i for i in top if nodes[i, 0] <= 2.0 + 1e-9]
    edges = [[int(load[i]), int(load[i + 1])] for i in range(len(load) - 1)]
    params = {
        "nodes": nodes.tolist(), "elements": elements.tolist(),
        "material_props": [{"E": 20000, "nu": 0.3, "c": 1e6, "phi": 0,
                            "psi": 0}],
        "gamma": 18.0,
        "element_groups": {"all": list(range(len(elements)))},
        "phases": [
            {"name": "Gravity", "active_soil_groups": ["all"]},
            {"name": "Load", "active_soil_groups": ["all"],
             "surface_loads": [[edges, 0.0, -100.0]]},
        ]}
    params.update(extra)
    return params


class TestStagedAdapter:
    def test_default_carries_judgment_and_stage_movements(self):
        r = call_agent("fem2d", "fem2d_staged", _staged_params())
        assert "error" not in r, r
        assert _returns("fem2d_staged") <= set(r)
        _check_judgment(r["judgment"])
        p1 = r["phases"][1]
        for key in ("delta_displacement", "cumulative_displacement",
                    "displacement_since_reference"):
            assert set(p1[key]) >= {"settlement_m", "heave_m",
                                    "horizontal_m", "total_m"}
        assert p1["reference_stage"] == "end of stage 0 'Gravity'"
        assert p1["delta_displacement"]["settlement_m"] > \
            p1["cumulative_displacement"]["settlement_m"]   # less negative

    def test_user_reference_has_no_judgment(self):
        r = call_agent("fem2d", "fem2d_staged",
                       _staged_params(reset_displacements_after="start"))
        assert "error" not in r, r
        assert "judgment" not in r
        p1 = r["phases"][1]
        assert p1["displacement_since_reference"] == \
            p1["cumulative_displacement"]
