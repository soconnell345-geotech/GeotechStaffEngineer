"""General 2D/3D frame analysis via PyNiteFEA.

Units at this interface: m, kN, kN/m, kPa (E, G). PyNite is unit-agnostic;
consistent kN/m/kPa inputs yield kN, kN*m, m outputs directly.

2D models (all node z = 0): out-of-plane DOFs (DZ, RX, RY) are restrained
automatically at every node so plane frames are stable without the user
specifying 3D supports (``auto_stabilize_2d=False`` to disable).
"""

import math

from pynite_agent.pynite_utils import import_pynite, SUPPORT_PRESETS
from pynite_agent.results import FrameResult, MemberResult

_COMBO = "Combo 1"          # PyNite's default load combination


def _num(v, name):
    if not (isinstance(v, (int, float)) and math.isfinite(v)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    return float(v)


def build_model(nodes, members, supports, nodal_loads=None,
                member_dist_loads=None, member_point_loads=None,
                auto_stabilize_2d=True):
    """Build (but do not analyze) a PyNite FEModel3D from flat dicts."""
    FEModel3D = import_pynite()
    model = FEModel3D()

    if not nodes or not members or not supports:
        raise ValueError("nodes, members, and supports are all required")

    names = set()
    for nd in nodes:
        name = str(nd["name"])
        if name in names:
            raise ValueError(f"duplicate node name '{name}'")
        names.add(name)
        model.add_node(name, _num(nd["x"], "node x"), _num(nd["y"], "node y"),
                       _num(nd.get("z", 0.0), "node z"))
    is_2d = all(abs(float(nd.get("z", 0.0))) < 1e-12 for nd in nodes)

    for i, mb in enumerate(members):
        mname = str(mb.get("name", f"M{i + 1}"))
        for req in ("i", "j", "E", "A", "Iz"):
            if req not in mb:
                raise ValueError(f"member '{mname}' missing '{req}'")
        if str(mb["i"]) not in names or str(mb["j"]) not in names:
            raise ValueError(f"member '{mname}' references unknown node")
        E = _num(mb["E"], "E")                      # kPa
        nu = _num(mb.get("nu", 0.3), "nu")
        G = _num(mb["G"], "G") if "G" in mb else E / (2.0 * (1.0 + nu))
        A = _num(mb["A"], "A")                      # m^2
        Iz = _num(mb["Iz"], "Iz")                   # m^4 (major axis)
        Iy = _num(mb.get("Iy", Iz), "Iy")
        J = _num(mb.get("J", Iy + Iz), "J")
        mat = f"mat_{mname}"
        sec = f"sec_{mname}"
        model.add_material(mat, E, G, nu, mb.get("rho", 0.0))
        model.add_section(sec, A, Iy, Iz, J)
        model.add_member(mname, str(mb["i"]), str(mb["j"]), mat, sec)

    supported_nodes = []
    for sp in supports:
        node = str(sp["node"])
        if node not in names:
            raise ValueError(f"support references unknown node '{node}'")
        if "type" in sp:
            preset = str(sp["type"]).lower()
            if preset not in SUPPORT_PRESETS:
                raise ValueError(
                    f"unknown support type '{preset}' "
                    f"(available: {sorted(SUPPORT_PRESETS)})")
            dx, dy, dz, rx, ry, rz = SUPPORT_PRESETS[preset]
        else:
            dx = bool(sp.get("dx", False)); dy = bool(sp.get("dy", False))
            dz = bool(sp.get("dz", False)); rx = bool(sp.get("rx", False))
            ry = bool(sp.get("ry", False)); rz = bool(sp.get("rz", False))
        if is_2d and auto_stabilize_2d:
            dz, rx, ry = True, True, True
        model.def_support(node, dx, dy, dz, rx, ry, rz)
        supported_nodes.append(node)

    if is_2d and auto_stabilize_2d:
        for name in names.difference(supported_nodes):
            model.def_support(name, False, False, True, True, True, False)

    for ld in (nodal_loads or []):
        node = str(ld["node"])
        direction = str(ld.get("direction", "FY")).upper()
        if direction not in ("FX", "FY", "FZ", "MX", "MY", "MZ"):
            raise ValueError(f"unknown nodal load direction '{direction}'")
        model.add_node_load(node, direction, _num(ld["value"], "load value"))

    for ld in (member_dist_loads or []):
        member = str(ld["member"])
        direction = str(ld.get("direction", "FY")).upper()
        if direction not in ("FX", "FY", "FZ", "Fx", "Fy", "Fz"):
            raise ValueError(f"unknown dist load direction '{direction}'")
        w1 = _num(ld.get("w1", ld.get("w")), "w1")
        w2 = _num(ld.get("w2", ld.get("w", w1)), "w2")
        kwargs = {}
        if "x1" in ld:
            kwargs["x1"] = _num(ld["x1"], "x1")
        if "x2" in ld:
            kwargs["x2"] = _num(ld["x2"], "x2")
        model.add_member_dist_load(member, direction, w1, w2, **kwargs)

    for ld in (member_point_loads or []):
        member = str(ld["member"])
        direction = str(ld.get("direction", "FY")).upper()
        model.add_member_pt_load(member, direction,
                                 _num(ld["value"], "P"), _num(ld["x"], "x"))

    return model, supported_nodes


def extract_results(model, supported_nodes) -> FrameResult:
    """Pull reactions + per-member envelopes out of an analyzed model."""
    reactions = {}
    for node in supported_nodes:
        nd = model.nodes[node]
        reactions[node] = {
            "FX_kN": float(nd.RxnFX[_COMBO]),
            "FY_kN": float(nd.RxnFY[_COMBO]),
            "FZ_kN": float(nd.RxnFZ[_COMBO]),
            "MX_kNm": float(nd.RxnMX[_COMBO]),
            "MY_kNm": float(nd.RxnMY[_COMBO]),
            "MZ_kNm": float(nd.RxnMZ[_COMBO]),
        }

    members_out = []
    max_defl = 0.0
    for name, mem in model.members.items():
        mmax = float(mem.max_moment("Mz"))
        mmin = float(mem.min_moment("Mz"))
        vmax = float(mem.max_shear("Fy"))
        vmin = float(mem.min_shear("Fy"))
        amax = float(mem.max_axial())
        amin = float(mem.min_axial())
        dmax = float(mem.max_deflection("dy"))
        dmin = float(mem.min_deflection("dy"))
        d_abs = max(abs(dmax), abs(dmin))
        max_defl = max(max_defl, d_abs)
        members_out.append(MemberResult(
            name=name,
            max_moment_kNm=mmax, min_moment_kNm=mmin,
            moment_abs_kNm=max(abs(mmax), abs(mmin)),
            max_shear_kN=vmax, min_shear_kN=vmin,
            shear_abs_kN=max(abs(vmax), abs(vmin)),
            max_axial_kN=amax, min_axial_kN=amin,
            max_deflection_m=dmax, min_deflection_m=dmin,
            deflection_abs_mm=d_abs * 1000.0,
        ))

    return FrameResult(
        n_nodes=len(model.nodes), n_members=len(model.members),
        reactions=reactions, members=members_out,
        max_deflection_mm=max_defl * 1000.0,
    )


def analyze_frame(nodes, members, supports, nodal_loads=None,
                  member_dist_loads=None, member_point_loads=None,
                  auto_stabilize_2d=True) -> FrameResult:
    """Linear-elastic frame analysis.

    Parameters
    ----------
    nodes : list of dict
        ``{"name", "x", "y"[, "z"]}`` — coordinates in m.
    members : list of dict
        ``{"name", "i", "j", "E" (kPa), "A" (m^2), "Iz" (m^4)
        [, "Iy", "J", "G", "nu"]}``. Iy defaults to Iz, J to Iy+Iz,
        G from E and nu (default 0.3).
    supports : list of dict
        ``{"node", "type": fixed|pinned|roller_y|roller_x}`` or explicit
        ``dx..rz`` boolean flags.
    nodal_loads : list of dict, optional
        ``{"node", "direction": FX|FY|FZ|MX|MY|MZ, "value"}`` (kN, kN*m).
    member_dist_loads : list of dict, optional
        ``{"member", "direction": FX|FY|FZ (global), "w"}`` uniform, or
        ``w1``/``w2`` (+ optional ``x1``/``x2``) trapezoidal, in kN/m.
    member_point_loads : list of dict, optional
        ``{"member", "direction", "value", "x"}`` — kN at x m from i-end.
    auto_stabilize_2d : bool
        For all-z=0 models, restrain out-of-plane DOFs everywhere.

    Returns
    -------
    FrameResult
    """
    model, supported = build_model(
        nodes, members, supports, nodal_loads=nodal_loads,
        member_dist_loads=member_dist_loads,
        member_point_loads=member_point_loads,
        auto_stabilize_2d=auto_stabilize_2d)
    model.analyze()
    return extract_results(model, supported)
