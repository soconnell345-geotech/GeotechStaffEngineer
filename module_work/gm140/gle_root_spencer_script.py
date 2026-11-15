"""
Spencer's two equations, written apart from ``slope_stability/gle.py``
(GM-140, "The GLE solver: a root, or no number"; check (b)).

What is solved. Spencer (1967, pp. 14 to 15; the account's section 9 cites
Eqs. 7 and 8, p. 15, and Wright's (1969) extension to noncircular surfaces):
the forces between slices are parallel, at one inclination theta, and the
resultant Q of the two interslice forces on a slice acts through the
midpoint of its base. F and theta are the pair at which

    sum(Q) = 0                              (force equilibrium), and
    sum(Q x arm about any one point) = 0    (moment equilibrium).

Spencer (1967) is not in a repository, so the slice equation for Q is NOT
transcribed here: it is the statics of one slice, solved as a 2 x 2 linear
system in the base normal force N and Q (the weight, N, the mobilised shear
S = (c l + (N - u l) tan phi) / F along the base, and Q at theta). The
module's solver works in the Fredlund-Krahn form (E and X on each side,
moments of W, N and S about an axis); this script has no E, no X, no lambda
and no axis fitted through the bases, so the two share only the slices.

Two settings (the brief's ruling 3):

  as printed   the equations above on every slice, with no rule for a
               base in tension;
  no friction  a slice whose effective normal force N - u l is negative is
               solved again with tan(phi) = 0 in both of its equations
               (the module's device, D29).

Run:  python module_work/gm140/gle_root_spencer_script.py
It prints every root of the made-up thin planar case of the brief in
theta 0 to 75 deg and F 0.2 to 3, under both settings, with the pore
pressure given as r_u and dry.
"""

import math
from typing import List, Optional, Tuple


def slice_NQ(s, F: float, theta: float, device: bool):
    """(N, Q, frictionless) of one slice at trial F and theta (radians).

    Axes x to the right, z up. Sliding is toward -x (the high side on the
    right), so the shear on the base resists along +t, t = (cos a, sin a).
    Q is the net interslice force on the slice, along (cos theta,
    sin theta). Unknowns N (on the base, along n = (-sin a, cos a)) and Q:

        x:  -N sin a + S cos a + Q cos theta - H = 0
        z:   N cos a + S sin a + Q sin theta - W = 0
        S = (c l + (N - u l) tan phi) / F

    H is a horizontal driving force on the slice (0 in the cases here).
    """
    W, a, l, u, c, H = s["W"], s["alpha"], s["l"], s["u"], s["c"], s["H"]
    sa, ca = math.sin(a), math.cos(a)
    st, ct = math.sin(theta), math.cos(theta)

    def solve(tp):
        k = (c * l - u * l * tp) / F        # the part of S not in N
        a11 = -sa + tp * ca / F
        a21 = ca + tp * sa / F
        b1 = H - k * ca
        b2 = W - k * sa
        det = a11 * st - a21 * ct
        if abs(det) < 1e-14:
            return None
        N = (b1 * st - b2 * ct) / det
        Q = (a11 * b2 - a21 * b1) / det
        return N, Q

    out = solve(s["tan_phi"])
    if out is None:
        return None
    frictionless = False
    if device and out[0] - u * l < 0.0 and s["tan_phi"] > 0.0:
        out = solve(0.0)
        frictionless = True
        if out is None:
            return None
    return out[0], out[1], frictionless


def residuals(slices, F: float, theta: float, device: bool):
    """(sum Q, sum of Q's moment about the origin) or None."""
    fq = mq = 0.0
    st, ct = math.sin(theta), math.cos(theta)
    for s in slices:
        out = slice_NQ(s, F, theta, device)
        if out is None:
            return None
        Q = out[1]
        fq += Q
        mq += Q * (s["x"] * st - s["z"] * ct)    # r x (Q cos, Q sin)
    return fq, mq


def _newton(slices, F, theta, device, it=60):
    """Newton's method on the two residuals from (F, theta); the root, or
    None."""
    for _ in range(it):
        r = residuals(slices, F, theta, device)
        if r is None:
            return None
        hF, hT = 1e-7 * max(F, 1e-3), 1e-7
        rF = residuals(slices, F + hF, theta, device)
        rT = residuals(slices, F, theta + hT, device)
        if rF is None or rT is None:
            return None
        j11, j21 = (rF[0] - r[0]) / hF, (rF[1] - r[1]) / hF
        j12, j22 = (rT[0] - r[0]) / hT, (rT[1] - r[1]) / hT
        det = j11 * j22 - j12 * j21
        if det == 0.0:
            return None
        dF = (-r[0] * j22 + r[1] * j12) / det
        dT = (-r[1] * j11 + r[0] * j21) / det
        # damped: never more than a tenth of F or 2 deg in one step
        k = min(1.0, 0.1 * F / max(abs(dF), 1e-300),
                math.radians(2.0) / max(abs(dT), 1e-300))
        F, theta = F + k * dF, theta + k * dT
        if not (0.05 < F < 50.0) or not (-1.5 < theta < 1.5):
            return None
        if abs(dF) < 1e-11 and abs(dT) < 1e-11:
            return F, theta
    return None


def scale(slices) -> Tuple[float, float]:
    w = sum(abs(s["W"]) for s in slices)
    arm = max(math.hypot(s["x"], s["z"]) for s in slices)
    return w, w * arm


def is_root(slices, F, theta, device, rel=1e-8) -> bool:
    """Both residuals are zero to rounding (a pole is not a root)."""
    r = residuals(slices, F, theta, device)
    if r is None:
        return False
    w, m = scale(slices)
    return abs(r[0]) < rel * w and abs(r[1]) < rel * m


def all_roots(slices, device: bool,
              theta_deg=(0.0, 75.0), F_range=(0.2, 3.0),
              n_theta: int = 76, n_F: int = 29) -> List[Tuple[float, float]]:
    """Every (F, theta in degrees) the two equations meet in the window,
    by Newton's method from a grid of starts; each is checked to be a root
    (both residuals zero to rounding) and listed once."""
    found: List[Tuple[float, float]] = []
    for i in range(n_theta):
        th = math.radians(theta_deg[0] + (theta_deg[1] - theta_deg[0])
                          * i / (n_theta - 1))
        for j in range(n_F):
            F0 = F_range[0] + (F_range[1] - F_range[0]) * j / (n_F - 1)
            out = _newton(slices, F0, th, device)
            if out is None:
                continue
            F, t = out
            td = math.degrees(t)
            if not (theta_deg[0] <= td <= theta_deg[1]
                    and F_range[0] <= F <= F_range[1]):
                continue
            if not is_root(slices, F, t, device):
                continue
            if all(abs(F - f) > 1e-6 or abs(td - d) > 1e-5
                   for f, d in found):
                found.append((F, td))
    return sorted(found, key=lambda p: p[1])


def state(slices, F, theta_deg, device):
    """Per slice: (N', Q, frictionless), and the interslice force on each
    interior side (the running sum of Q from the low end, signed so that
    compression is positive: Z_j = -sum(Q_1..Q_j) along theta)."""
    th = math.radians(theta_deg)
    rows, Z, run = [], [], 0.0
    for s in slices:
        N, Q, fl = slice_NQ(s, F, th, device)
        rows.append((N - s["u"] * s["l"], Q, fl))
        run += Q
        Z.append(-run)
    return rows, Z[:-1]


# ---------------------------------------------------------------------------
# The slices of the module, as plain numbers (only the slicing is shared)
# ---------------------------------------------------------------------------

def from_module(slices) -> List[dict]:
    """Plain dicts from ``slope_stability.slices.build_slices`` output,
    turned so the high side is on the right (sliding toward -x)."""
    drive = sum((s.weight + s.surcharge_force) * math.sin(s.alpha)
                for s in slices)
    sg = -1.0 if drive < 0 else 1.0
    out = []
    for s in slices:
        out.append({
            "x": sg * s.x_mid, "z": s.z_base, "alpha": sg * s.alpha,
            "l": s.base_length, "W": s.weight + s.surcharge_force,
            "u": s.pore_pressure, "c": s.c,
            "tan_phi": math.tan(math.radians(s.phi)),
            "H": s.seismic_force + s.crack_water_force - sg * s.pond_hforce,
        })
    return sorted(out, key=lambda d: d["x"])


BETA = math.degrees(math.atan(0.5))      # the 2H:1V face, 26.565 deg
GAMMA, PHI, GAMMA_W = 19.5, 34.0, 9.81
SURFACE = [(20.0, 10.0), (22.0, 9.0), (380.0, 188.0), (382.0, 191.0)]
GROUND = [(-20.0, 0.0), (0.0, 0.0), (400.0, 200.0), (420.0, 200.0)]


def case(water: str, n_slices: int = 60):
    """(geometry, slip surface, module slices) of the brief's made-up thin
    planar case. ``water``: 'ru' (r_u = (gamma_w / gamma) cos^2(beta) on
    every slice, which is the u = gamma_w z cos^2(beta) of seepage parallel
    to the face), 'head' (a water line on the ground, read as the full
    head below it, u = gamma_w z: the module's reading of a water line) or
    'dry'."""
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.slip_surface import PolylineSlipSurface
    from slope_stability.slices import build_slices
    ru = (GAMMA_W / GAMMA) * math.cos(math.radians(BETA)) ** 2
    layer = SlopeSoilLayer(
        name="sand", top_elevation=200.0, bottom_elevation=-50.0,
        gamma=GAMMA, gamma_sat=GAMMA, phi=PHI, c_prime=0.0,
        ru=ru if water == "ru" else 0.0)
    kw = {}
    if water == "head":
        kw = {"gwt_points": list(GROUND)}
    geom = SlopeGeometry(surface_points=list(GROUND), soil_layers=[layer],
                         **kw)
    slip = PolylineSlipSurface(points=list(SURFACE))
    return geom, slip, build_slices(geom, slip, n_slices)


def closed_form(water: str) -> float:
    """The infinite slope, c' = 0 (FHWA NHI-06-088 Sec. 6.3.2, p. 6-8):
    dry, tan(phi') / tan(beta); seepage parallel to the face (Eq. 6-10),
    (gamma' / gamma_sat) tan(phi') / tan(beta)."""
    t = math.tan(math.radians(PHI)) / math.tan(math.radians(BETA))
    if water == "dry":
        return t
    if water == "head":         # u = gamma_w z: r_u = gamma_w / gamma
        b = math.radians(BETA)
        return (1.0 - GAMMA_W / GAMMA / math.cos(b) ** 2) * t
    return (GAMMA - GAMMA_W) / GAMMA * t


def main():
    # ('head' is left out: with a water line on the ground the module's
    # slices carry a pond thrust above the base, which this script's one
    # horizontal force through the base midpoint does not place)
    for water in ("ru", "dry"):
        try:
            _, _, sl = case(water)
        except Exception as exc:                      # pragma: no cover
            print(f"{water}: could not build the case ({exc})")
            continue
        d = from_module(sl)
        print(f"\n{water}: closed form {closed_form(water):.4f}, "
              f"{len(d)} slices")
        for device in (False, True):
            name = "no friction" if device else "as printed "
            for F, td in all_roots(d, device):
                rows, Z = state(d, F, td, device)
                tb = [i + 1 for i, r in enumerate(rows) if r[0] < -1e-9]
                ts = [j + 1 for j, z in enumerate(Z) if z < -1e-6]
                print(f"  {name}: theta {td:7.3f} deg, F {F:.4f}; "
                      f"N' slice 1 {rows[0][0]:9.2f} kN/m, "
                      f"side 1 {Z[0]:9.2f} kN/m; "
                      f"bases in tension {tb[:6]}, sides in tension "
                      f"{ts[:6]}{'...' if len(ts) > 6 else ''}")


if __name__ == "__main__":
    main()
