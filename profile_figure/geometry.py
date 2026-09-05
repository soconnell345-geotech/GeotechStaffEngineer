"""
Geometry resolution and validation for subsurface profile schematics.

Pure Python — no matplotlib — so the stacking rules can be tested (and the
errors raised) without a plotting backend.

Everything is resolved to ELEVATIONS internally, even when the caller works in
depths: a depth ``d`` is elevation ``ground - d``.  The vertical axis can still
be *labelled* in depth (see ``axis``), which changes presentation only.

All units SI: meters (m), kPa for surcharge.
"""

from typing import Optional

#: Elevations closer than this are treated as coincident (m).
TOL = 1e-6

#: Foundation kinds the overlay knows how to draw.
FOUNDATION_TYPES = ("pile", "micropile", "drilled_shaft", "footing", "wall")

#: Vertical-axis labelling modes.
AXIS_MODES = ("auto", "elevation", "depth")


def _num(value, what: str) -> float:
    """Coerce to float with an actionable message instead of a TypeError."""
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"profile_figure: {what} must be a number (m), got {value!r}.")


def _as_dict(item, what: str) -> dict:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "to_dict"):
        return dict(item.to_dict())
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    raise ValueError(
        f"profile_figure: each {what} must be a dict, got {type(item).__name__}.")


def _elevation_of(spec: dict, elev_key: str, depth_key: str, ground: float,
                  what: str) -> Optional[float]:
    """Resolve an elevation given either an elevation or a depth key."""
    if spec.get(elev_key) is not None:
        return _num(spec[elev_key], f"{what}.{elev_key}")
    if spec.get(depth_key) is not None:
        return ground - _num(spec[depth_key], f"{what}.{depth_key}")
    return None


def _has_elevation_input(layers, ground_elevation, extras) -> bool:
    """True when the caller expressed anything in elevations (not depths)."""
    if ground_elevation is not None:
        return True
    for lay in layers:
        if lay.get("top_elevation") is not None or \
                lay.get("bottom_elevation") is not None:
            return True
    for spec in extras:
        if not spec:
            continue
        if any(k.endswith("_elevation") and spec.get(k) is not None
               for k in spec):
            return True
    return False


def resolve_ground(layers, ground_elevation=None) -> float:
    """Ground-surface elevation: explicit, else the first layer top, else 0."""
    if ground_elevation is not None:
        return _num(ground_elevation, "ground_elevation")
    if layers and layers[0].get("top_elevation") is not None:
        return _num(layers[0]["top_elevation"], "layers[0].top_elevation")
    return 0.0


def resolve_layers(layers, ground: float) -> list:
    """Stack layers top-down into resolved {name, top, bottom, thickness} dicts.

    A layer's top is its ``top_elevation`` / ``top_depth`` when given, else the
    previous layer's bottom (the first layer starts at the ground surface).
    Its bottom comes from ``bottom_elevation`` / ``bottom_depth`` / ``thickness``.

    Raises
    ------
    ValueError
        On a layer with no resolvable bottom, a non-positive thickness, or a
        gap/overlap against the layer above — a subsurface column has to be
        continuous, and a silent gap would draw a misleading figure.
    """
    if not layers:
        raise ValueError(
            "profile_figure: 'layers' is required and must hold at least one "
            "layer, e.g. [{'name': 'Soft clay', 'thickness': 6.0}].")

    resolved = []
    prev_bottom = None
    for i, raw in enumerate(layers):
        spec = _as_dict(raw, "layers[] entry")
        name = str(spec.get("name") or spec.get("description")
                   or f"Layer {i + 1}")
        label = f"layers[{i}] ('{name}')"

        top = _elevation_of(spec, "top_elevation", "top_depth", ground, label)
        if top is None:
            top = ground if prev_bottom is None else prev_bottom
        elif prev_bottom is not None and abs(top - prev_bottom) > 1e-4:
            gap = prev_bottom - top
            how = "gap" if gap > 0 else "overlap"
            raise ValueError(
                f"profile_figure: {label} top El. {top:.3f} m leaves a "
                f"{abs(gap):.3f} m {how} below the previous layer, which ends "
                f"at El. {prev_bottom:.3f} m. Layers must stack continuously — "
                "give thicknesses and omit the tops, or fix the elevations.")

        bottom = _elevation_of(spec, "bottom_elevation", "bottom_depth",
                               ground, label)
        if bottom is None:
            if spec.get("thickness") is None:
                raise ValueError(
                    f"profile_figure: {label} needs a 'thickness' (m) or a "
                    "'bottom_elevation'/'bottom_depth'.")
            bottom = top - _num(spec["thickness"], f"{label}.thickness")

        thickness = top - bottom
        if thickness <= TOL:
            raise ValueError(
                f"profile_figure: {label} has thickness {thickness:.3f} m — "
                "every layer must be thicker than zero, with its bottom BELOW "
                "its top (elevations decrease with depth).")

        resolved.append({
            "name": name,
            "top": top,
            "bottom": bottom,
            "thickness": thickness,
            "description": str(spec.get("description") or ""),
            "settling": bool(spec.get("settling", False)),
            "color": spec.get("color"),
            "hatch": spec.get("hatch"),
        })
        prev_bottom = bottom

    return resolved


def resolve_fill(fill, ground: float) -> Optional[dict]:
    """Resolve the optional fill / embankment block placed ABOVE ground."""
    if fill in (None, False):
        return None
    if isinstance(fill, (int, float)):
        fill = {"thickness": fill}
    spec = _as_dict(fill, "fill")
    top = _elevation_of(spec, "top_elevation", "top_depth", ground, "fill")
    if top is None:
        if spec.get("thickness") is None:
            raise ValueError(
                "profile_figure: 'fill' needs a 'thickness' (m) or a "
                "'top_elevation'.")
        top = ground + _num(spec["thickness"], "fill.thickness")
    thickness = top - ground
    if thickness <= TOL:
        raise ValueError(
            f"profile_figure: fill thickness is {thickness:.3f} m — fill sits "
            "ABOVE the ground surface, so its top elevation must be higher "
            f"than the ground surface (El. {ground:.3f} m).")
    return {
        "name": str(spec.get("name") or "Fill"),
        "top": top,
        "bottom": ground,
        "thickness": thickness,
        "description": str(spec.get("description") or ""),
        "color": spec.get("color"),
        "hatch": spec.get("hatch"),
        "settling": bool(spec.get("settling", False)),
    }


def resolve_water(water_elevation, water_depth, ground: float) -> Optional[float]:
    if water_elevation is not None:
        return _num(water_elevation, "water_elevation")
    if water_depth is not None:
        return ground - _num(water_depth, "water_depth")
    return None


def resolve_surcharge(surcharge) -> Optional[dict]:
    if surcharge in (None, False):
        return None
    if isinstance(surcharge, (int, float)):
        surcharge = {"pressure": surcharge}
    spec = _as_dict(surcharge, "surcharge")
    if spec.get("pressure") is None:
        raise ValueError(
            "profile_figure: 'surcharge' needs a 'pressure' (kPa), e.g. "
            "{'pressure': 20.0} or just 20.0.")
    pressure = _num(spec["pressure"], "surcharge.pressure")
    label = spec.get("label") or f"q = {pressure:g} kPa"
    return {"pressure": pressure, "label": str(label)}


def resolve_foundation(foundation, ground: float, top: float) -> Optional[dict]:
    """Resolve the optional foundation overlay (pile / shaft / footing / wall).

    ``top`` is the top of the drawn section (fill top when fill is present),
    used as the default head elevation.
    """
    if foundation in (None, False):
        return None
    spec = _as_dict(foundation, "foundation")

    ftype = str(spec.get("type") or "pile").strip().lower()
    if ftype not in FOUNDATION_TYPES:
        raise ValueError(
            f"profile_figure: foundation type '{ftype}' is not supported. "
            f"Use one of {list(FOUNDATION_TYPES)}.")

    head = _elevation_of(spec, "head_elevation", "head_depth", ground,
                         "foundation")
    if head is None:
        head = top

    tip = _elevation_of(spec, "tip_elevation", "tip_depth", ground,
                        "foundation")
    if tip is None:
        for key in ("length", "embedded_length", "thickness"):
            if spec.get(key) is not None:
                tip = head - _num(spec[key], f"foundation.{key}")
                break
    if tip is None:
        raise ValueError(
            "profile_figure: the foundation needs a 'tip_elevation' (or "
            "'tip_depth', or a 'length' below the head) so the schematic knows "
            "how deep it goes.")

    length = head - tip
    if length <= TOL:
        raise ValueError(
            f"profile_figure: the foundation tip (El. {tip:.3f} m) must be "
            f"BELOW its head (El. {head:.3f} m) — elevations decrease with "
            "depth. Got a length of "
            f"{length:.3f} m.")

    size = None
    for key in ("diameter", "width", "size"):
        if spec.get(key) is not None:
            size = _num(spec[key], f"foundation.{key}")
            break
    if size is None or size <= 0:
        raise ValueError(
            "profile_figure: the foundation needs a 'diameter' (pile/shaft/"
            "wall) or 'width' (footing) in meters — it labels the schematic "
            "even though the horizontal scale is exaggerated.")

    default_labels = {
        "pile": "Pile", "micropile": "Micropile",
        "drilled_shaft": "Drilled shaft", "footing": "Footing", "wall": "Wall",
    }
    label = str(spec.get("label") or default_labels[ftype])

    return {
        "type": ftype,
        "head": head,
        "tip": tip,
        "length": length,
        "size": size,
        "size_key": "width" if ftype == "footing" else "diameter",
        "label": label,
    }


def resolve_annotations(annotations, ground: float) -> list:
    """Resolve callouts to {elevation, text, side} dicts."""
    if not annotations:
        return []
    out = []
    for i, raw in enumerate(annotations):
        spec = _as_dict(raw, "annotations[] entry")
        elev = _elevation_of(spec, "elevation", "depth", ground,
                             f"annotations[{i}]")
        if elev is None:
            raise ValueError(
                f"profile_figure: annotations[{i}] needs an 'elevation' or a "
                "'depth' (m) to point at.")
        text = str(spec.get("text") or spec.get("label") or "").strip()
        if not text:
            raise ValueError(
                f"profile_figure: annotations[{i}] needs a 'text' string.")
        side = str(spec.get("side") or "right").strip().lower()
        if side not in ("left", "right"):
            raise ValueError(
                f"profile_figure: annotations[{i}] side '{side}' must be "
                "'left' or 'right'.")
        out.append({"elevation": elev, "text": text, "side": side})
    return out


def resolve_profile(layers, *, ground_elevation=None, fill=None,
                    water_elevation=None, water_depth=None, surcharge=None,
                    foundation=None, annotations=None, axis="auto") -> dict:
    """Resolve every input into drawable geometry, validating as it goes.

    Returns a dict with ``layers``, ``ground``, ``base``, ``top``, ``fill``,
    ``water``, ``surcharge``, ``foundation``, ``annotations``, ``axis`` and
    ``warnings``.
    """
    axis = str(axis or "auto").strip().lower()
    if axis not in AXIS_MODES:
        raise ValueError(
            f"profile_figure: axis '{axis}' must be one of {list(AXIS_MODES)}.")

    layer_specs = [_as_dict(lay, "layers[] entry") for lay in (layers or [])]
    ground = resolve_ground(layer_specs, ground_elevation)

    resolved_layers = resolve_layers(layer_specs, ground)
    fill_block = resolve_fill(fill, ground)
    water = resolve_water(water_elevation, water_depth, ground)
    surcharge_block = resolve_surcharge(surcharge)
    top = fill_block["top"] if fill_block else ground
    foundation_block = resolve_foundation(foundation, ground, top)
    notes = resolve_annotations(annotations, ground)

    base = resolved_layers[-1]["bottom"]

    if axis == "auto":
        extras = [f for f in (fill if isinstance(fill, dict) else None,
                              foundation if isinstance(foundation, dict) else None)
                  if f]
        if water_elevation is not None:
            extras.append({"water_elevation": water_elevation})
        axis = ("elevation"
                if _has_elevation_input(layer_specs, ground_elevation, extras)
                else "depth")

    warnings = []
    if water is not None:
        if water > top + TOL:
            warnings.append(
                f"water table (El. {water:.2f} m) is above the top of the "
                f"section (El. {top:.2f} m) — shown as ponded water.")
        elif water < base - TOL:
            warnings.append(
                f"water table (El. {water:.2f} m) is below the deepest layer "
                f"(El. {base:.2f} m) and is not drawn on the section.")
    if foundation_block:
        if foundation_block["tip"] < base - 1e-4:
            warnings.append(
                f"{foundation_block['label']} tip (El. "
                f"{foundation_block['tip']:.2f} m) is below the deepest layer "
                f"(El. {base:.2f} m) — the profile does not cover the "
                "bearing stratum; add the layer the foundation bears in.")
        if foundation_block["head"] > top + 1e-4:
            warnings.append(
                f"{foundation_block['label']} head (El. "
                f"{foundation_block['head']:.2f} m) is above the ground/fill "
                "surface — shown as a stick-up.")
    for note in notes:
        lo = min(base, foundation_block["tip"]) if foundation_block else base
        if not (lo - TOL <= note["elevation"] <= top + TOL):
            warnings.append(
                f"callout '{note['text']}' at El. {note['elevation']:.2f} m "
                "falls outside the drawn section.")

    return {
        "layers": resolved_layers,
        "ground": ground,
        "base": base,
        "top": top,
        "fill": fill_block,
        "water": water,
        "surcharge": surcharge_block,
        "foundation": foundation_block,
        "annotations": notes,
        "axis": axis,
        "warnings": warnings,
    }
