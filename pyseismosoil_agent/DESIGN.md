# PySeismoSoil Agent — Design Notes

## Purpose

Wraps the PySeismoSoil library for:
1. **Nonlinear soil curves** — G/Gmax and damping from MKZ and HH models
2. **Vs profile analysis** — Vs30, f0, z1 from layered shear wave profiles

## Architecture

```
pyseismosoil_agent/
  __init__.py              # exports generate_curves, analyze_vs_profile
  pyseismosoil_utils.py    # has_pyseismosoil(), import helpers
  soil_curves.py           # generate_curves(), analyze_vs_profile()
  results.py               # CurveResult, VsProfileResult
  tests/
    test_pyseismosoil_agent.py
  DESIGN.md
pyseismosoil_agent_foundry.py  # Foundry wrapper
```

## Constitutive Models

### MKZ (Modified Kodner-Zelasko)
Parameters: gamma_ref, beta, s, Gmax
- Widely used for modulus reduction
- Simpler than HH

### HH (Hybrid Hyperbolic)
Parameters: gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d
- More flexible, better fits large-strain behavior
- Includes explicit damping parameters

## Vs Profile Methods

- **Vs30** — time-averaged Vs in top 30m (NEHRP site classification)
- **f0 (Borcherdt-Hartzell)** — f0 = Vs/(4H) quarter-wavelength
- **f0 (Roesset)** — transfer function fundamental frequency
- **z1** — depth to Vs >= 1000 m/s (basin depth proxy)

## Profile Format

PySeismoSoil uses [thickness, Vs] arrays. Last layer thickness=0 = halfspace.

## Edge Cases

- Vs_Profile format is [thickness, Vs], NOT [depth, Vs]
- Last layer thickness must be 0 (halfspace)
- HH model requires all 9 parameters, MKZ requires 4
- Strain input is in percent (%), not decimal
