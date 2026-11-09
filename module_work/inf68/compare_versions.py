"""INF-68 check: run the site-response cases on whichever pystrata is installed
and write the results as JSON (python compare_versions.py out.json). Run it
once in a Python 3.12 venv (pystrata 0.5.4) and once in a 3.13 venv (0.6)."""
import json, sys
import numpy as np
import pystrata
from pystrata_agent import analyze_eql_site_response, analyze_linear_site_response

def dar(t=10.0, Vs=250, uw=18.0, pi=30):
    return {"thickness": t, "Vs": Vs, "unit_wt": uw, "soil_model": "darendeli", "plas_index": pi}
def menq(t=8.0, Vs=350, uw=20.0):
    return {"thickness": t, "Vs": Vs, "unit_wt": uw, "soil_model": "menq"}
def lin(t=5.0, Vs=400, uw=20.0, d=0.02):
    return {"thickness": t, "Vs": Vs, "unit_wt": uw, "soil_model": "linear", "damping": d}
def cust(t=12.0, Vs=180, uw=17.0):
    return {"thickness": t, "Vs": Vs, "unit_wt": uw, "soil_model": "custom",
            "strains": [0.0001, 0.001, 0.01, 0.1], "mod_reduc": [1.0, 0.92, 0.55, 0.10],
            "damping_values": [0.01, 0.025, 0.07, 0.20]}
rock = {"thickness": 0, "Vs": 760, "unit_wt": 24.0, "soil_model": "linear", "damping": 0.01}

cases = {
  "darendeli_10m_pi30": ([dar(), rock], {}),
  "darendeli_10m_pi30_0.05g": ([dar(), rock], {"target_pga_g": 0.05}),
  "menq_8m": ([menq(), rock], {}),
  "menq_8m_0.6g": ([menq(), rock], {"target_pga_g": 0.6}),
  "dar_menq_lin_15m": ([dar(5.0), menq(5.0), lin(5.0), rock], {}),
  "custom_12m": ([cust(), rock], {}),
  "dar_over_menq_0.2g": ([dar(6.0, 200, 17.5, 20), menq(10.0, 320, 20.0), rock], {"target_pga_g": 0.2}),
}
import importlib.metadata as md
out = {"pystrata": md.version("pystrata"), "python": sys.version.split()[0]}
for name, (layers, kw) in cases.items():
    r = analyze_eql_site_response(layers=layers, motion="synthetic_pulse", **kw)
    out[name] = {"amp": r.amplification_factor, "pga_s": r.pga_surface_g,
                 "n_it": r.n_iterations, "conv": r.converged,
                 "vs_c": [float(x) for x in r.compatible_Vs],
                 "vs_i": [float(x) for x in r.initial_Vs],
                 "strain_max_pct": float(np.max(r.max_strain_pct)),
                 "sa_peak": float(np.max(r.Sa_surface_g))}
for name in ("darendeli_10m_pi30", "dar_menq_lin_15m"):
    layers = cases[name][0]
    r = analyze_linear_site_response(layers=layers, motion="synthetic_pulse")
    out["linear:" + name] = {"amp": r.amplification_factor, "pga_s": r.pga_surface_g,
                 "n_it": r.n_iterations, "conv": r.converged,
                 "vs_c": [float(x) for x in r.compatible_Vs], "vs_i": [float(x) for x in r.initial_Vs],
                 "strain_max_pct": float(np.max(r.max_strain_pct)), "sa_peak": float(np.max(r.Sa_surface_g))}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("ok", out["pystrata"])
