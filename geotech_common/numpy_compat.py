"""
numpy names that differ between numpy 1.26 and numpy 2.

Foundry's transforms repos hold numpy at 1.26.4 (the platform's pyspark
needs numpy < 2), while every other host runs numpy 2 (BACKLOG INF-53). A
name numpy 2 added or renamed is taken from here, so the code runs on both.
"""

import numpy as np

# numpy 2.0 renamed trapz to trapezoid, and numpy 2.4 removed trapz.
trapezoid = getattr(np, "trapezoid", None) or np.trapz
