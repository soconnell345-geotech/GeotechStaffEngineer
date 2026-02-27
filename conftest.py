"""
Root-level pytest conftest.py — shared configuration for all test modules.

Sets the matplotlib backend to 'Agg' (non-interactive) once, so individual
test files don't each need to do it themselves.
"""

import matplotlib
matplotlib.use("Agg")
