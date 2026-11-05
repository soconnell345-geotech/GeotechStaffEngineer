"""Moved to ``funhouse_agent.charts.plan_furniture`` (INF-31, the app split's step 6).

This name is the same module, so an import or a patch by either name
reaches the one copy of the code. The old name goes at step 16.
"""

import sys as _sys

from funhouse_agent.charts import plan_furniture as _moved

_sys.modules[__name__] = _moved
