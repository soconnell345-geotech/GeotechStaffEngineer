"""DM7 reference adapter — 340+ NAVFAC DM7 equations (UFC 3-220-10/20).

Special handling: auto-discovers functions via inspect and resolves name
collisions across 15 chapter modules with chapter-key prefixes, mirroring
the logic in geotech-references/agents/dm7_agent.py.
"""

import inspect

from funhouse_agent.adapters._reference_common import (
    has_callable_param, extract_method_info, make_wrapper,
)


def _build():
    from geotech_references.dm7_1 import (
        chapter1 as dm7_1_ch1, chapter2 as dm7_1_ch2,
        chapter3 as dm7_1_ch3, chapter4 as dm7_1_ch4,
        chapter5 as dm7_1_ch5, chapter6 as dm7_1_ch6,
        chapter7 as dm7_1_ch7, chapter8 as dm7_1_ch8,
    )
    from geotech_references.dm7_2 import (
        prologue as dm7_2_pro,
        chapter2 as dm7_2_ch2, chapter3 as dm7_2_ch3,
        chapter4 as dm7_2_ch4, chapter5 as dm7_2_ch5,
        chapter6 as dm7_2_ch6, chapter7 as dm7_2_ch7,
    )

    CHAPTER_INFO = {
        "dm7_1_ch1": {
            "module": dm7_1_ch1,
            "category": "DM7.1 Ch1 - Identification & Classification",
            "reference": "UFC 3-220-10, Chapter 1",
        },
        "dm7_1_ch2": {
            "module": dm7_1_ch2,
            "category": "DM7.1 Ch2 - Field Exploration & Testing",
            "reference": "UFC 3-220-10, Chapter 2",
        },
        "dm7_1_ch3": {
            "module": dm7_1_ch3,
            "category": "DM7.1 Ch3 - Laboratory Testing",
            "reference": "UFC 3-220-10, Chapter 3",
        },
        "dm7_1_ch4": {
            "module": dm7_1_ch4,
            "category": "DM7.1 Ch4 - Distribution of Stresses",
            "reference": "UFC 3-220-10, Chapter 4",
        },
        "dm7_1_ch5": {
            "module": dm7_1_ch5,
            "category": "DM7.1 Ch5 - Consolidation & Settlement",
            "reference": "UFC 3-220-10, Chapter 5",
        },
        "dm7_1_ch6": {
            "module": dm7_1_ch6,
            "category": "DM7.1 Ch6 - Seepage & Drainage",
            "reference": "UFC 3-220-10, Chapter 6",
        },
        "dm7_1_ch7": {
            "module": dm7_1_ch7,
            "category": "DM7.1 Ch7 - Slope Stability",
            "reference": "UFC 3-220-10, Chapter 7",
        },
        "dm7_1_ch8": {
            "module": dm7_1_ch8,
            "category": "DM7.1 Ch8 - Correlations for Soil Properties",
            "reference": "UFC 3-220-10, Chapter 8",
        },
        "dm7_2_pro": {
            "module": dm7_2_pro,
            "category": "DM7.2 Prologue - Shear Strength",
            "reference": "UFC 3-220-20, Prologue",
        },
        "dm7_2_ch2": {
            "module": dm7_2_ch2,
            "category": "DM7.2 Ch2 - Excavations & Retained Cuts",
            "reference": "UFC 3-220-20, Chapter 2",
        },
        "dm7_2_ch3": {
            "module": dm7_2_ch3,
            "category": "DM7.2 Ch3 - Earthwork & Compaction",
            "reference": "UFC 3-220-20, Chapter 3",
        },
        "dm7_2_ch4": {
            "module": dm7_2_ch4,
            "category": "DM7.2 Ch4 - Rigid Retaining Structures",
            "reference": "UFC 3-220-20, Chapter 4",
        },
        "dm7_2_ch5": {
            "module": dm7_2_ch5,
            "category": "DM7.2 Ch5 - Shallow Foundations",
            "reference": "UFC 3-220-20, Chapter 5",
        },
        "dm7_2_ch6": {
            "module": dm7_2_ch6,
            "category": "DM7.2 Ch6 - Deep Foundations",
            "reference": "UFC 3-220-20, Chapter 6",
        },
        "dm7_2_ch7": {
            "module": dm7_2_ch7,
            "category": "DM7.2 Ch7 - Probability & Reliability",
            "reference": "UFC 3-220-20, Chapter 7",
        },
    }

    registry = {}
    info = {}
    name_collisions = {}  # name -> first chapter key that owns it

    for ch_key, ch_meta in CHAPTER_INFO.items():
        mod = ch_meta["module"]
        cat = ch_meta["category"]
        ref = ch_meta["reference"]

        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if name.startswith("_"):
                continue
            if has_callable_param(func):
                continue

            if name in registry:
                # Name collision — prefix both entries with chapter key
                if name not in name_collisions:
                    # Rename the existing (first) entry
                    first_ch = None
                    for prev_key, prev_meta in CHAPTER_INFO.items():
                        if prev_key == ch_key:
                            break
                        prev_mod = prev_meta["module"]
                        if (hasattr(prev_mod, name)
                                and getattr(prev_mod, name)
                                is registry[name].__wrapped__):
                            first_ch = prev_key
                            break
                    if first_ch:
                        new_name = f"{first_ch}_{name}"
                        registry[new_name] = registry.pop(name)
                        info[new_name] = info.pop(name)
                        name_collisions[name] = first_ch

                qualified = f"{ch_key}_{name}"
            else:
                qualified = name

            registry[qualified] = make_wrapper(func)
            info[qualified] = extract_method_info(func, cat, ref)

    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
