"""
Utility functions for pydiggs_agent module.
"""

import os


def has_pydiggs() -> bool:
    """
    Check if pydiggs is installed.

    Returns:
        bool: True if pydiggs is available, False otherwise
    """
    try:
        import pydiggs
        return True
    except ImportError:
        return False


def get_schema_path(version: str = "2.6") -> str:
    """
    Get the path to the bundled DIGGS schema file.

    Args:
        version: Schema version ("2.6" or "2.5.a")

    Returns:
        Absolute path to the schema XSD file

    Raises:
        ImportError: If pydiggs is not installed
        ValueError: If schema version is not supported
    """
    if not has_pydiggs():
        raise ImportError("pydiggs is not installed")

    import pydiggs

    pydiggs_dir = os.path.dirname(pydiggs.__file__)

    if version == "2.6":
        schema_path = os.path.join(pydiggs_dir, "schemas", "diggs-schema-2.6", "Diggs.xsd")
    elif version == "2.5.a":
        schema_path = os.path.join(pydiggs_dir, "schemas", "diggs-schema-2.5.a", "Complete.xsd")
    else:
        raise ValueError(f"Unsupported schema version: {version}. Use '2.6' or '2.5.a'")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    return schema_path


def get_dictionary_path() -> str:
    """
    Get the path to the bundled DIGGS dictionary file.

    Returns:
        Absolute path to the properties.xml dictionary file

    Raises:
        ImportError: If pydiggs is not installed
    """
    if not has_pydiggs():
        raise ImportError("pydiggs is not installed")

    import pydiggs

    pydiggs_dir = os.path.dirname(pydiggs.__file__)
    dict_path = os.path.join(pydiggs_dir, "dictionaries", "properties.xml")

    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    return dict_path
