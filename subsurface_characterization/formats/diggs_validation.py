"""
DIGGS XML validation format adapter.

Provides validation functions for DIGGS (Data Interchange for Geotechnical and
Geoenvironmental Specialists) XML files.

**Schema check, DIGGS 2.6: no optional package needed.** The DIGGS 2.6 schema
ships with this package (``schemas/diggs-schema-2.6.zip``, the unmodified
files under their MPL-2.0 licence; see ``schemas/README.md``) and is checked
with lxml, which is already a dependency. Before 2026-10-08 the check went
through the optional ``pydiggs`` package, which is deliberately not installed
on Databricks (its dependencies replaced the notebook's Pygments and the
kernel was killed, 5.10.1), so on the cluster nothing could check a DIGGS
file at all -- and in the 2026-10-06 field session the agent wrote a
hand-typed "DIGGS" file that fails the schema at its root.

**Dictionary check and DIGGS 2.5.a: still pydiggs**, an OPTIONAL dependency.
Use ``has_pydiggs()`` to check availability; those functions raise a clear
ImportError when it is missing.

Public API
----------
validate_diggs_schema() - Validate DIGGS XML against XSD schema
validate_diggs_dictionary() - Validate DIGGS propertyClass values against dictionary
DiggValidationResult - Result dataclass with summary() and to_dict()
has_pydiggs() - Check if pydiggs is installed
has_schema_check() - Whether a schema version can be checked here
bundled_schema_path() - The bundled DIGGS 2.6 ``Diggs.xsd``, unpacked on first use

Note: this is the *validation* adapter (pydiggs). For DIGGS *data extraction* into a
SiteModel, use ``subsurface_characterization.parse_diggs`` (native parser, no
external dependency).

Example
-------
>>> from subsurface_characterization.formats import validate_diggs_schema
>>> result = validate_diggs_schema(filepath="report.xml")
>>> print(result.summary())
"""

import hashlib
import os
import tempfile
import threading
import zipfile
from typing import Dict, Optional

from subsurface_characterization.formats.diggs_validation_results import (
    DiggValidationResult,
)


#: The DIGGS schema version that ships with this package.
BUNDLED_SCHEMA_VERSION = "2.6"

#: The bundled schema: every file ``Diggs.xsd`` imports, unmodified, zipped.
SCHEMA_ZIP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "schemas", "diggs-schema-2.6.zip")

#: Where the zip is unpacked (a folder); the system temp folder by default.
SCHEMA_CACHE_ENV = "GEOTECH_DIGGS_SCHEMA_CACHE"

_lock = threading.Lock()
_unpacked: Optional[str] = None
_compiled: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# Optional-dependency guards
# ---------------------------------------------------------------------------

def has_pydiggs() -> bool:
    """
    Check if pydiggs is installed.

    Returns:
        bool: True if pydiggs is available, False otherwise
    """
    try:
        import pydiggs  # noqa: F401
        return True
    except ImportError:
        return False


def _has_lxml() -> bool:
    try:
        from lxml import etree  # noqa: F401
        return True
    except ImportError:
        return False


def has_schema_check(schema_version: str = BUNDLED_SCHEMA_VERSION) -> bool:
    """Whether ``schema_version`` can be checked here: DIGGS 2.6 needs only
    lxml and the bundled schema; 2.5.a needs pydiggs."""
    if schema_version == BUNDLED_SCHEMA_VERSION:
        return _has_lxml() and os.path.isfile(SCHEMA_ZIP)
    return has_pydiggs()


def _cache_root() -> str:
    return os.environ.get(SCHEMA_CACHE_ENV) or os.path.join(
        tempfile.gettempdir(), "geotech_diggs_schema")


def bundled_schema_path() -> str:
    """The bundled DIGGS 2.6 ``Diggs.xsd``, unpacked once per zip.

    The zip is unpacked into a folder named by its own hash (so an upgraded
    package never reads an older copy), first into a private folder and then
    renamed into place, so two processes unpacking at once both end with a
    complete copy. Raises ``FileNotFoundError`` when the zip is missing."""
    global _unpacked
    with _lock:
        if _unpacked and os.path.isfile(os.path.join(_unpacked, "Diggs.xsd")):
            return os.path.join(_unpacked, "Diggs.xsd")
        if not os.path.isfile(SCHEMA_ZIP):
            raise FileNotFoundError(f"bundled DIGGS schema not found: {SCHEMA_ZIP}")
        with open(SCHEMA_ZIP, "rb") as fh:
            tag = hashlib.sha256(fh.read()).hexdigest()[:12]
        root = _cache_root()
        target = os.path.join(root, f"diggs-schema-{BUNDLED_SCHEMA_VERSION}-{tag}")
        if not os.path.isfile(os.path.join(target, "Diggs.xsd")):
            os.makedirs(root, exist_ok=True)
            work = tempfile.mkdtemp(prefix=".unpack-", dir=root)
            with zipfile.ZipFile(SCHEMA_ZIP) as zf:
                zf.extractall(work)
            try:
                os.replace(work, target)
            except OSError:
                # Another process put a complete copy there first.
                import shutil
                shutil.rmtree(work, ignore_errors=True)
                if not os.path.isfile(os.path.join(target, "Diggs.xsd")):
                    raise
        _unpacked = target
        return os.path.join(target, "Diggs.xsd")


def _compiled_schema(schema_path: str):
    """The lxml ``XMLSchema`` for ``schema_path``, compiled once per process
    (DIGGS 2.6 takes a second or two). The parser never touches the network:
    every import is a local file."""
    with _lock:
        got = _compiled.get(schema_path)
    if got is not None:
        return got
    from lxml import etree
    parser = etree.XMLParser(no_network=True, resolve_entities=False)
    schema = etree.XMLSchema(etree.parse(schema_path, parser))
    with _lock:
        _compiled[schema_path] = schema
    return schema


def get_schema_path(version: str = "2.6") -> str:
    """
    Get the path to the DIGGS schema file.

    Args:
        version: Schema version ("2.6" or "2.5.a")

    Returns:
        Absolute path to the schema XSD file. 2.6 is the copy bundled with
        this package (no optional dependency); 2.5.a is pydiggs' copy.

    Raises:
        ImportError: If version is 2.5.a and pydiggs is not installed
        ValueError: If schema version is not supported
    """
    if version not in (BUNDLED_SCHEMA_VERSION, "2.5.a"):
        raise ValueError(f"Unsupported schema version: {version}. Use '2.6' or '2.5.a'")
    if version == BUNDLED_SCHEMA_VERSION:
        return bundled_schema_path()
    if not has_pydiggs():
        raise ImportError("pydiggs is not installed")

    import pydiggs

    pydiggs_dir = os.path.dirname(pydiggs.__file__)

    if version == "2.5.a":
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_diggs_schema(
    filepath: Optional[str] = None,
    content: Optional[str] = None,
    schema_version: str = "2.6"
) -> DiggValidationResult:
    """
    Validate DIGGS XML against XSD schema.

    Args:
        filepath: Path to DIGGS XML file (mutually exclusive with content)
        content: DIGGS XML as string (mutually exclusive with filepath)
        schema_version: Schema version to validate against ("2.6" or "2.5.a")

    Returns:
        DiggValidationResult with validation outcome

    Raises:
        ImportError: If schema_version is 2.5.a and pydiggs is not installed
            (2.6 is checked against the bundled schema with lxml)
        ValueError: If neither or both filepath and content are provided,
                   or if schema_version is invalid
    """
    # Validate inputs
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Only one of filepath or content should be provided")
    if schema_version not in ("2.6", "2.5.a"):
        raise ValueError(f"Invalid schema version: {schema_version}. Use '2.6' or '2.5.a'")

    if schema_version == BUNDLED_SCHEMA_VERSION:
        return _validate_bundled(filepath, content)

    if not has_pydiggs():
        raise ImportError(
            "pydiggs is not installed. Install with: pip install pydiggs"
        )

    from pydiggs import validator

    temp_file = None
    try:
        # Handle content input - write to temp file
        if content is not None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.xml',
                delete=False,
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            filepath = temp_file.name
            source = "content"
        else:
            source = os.path.basename(filepath)

        # Get schema path
        schema_path = get_schema_path(schema_version)

        # Create validator and run schema check
        v = validator(
            instance_path=filepath,
            schema_path=schema_path,
            output_log=False  # Don't write .log files
        )

        # Run schema validation
        v.schema_check()

        # Check for syntax errors
        if v.syntax_error_log is not None:
            errors = [str(v.syntax_error_log)]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=1,
                errors=errors
            )

        # Check for schema parse errors
        if v.schema_error_log is not None:
            errors = [str(v.schema_error_log)]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=1,
                errors=errors
            )

        # Check validation results
        if v.schema_validation_log is None:
            # None means valid
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=True,
                n_errors=0,
                errors=[]
            )
        else:
            # Has errors
            errors = [str(e) for e in v.schema_validation_log]
            return DiggValidationResult(
                source=source,
                check_type="schema",
                schema_version=schema_version,
                is_valid=False,
                n_errors=len(errors),
                errors=errors
            )

    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Best effort cleanup


def _validate_bundled(filepath: Optional[str],
                      content: Optional[str]) -> DiggValidationResult:
    """Check against the bundled DIGGS 2.6 schema with lxml: the same
    XSD and the same libxml2 check pydiggs runs, with no optional package
    and no network access. Errors are lxml's own log lines."""
    from lxml import etree

    source = "content" if content is not None else os.path.basename(filepath)
    schema = _compiled_schema(bundled_schema_path())
    parser = etree.XMLParser(no_network=True, resolve_entities=False,
                             huge_tree=True)
    try:
        if content is not None:
            data = content.encode("utf-8") if isinstance(content, str) else content
            doc = etree.fromstring(data, parser).getroottree()
        else:
            doc = etree.parse(filepath, parser)
    except etree.XMLSyntaxError as exc:
        return DiggValidationResult(
            source=source, check_type="schema",
            schema_version=BUNDLED_SCHEMA_VERSION, is_valid=False, n_errors=1,
            errors=[f"not well-formed XML: {exc}"])
    if schema.validate(doc):
        return DiggValidationResult(
            source=source, check_type="schema",
            schema_version=BUNDLED_SCHEMA_VERSION, is_valid=True, n_errors=0,
            errors=[])
    errors = [str(e) for e in schema.error_log]
    return DiggValidationResult(
        source=source, check_type="schema",
        schema_version=BUNDLED_SCHEMA_VERSION, is_valid=False,
        n_errors=len(errors), errors=errors)


def validate_diggs_dictionary(
    filepath: Optional[str] = None,
    content: Optional[str] = None
) -> DiggValidationResult:
    """
    Validate DIGGS propertyClass values against DIGGS dictionary.

    Args:
        filepath: Path to DIGGS XML file (mutually exclusive with content)
        content: DIGGS XML as string (mutually exclusive with filepath)

    Returns:
        DiggValidationResult with validation outcome

    Raises:
        ImportError: If pydiggs is not installed
        ValueError: If neither or both filepath and content are provided
    """
    if not has_pydiggs():
        raise ImportError(
            "pydiggs is not installed. Install with: pip install pydiggs"
        )

    # Validate inputs
    if filepath is None and content is None:
        raise ValueError("Either filepath or content must be provided")
    if filepath is not None and content is not None:
        raise ValueError("Only one of filepath or content should be provided")

    from pydiggs import validator

    temp_file = None
    try:
        # Handle content input - write to temp file
        if content is not None:
            temp_file = tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.xml',
                delete=False,
                encoding='utf-8'
            )
            temp_file.write(content)
            temp_file.close()
            filepath = temp_file.name
            source = "content"
        else:
            source = os.path.basename(filepath)

        # Get dictionary path
        dictionary_path = get_dictionary_path()

        # Create validator and run dictionary check
        v = validator(
            instance_path=filepath,
            dictionary_path=dictionary_path,
            output_log=False  # Don't write .log files
        )

        # Run dictionary validation
        v.dictionary_check()

        # Check validation results — attribute only exists when errors found
        if getattr(v, 'dictionary_validation_log', None) is None:
            # None means valid
            return DiggValidationResult(
                source=source,
                check_type="dictionary",
                is_valid=True,
                n_errors=0,
                errors=[]
            )
        else:
            # Has undefined properties
            errors = v.dictionary_validation_log
            return DiggValidationResult(
                source=source,
                check_type="dictionary",
                is_valid=False,
                n_errors=len(errors),
                errors=errors
            )

    finally:
        # Clean up temp file if created
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except Exception:
                pass  # Best effort cleanup
