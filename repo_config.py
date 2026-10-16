"""Shared setup for the analysis scripts of this repository.

1. Paths. The scripts never contain machine-specific paths. Every input or output location is looked up
   by a key from a TOML file with a [paths] table (see config.example.toml).
   Lookup order for the configuration file:
   a. the file named in the environment variable ECG_CODE_CONFIG,
   b. config.toml in the current working directory,
   c. config.toml next to this module.
2. Reference values. Some scripts compare results with expected values that are not published in the
   manuscripts. These values are not part of the repository; the scripts read them from a local file
   (key "reference_values" in [paths], default reference_values.json in the working directory or next
   to this module). When the file or a value is missing, the check is skipped with a note.
3. Imports. The scripts were written in a working tree with different folder names. Importing this module
   puts the repository root and the measurement package folder on sys.path and installs two import hooks:
   `werkzeuge.<original folder>.<module>` resolves to the renamed repository folders (see import_map.json),
   and a bare `import <module>` that is not found elsewhere resolves to the file of that name in the
   repository (same folder first).
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    tomllib = None

_ROOT = Path(__file__).resolve().parent
_CACHE = None
_REF_CACHE = None
_MAP = None
_INDEX = None
_MISSING = object()


# --------------------------------------------------------------------------- paths
def _find_config():
    env = os.environ.get("ECG_CODE_CONFIG")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / "config.toml")
    candidates.append(_ROOT / "config.toml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cfg = _find_config()
    if cfg is None or tomllib is None:
        _CACHE = {}
        return _CACHE
    with cfg.open("rb") as fh:
        _CACHE = tomllib.load(fh).get("paths", {})
    return _CACHE


def data_path(key):
    """Return the configured path for `key` as a pathlib.Path.

    Exits with a readable message when the key is not configured.
    """
    paths = _load()
    if key not in paths:
        sys.exit(
            "Path key '%s' is not set. Copy config.example.toml to config.toml, "
            "fill in the [paths] table and set ECG_CODE_CONFIG if the file is elsewhere." % key
        )
    return Path(paths[key])


def optional_path(key, default=None):
    """Like data_path, but returns `default` when the key is missing."""
    paths = _load()
    return Path(paths[key]) if key in paths else default


# --------------------------------------------------------------------------- reference values
def _load_reference():
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    p = optional_path("reference_values")
    candidates = [p] if p else []
    candidates += [Path.cwd() / "reference_values.json", _ROOT / "reference_values.json"]
    _REF_CACHE = {}
    for c in candidates:
        if c and c.is_file():
            with c.open(encoding="utf-8") as fh:
                _REF_CACHE = json.load(fh)
            break
    return _REF_CACHE


def reference(name, default=None):
    """Value stored under `name` in the local reference file, or `default` when there is none."""
    return _load_reference().get(name, default)


def check_reference(name, actual, tolerance=0.0, what=None):
    """Compare `actual` with the local reference value `name`.

    Returns True when equal (numbers: within `tolerance`), None when no reference value is stored
    (a note is printed and the check is skipped), and raises AssertionError on a mismatch.
    """
    ref = _load_reference().get(name, _MISSING)
    if ref is _MISSING:
        print("NOTE: no local reference value '%s'; check skipped." % name)
        return None
    if isinstance(ref, (int, float)) and isinstance(actual, (int, float)) and not isinstance(ref, bool):
        ok = abs(actual - ref) <= tolerance
    else:
        ok = actual == ref
    if not ok:
        raise AssertionError("%s: %r differs from the local reference value %r" % (what or name, actual, ref))
    return True


# --------------------------------------------------------------------------- imports
def _import_map():
    global _MAP
    if _MAP is None:
        p = _ROOT / "import_map.json"
        _MAP = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    return _MAP


class _OriginalLayoutFinder(importlib.abc.MetaPathFinder):
    """Resolve werkzeuge.<original folder>.<module> and audit_tools.<module> to the renamed repository folders."""

    def find_spec(self, fullname, path=None, target=None):
        top, _, rest = fullname.partition(".")
        if top == "werkzeuge":
            if not rest:
                return None
            folder = rest.replace(".", "/")
        elif top == "audit_tools":
            folder = ("audit_tools/" + rest.replace(".", "/")).rstrip("/")
        else:
            return None
        mapping = _import_map()
        if folder in mapping:
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = []
            return spec
        parent, _, leaf = folder.rpartition("/")
        if top == "audit_tools" and not parent:
            parent = "audit_tools"
        for d in mapping.get(parent, []):
            f = _ROOT / d / (leaf + ".py")
            if f.is_file():
                return importlib.util.spec_from_file_location(fullname, f)
        return None


def _script_index():
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        for top in sorted(_ROOT.iterdir()):
            if top.is_dir() and not top.name.startswith((".", "looplens_snapshot", "__")):
                for f in top.rglob("*.py"):
                    if "__pycache__" not in f.parts:
                        _INDEX.setdefault(f.stem, []).append(f)
    return _INDEX


def _caller_dir():
    f = sys._getframe(2)
    while f is not None and ("importlib" in f.f_code.co_filename or f.f_code.co_filename.startswith("<frozen")):
        f = f.f_back
    if f is None:
        return None
    return Path(f.f_globals.get("__file__") or f.f_code.co_filename).resolve().parent


class _SiblingFinder(importlib.abc.MetaPathFinder):
    """Fallback for bare `import module` of a script that lives elsewhere in the repository."""

    def find_spec(self, fullname, path=None, target=None):
        if "." in fullname:
            return None
        cands = _script_index().get(fullname)
        if not cands:
            return None
        here = _caller_dir()
        pick = [c for c in cands if here is not None and c.parent == here]
        if not pick and here is not None:
            work = next((p for p in here.parents if p.parent == _ROOT), None)
            pick = [c for c in cands if work is not None and work in c.parents]
        return importlib.util.spec_from_file_location(fullname, (pick or cands)[0])


def _install():
    root, snap = str(_ROOT), str(_ROOT / "looplens_snapshot_2026-10-07")
    if root not in sys.path:
        sys.path.insert(0, root)
    if snap not in sys.path:
        sys.path.append(snap)
    if not any(isinstance(x, _OriginalLayoutFinder) for x in sys.meta_path):
        sys.meta_path.insert(0, _OriginalLayoutFinder())
    if not any(isinstance(x, _SiblingFinder) for x in sys.meta_path):
        sys.meta_path.append(_SiblingFinder())


_install()
