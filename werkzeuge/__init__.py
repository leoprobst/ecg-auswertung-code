"""Compatibility package name of the original working tree.

The analysis scripts import their helpers as `werkzeuge.<folder>.<module>`. In this repository the folders
have descriptive names; importing this package installs the import hook of `repo_config` that maps the
original names to the renamed folders (import_map.json). Nothing else lives here.
"""
import repo_config  # noqa: F401  (installs the hooks)
