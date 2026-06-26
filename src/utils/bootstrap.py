"""Runtime bootstrap helpers shared by every entry point.

Entry points (the scripts in ``scripts/`` and the Colab notebook) call
:func:`setup` once at start-up to:

1. ensure the repository root is importable (so ``import config`` / ``import
   src`` work regardless of the current working directory),
2. route the HuggingFace caches onto the (Drive-backed) project cache,
3. create the base artefact directories, and
4. surface any environment warnings (e.g. Drive not mounted on Colab).

Because importing this module already requires ``src`` to be on ``sys.path``,
scripts use a two-line shim *before* importing it::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.utils.bootstrap import setup
"""

from __future__ import annotations

import sys
from pathlib import Path

# ``src/utils/bootstrap.py`` -> parents[2] is the repository root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def ensure_project_on_path() -> Path:
    """Insert the repository root at the front of ``sys.path`` (idempotent)."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return PROJECT_ROOT


def setup(*, ensure_dirs: bool = True, configure_cache: bool = True):
    """Initialise the runtime and return the resolved :class:`ProjectPaths`.

    Parameters
    ----------
    ensure_dirs:
        When ``True`` (default) create all base artefact directories.
    configure_cache:
        When ``True`` (default) point the HuggingFace caches at the project
        cache directory before any model/dataset download happens.
    """
    ensure_project_on_path()

    # Imported here (not at module top) so the path shim above is guaranteed to
    # have run for callers that import this module by file path.
    from config.paths import configure_hf_cache, get_paths
    from src.utils.logging_utils import get_logger

    paths = get_paths()
    if ensure_dirs:
        paths.ensure()
    if configure_cache:
        configure_hf_cache(paths)

    logger = get_logger("finai.bootstrap", log_dir=paths.logs_dir)
    logger.debug("FINAI_HOME resolved to %s", paths.home)
    for warning in paths.warnings():
        logger.warning(warning)

    return paths


__all__ = ["PROJECT_ROOT", "ensure_project_on_path", "setup"]
