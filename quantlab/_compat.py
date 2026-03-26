"""External dependency resolution for optional integrations.

Centralizes all sys.path manipulation for Kronos, RD-Agent, and qlib scripts.
Other quantlab modules should NEVER modify sys.path directly.

Environment variables:
    QUANTLAB_KRONOS_PATH  - Path to Kronos repo (default: auto-detect sibling dir)
    QUANTLAB_RDAGENT_PATH - Path to RD-Agent repo (default: auto-detect sibling dir)
    QUANTLAB_QLIB_DIR     - Path to qlib source repo (default: auto-detect sibling dir)
"""

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _monorepo_root() -> Optional[Path]:
    """Detect monorepo root by looking for quantlab/ as a sibling directory."""
    pkg_dir = Path(__file__).resolve().parent  # .../quantlab/
    candidate = pkg_dir.parent                  # .../Quant/
    if (candidate / "quantlab").is_dir():
        return candidate
    return None


def _ensure_on_path(path: Path, label: str) -> None:
    """Add *path* to sys.path if not already present."""
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)
        logger.debug("Added %s to sys.path: %s", label, s)


# ------------------------------------------------------------------
# Kronos
# ------------------------------------------------------------------

def get_kronos_path() -> Optional[Path]:
    """Resolve Kronos installation path."""
    env = os.environ.get("QUANTLAB_KRONOS_PATH")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    root = _monorepo_root()
    if root and (root / "Kronos").is_dir():
        return root / "Kronos"
    return None


def import_kronos_module(name: str = "model.kronos"):
    """Import a module from the Kronos project.

    Tries direct import first (works if Kronos is pip-installed),
    then falls back to path-based resolution.

    Returns the imported module.
    Raises ImportError if Kronos is not available.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    path = get_kronos_path()
    if path is None:
        raise ImportError(
            f"Cannot import '{name}': Kronos not found. "
            "Set QUANTLAB_KRONOS_PATH or place Kronos/ alongside quantlab/."
        )
    _ensure_on_path(path, "Kronos")
    return importlib.import_module(name)


# ------------------------------------------------------------------
# RD-Agent
# ------------------------------------------------------------------

def get_rdagent_path() -> Optional[Path]:
    """Resolve RD-Agent installation path."""
    env = os.environ.get("QUANTLAB_RDAGENT_PATH")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    root = _monorepo_root()
    if root and (root / "RD-Agent").is_dir():
        return root / "RD-Agent"
    return None


def import_rdagent_module(name: str):
    """Import a module from the RD-Agent project.

    Tries direct import first, then falls back to path-based resolution.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

    path = get_rdagent_path()
    if path is None:
        raise ImportError(
            f"Cannot import '{name}': RD-Agent not found. "
            "Set QUANTLAB_RDAGENT_PATH or place RD-Agent/ alongside quantlab/."
        )
    _ensure_on_path(path, "RD-Agent")
    return importlib.import_module(name)


# ------------------------------------------------------------------
# qlib scripts (dump_bin etc.)
# ------------------------------------------------------------------

def get_qlib_scripts_path() -> Optional[Path]:
    """Resolve path to qlib/scripts/ directory."""
    env = os.environ.get("QUANTLAB_QLIB_DIR")
    if env:
        p = Path(env).resolve() / "scripts"
        if p.is_dir():
            return p
    root = _monorepo_root()
    if root:
        p = root / "qlib" / "scripts"
        if p.is_dir():
            return p
    return None


def import_dump_bin():
    """Import ``DumpDataUpdate`` from qlib's scripts directory.

    This class is NOT part of qlib's pip package — it lives in the
    qlib source repo under ``scripts/dump_bin.py``.

    Returns the ``DumpDataUpdate`` class.
    """
    try:
        from dump_bin import DumpDataUpdate
        return DumpDataUpdate
    except ImportError:
        pass

    scripts_path = get_qlib_scripts_path()
    if scripts_path is None:
        raise ImportError(
            "Cannot import dump_bin: qlib source repo not found. "
            "Set QUANTLAB_QLIB_DIR to the qlib source directory."
        )
    _ensure_on_path(scripts_path, "qlib/scripts")
    from dump_bin import DumpDataUpdate
    return DumpDataUpdate
