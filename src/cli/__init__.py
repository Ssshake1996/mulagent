"""CLI package for mul-agent — TUI and headless REPL interfaces."""

import sys
from pathlib import Path


def ensure_src_path() -> None:
    """Add src/ to sys.path so internal imports resolve correctly.

    Only needed when running directly (``python src/cli/main.py``).
    After ``pip install -e .`` this is a no-op because setuptools
    already puts ``src/`` on the path.
    """
    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
