#!/usr/bin/env python
"""WaveFlow release CLI: python reproduce.py <command> [options].

This is a thin wrapper around reproduce/reproduce.py (same as running
`python reproduce/reproduce.py`); it keeps commands runnable from the
repository root.
"""
import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "reproduce" / "reproduce.py"
    runpy.run_path(str(target), run_name="__main__")
