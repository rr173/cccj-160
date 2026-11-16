#!/usr/bin/env python3
"""Isolated (-I) launcher that puts the local src/ tree on sys.path.

Python's -I flag ignores PYTHONPATH (a deliberate safety property in this
project, so a planted json.py next to a data file can't hijack imports). This
wrapper keeps that isolation, makes the local packages importable, and then
calls the target module's ``main(argv)`` entry point directly. This avoids the
double-execution semantics of ``runpy.run_module`` while preserving relative
imports.
"""
import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, os.pardir, "src"))
sys.path.insert(0, SRC)

if len(sys.argv) < 2:
    raise SystemExit("usage: python -I scripts/run_module.py <dotted.module> [args...]")

module_name = sys.argv[1]
argv = sys.argv[2:]
module = importlib.import_module(module_name)
if not hasattr(module, "main"):
    raise SystemExit(f"module {module_name} exposes no main(argv) entry point")
module.main(argv)
