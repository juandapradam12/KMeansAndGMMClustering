#!/usr/bin/env python3
"""CLI shim — prefer ``cluster-bench`` after ``pip install -e .``."""

from __future__ import annotations

from clustering.cli import main

if __name__ == "__main__":
    main()
