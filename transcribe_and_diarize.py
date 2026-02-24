#!/usr/bin/env python3
"""
Legacy entry point -- delegates to main.py.

This file exists so that existing commands like:
    python transcribe_and_diarize.py video.mov
continue to work after the modular refactor.

All logic now lives in the ``diarize`` package.
See main.py for the canonical CLI entry point.
"""

from main import main

if __name__ == "__main__":
    main()
