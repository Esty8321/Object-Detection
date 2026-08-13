"""Compatibility shim for the requested 'single examiner' entrypoint.

Use pdc_examiner.py for the real implementation. This file exists so a rushed tester can still find it.
"""
from pdc_examiner import main

if __name__ == "__main__":
    main()
