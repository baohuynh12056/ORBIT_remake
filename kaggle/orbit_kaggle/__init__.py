"""Resumable, local-only ToneBank experiments. No model is loaded at import."""
__version__ = "0.1.0"


def enable_vendor():
    import sys
    from pathlib import Path
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    for path in (vendor, vendor / "ksteering"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


enable_vendor()
