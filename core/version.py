"""
core/version.py — single source of truth for the Operon version.

Keep this in sync with pyproject.toml. Everything else (banner, --version,
updater) imports __version__ from here.
"""

__version__ = "3.1.0"
