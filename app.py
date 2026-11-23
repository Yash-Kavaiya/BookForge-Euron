"""Vercel / uvicorn entrypoint: ``uvicorn app:app``.

Vercel auto-detects the FastAPI instance named ``app`` in this file.
"""

from bookforge.web.app import app

__all__ = ["app"]
