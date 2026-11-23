"""BookForge web app: a lean, Vercel-ready FastAPI front end on the Euron EURI API.

This package deliberately avoids importing Google ADK, yt-dlp or Whisper so the
serverless bundle stays small. The full YouTube-channel pipeline lives in
``bookforge.agents`` and runs locally (``pip install ".[pipeline]"``).
"""
