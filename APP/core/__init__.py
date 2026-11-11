"""Shared code for the ISL Streamlit app. Sets up the import paths to your existing modules.

Nothing in your existing folders is copied or changed: the app imports from them.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = APP_DIR.parent

for folder in ("SCRIPTS", "SIGN_TO_TEXT", "TEXT_TO_SIGN", "SPEECH_TO_TEXT"):
    path = str(PROJECT_DIR / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
