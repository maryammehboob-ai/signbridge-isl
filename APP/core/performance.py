"""Read-only loader for the saved model evaluation (MODEL_EVALUATION/landmarks).

Nothing is computed or estimated here: every number comes from results.json, written by
MODEL_EVALUATION/landmarks/evaluate_landmarks.py. If the file is missing, the page says so.
"""
import json

from . import PROJECT_DIR

EVAL_DIR = PROJECT_DIR / "MODEL_EVALUATION" / "landmarks"
RESULTS = EVAL_DIR / "results.json"


def load_results():
    """The saved results as a dict, or None when the evaluation has not been run."""
    try:
        return json.loads(RESULTS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def chart(folder, name):
    """Path of a saved chart picture, or None if it is not there."""
    path = EVAL_DIR / folder / name
    return path if path.is_file() else None
