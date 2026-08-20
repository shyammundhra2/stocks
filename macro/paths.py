"""
Central filesystem paths for the repo.

Model artifacts (.joblib) live in a single ``models/`` directory at the repo
root. Resolving them from this file's location (not the current working
directory) means loads/saves work no matter where a script is launched from -
``python app.py``, ``python training/train_trends.py``, or a backtest under
``backtest/``.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")


def model_path(name: str) -> str:
    """Absolute path to a model artifact ``name`` in the repo ``models/`` dir."""
    return os.path.join(MODELS_DIR, name)
