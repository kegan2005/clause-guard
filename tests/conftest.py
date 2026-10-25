"""Pytest setup: make the project root importable and isolate model folders."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    """Point ClauseGuard at an empty models folder so no trained model is found."""
    from ml import paths

    monkeypatch.setattr(paths, "MODELS_DIR", tmp_path / "models")
    return tmp_path / "models"
