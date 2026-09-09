"""Workspace-local test paths for restricted Windows environments."""

from pathlib import Path
import uuid

import pytest


@pytest.fixture
def tmp_path() -> Path:
    """Avoid OS temp ACL failures by isolating ignored fixtures under project artifacts."""
    path = Path("artifacts") / "test_scratch" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()
