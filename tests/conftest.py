import os
import tempfile
from pathlib import Path

import pytest

# Point Hermès at a temp /data so tests are hermetic.
_TMP = Path(tempfile.mkdtemp(prefix="hermes-test-"))
os.environ.setdefault("HERMES_DATA_DIR", str(_TMP))
os.environ.setdefault("HERMES_STAGE", "stage_0_paper")
os.environ.setdefault("DATABASE_URL", f"sqlite+aiosqlite:///{_TMP}/test.db")
os.environ.setdefault("VECTOR_DIR", str(_TMP / "chroma"))
os.environ.setdefault("TRM_ENABLED", "false")
os.environ.setdefault("LLM_PRIMARY_BASE_URL", "http://127.0.0.1:0/v1")
os.environ.setdefault("LLM_PRIMARY_MODEL", "test")
os.environ.setdefault("HERMES_BANKROLL_USD", "100")


@pytest.fixture(scope="session")
def tmp_data_dir() -> Path:
    return _TMP
