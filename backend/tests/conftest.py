from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

TEST_ROOT = Path(tempfile.mkdtemp(prefix="reconstruction-tests-"))
BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

os.environ.setdefault("APP_ENV", "test")
os.environ["DATABASE_URL"] = f"sqlite:///{(TEST_ROOT / 'test.db').as_posix()}"
os.environ["STORAGE_BACKEND"] = "local"
os.environ["STORAGE_LOCAL_ROOT"] = (TEST_ROOT / "storage").as_posix()
os.environ["WORKSPACE_ROOT"] = (TEST_ROOT / "workspace").as_posix()
os.environ["ASSET_IMPORT_ROOTS"] = (TEST_ROOT / "imports").as_posix()
os.environ["ENGINE_CONFIG_PATH"] = str(REPO_ROOT / "configs" / "engine.yaml")
os.environ["CELERY_TASK_ALWAYS_EAGER"] = "true"
os.environ["COLMAP_FAKE_RUNNER"] = "true"
os.environ["NERFSTUDIO_FAKE_RUNNER"] = "true"
os.environ["ADMIN_API_TOKEN"] = "test-admin-token"
os.environ["INTERNAL_CONSOLE_API_TOKEN"] = "test-console-token"
os.environ["EXTERNAL_SYSTEM_API_TOKEN"] = "test-external-token"
os.environ["READ_ONLY_VIEWER_API_TOKEN"] = "test-readonly-token"

import pytest
from fastapi.testclient import TestClient

from app.database import engine, init_db
from app.main import app
from app.models import Base


@pytest.fixture(autouse=True)
def clean_database():
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-admin-token"}
