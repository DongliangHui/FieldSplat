from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.services.resource_locks import resource_lock


def test_resource_lock_reclaims_stale_pid_file(tmp_path: Path) -> None:
    settings = Settings(workspace_root=str(tmp_path / "workspace"))
    lock_dir = tmp_path / "workspace" / "locks"
    lock_dir.mkdir(parents=True)
    lock_path = lock_dir / "gpu-heavy.lock"
    lock_path.write_text("pid=999999\nresource=gpu-heavy\n", encoding="utf-8")

    with resource_lock("gpu-heavy", settings=settings, timeout_seconds=1, poll_seconds=0.01) as acquired:
        assert acquired == lock_path
        assert "pid=" in lock_path.read_text(encoding="utf-8")

    assert not lock_path.exists() or lock_path.read_text(encoding="utf-8")
