from __future__ import annotations

import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.config import Settings, get_settings

try:  # pragma: no cover - exercised in Linux worker containers.
    import fcntl
except ImportError:  # pragma: no cover - exercised by Windows unit tests.
    fcntl = None  # type: ignore[assignment]


@contextmanager
def resource_lock(
    name: str,
    *,
    settings: Settings | None = None,
    timeout_seconds: int | None = None,
    poll_seconds: float = 2.0,
    stale_seconds: int | None = 6 * 60 * 60,
) -> Iterator[Path]:
    settings = settings or get_settings()
    lock_dir = Path(settings.workspace_root) / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{_safe_name(name)}.lock"
    if fcntl is not None:
        with _flock_resource(lock_path, name, timeout_seconds=timeout_seconds, poll_seconds=poll_seconds) as acquired:
            yield acquired
        return

    started = time.monotonic()
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, _lock_metadata(name).encode("utf-8"))
            break
        except FileExistsError:
            if _stale_lock(lock_path, stale_seconds=stale_seconds):
                try:
                    lock_path.unlink()
                    continue
                except FileNotFoundError:
                    continue
            if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                raise TimeoutError(f"Timed out waiting for resource lock: {name}")
            time.sleep(poll_seconds)
    try:
        yield lock_path
    finally:
        if fd is not None:
            os.close(fd)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


@contextmanager
def _flock_resource(lock_path: Path, name: str, *, timeout_seconds: int | None, poll_seconds: float) -> Iterator[Path]:
    started = time.monotonic()
    with lock_path.open("a+", encoding="utf-8") as fh:
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)  # type: ignore[union-attr]
                break
            except BlockingIOError:
                if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                    raise TimeoutError(f"Timed out waiting for resource lock: {name}")
                time.sleep(poll_seconds)
        fh.seek(0)
        fh.truncate()
        fh.write(_lock_metadata(name))
        fh.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]


def _lock_metadata(name: str) -> str:
    return f"pid={os.getpid()}\nhost={socket.gethostname()}\nresource={name}\ncreated_at={int(time.time())}\n"


def _stale_lock(lock_path: Path, *, stale_seconds: int | None) -> bool:
    try:
        contents = lock_path.read_text(encoding="utf-8", errors="ignore")
        age_seconds = time.time() - lock_path.stat().st_mtime
    except FileNotFoundError:
        return False
    pid = _metadata_pid(contents)
    if pid is not None and _pid_exists(pid):
        return False
    if pid is not None:
        return True
    return stale_seconds is not None and age_seconds > stale_seconds


def _metadata_pid(contents: str) -> int | None:
    for line in contents.splitlines():
        key, _, value = line.partition("=")
        if key.strip() != "pid":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
