from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.config import Settings, get_settings
from app.models import Asset
from app.operators.base import CommandResult


CACHE_SCHEMA_VERSION = "stage-cache-v1"


@dataclass(frozen=True)
class StageCacheEntry:
    stage_name: str
    cache_key: str
    cache_dir: Path
    hit: bool

    @property
    def metadata_path(self) -> Path:
        return self.cache_dir / "cache_metadata.json"

    def summary(self) -> dict[str, Any]:
        return {
            "cache_hit": self.hit,
            "cache_key": self.cache_key,
            "cache_dir": str(self.cache_dir),
        }


class StageCache:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.root = Path(self.settings.workspace_root) / "cache" / "stages"

    def entry(
        self,
        stage_name: str,
        *,
        inputs: Iterable[Any],
        stage_config: dict[str, Any] | None = None,
        algorithm_version: str = "v1",
    ) -> StageCacheEntry:
        cache_key = build_stage_cache_key(
            inputs=inputs,
            stage_name=stage_name,
            stage_config=stage_config or {},
            algorithm_version=algorithm_version,
        )
        cache_dir = self.root / _safe_stage_name(stage_name) / cache_key
        return StageCacheEntry(stage_name=stage_name, cache_key=cache_key, cache_dir=cache_dir, hit=cache_dir.exists())

    def restore(self, entry: StageCacheEntry, target_dir: Path) -> bool:
        if not entry.cache_dir.exists():
            return False
        target_dir.mkdir(parents=True, exist_ok=True)
        _copy_tree(entry.cache_dir, target_dir, exclude_names={"cache_metadata.json"})
        return True

    def save(
        self,
        entry: StageCacheEntry,
        source_dir: Path,
        *,
        metadata: dict[str, Any] | None = None,
        exclude_names: set[str] | None = None,
    ) -> None:
        if not source_dir.exists():
            return
        entry.cache_dir.mkdir(parents=True, exist_ok=True)
        excluded = {"cache_metadata.json"}
        if exclude_names:
            excluded.update(exclude_names)
        _copy_tree(source_dir, entry.cache_dir, exclude_names=excluded)
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "stage_name": entry.stage_name,
            "cache_key": entry.cache_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        entry.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_stage_cache_key(
    *,
    inputs: Iterable[Any],
    stage_name: str,
    stage_config: dict[str, Any],
    algorithm_version: str,
) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "stage_name": stage_name,
        "stage_config": _jsonable(stage_config),
        "algorithm_version": algorithm_version,
        "inputs": [_fingerprint_input(item) for item in inputs],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def cache_hit_command(operator_name: str, stage_key: str, cache_key: str, cwd: Path) -> CommandResult:
    now = datetime.now(timezone.utc)
    return CommandResult(
        operator_name=operator_name,
        stage_key=stage_key,
        command=["cache-hit", cache_key],
        cwd=str(cwd),
        stdout=f"stage cache hit: {cache_key}",
        stderr="",
        exit_code=0,
        started_at=now,
        finished_at=now,
    )


def _fingerprint_input(item: Any) -> dict[str, Any]:
    if isinstance(item, Path):
        return _fingerprint_path(item)
    if isinstance(item, Asset):
        return _fingerprint_asset(item)
    if isinstance(item, str):
        path = Path(item)
        if path.exists():
            return _fingerprint_path(path)
        return {"type": "string", "value": item}
    if isinstance(item, dict):
        return {"type": "dict", "value": _jsonable(item)}
    return {"type": type(item).__name__, "value": repr(item)}


def _fingerprint_asset(asset: Asset) -> dict[str, Any]:
    updated_at = getattr(asset, "updated_at", None)
    created_at = getattr(asset, "created_at", None)
    return {
        "type": "asset",
        "id": asset.id,
        "storage_uri": asset.storage_uri,
        "filename": asset.filename,
        "original_filename": asset.original_filename,
        "asset_type": asset.asset_type,
        "role": asset.role,
        "size_bytes": asset.size_bytes,
        "mtime": updated_at.isoformat() if updated_at else created_at.isoformat() if created_at else None,
    }


def _fingerprint_path(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        return {"type": "path", "path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "type": "path",
        "path": str(resolved),
        "name": resolved.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _safe_stage_name(stage_name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stage_name)


def _copy_tree(source: Path, target: Path, *, exclude_names: set[str]) -> None:
    for child in source.iterdir():
        if child.name in exclude_names:
            continue
        destination = target / child.name
        if child.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(child, destination)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)
