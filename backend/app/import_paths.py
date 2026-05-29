from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import HTTPException

from app.config import get_settings


MEDIA_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".osv", ".insv"}


def _normalize_for_prefix(value: str) -> str:
    normalized = value.strip().strip('"').replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return normalized.rstrip("/").lower()


def _join_container_path(container_root: str, relative: str) -> str:
    root = container_root.replace("\\", "/").rstrip("/") or "/host-imports"
    rel = relative.replace("\\", "/").strip("/")
    return f"{root}/{rel}" if rel else root


def translate_host_import_path(path_value: str) -> str:
    """Translate a configured host path alias into the container import path.

    The API runs inside Docker, so a Windows path such as
    F:\\video2splat\\samples\\ai_sample\\pic is not directly readable there.
    Compose mounts HOST_IMPORT_ROOT to HOST_IMPORT_CONTAINER_ROOT; this helper
    maps the user-facing path to the readable container path before validation.
    """

    raw = path_value.strip().strip('"')
    if not raw:
        return raw
    settings = get_settings()
    host_root = settings.host_import_root.strip()
    container_root = settings.host_import_container_root.strip() or "/host-imports"
    if not host_root:
        return raw

    normalized_raw = _normalize_for_prefix(raw)
    normalized_host = _normalize_for_prefix(host_root)
    if normalized_raw == normalized_host:
        return container_root
    if normalized_raw.startswith(f"{normalized_host}/"):
        relative = raw.replace("\\", "/")[len(host_root.replace("\\", "/").rstrip("/")) :].strip("/")
        return _join_container_path(container_root, relative)
    return raw


def resolve_configured_import_path(path_value: str, *, noun: str = "Input path") -> Path:
    translated = translate_host_import_path(path_value)
    source = Path(translated).expanduser().resolve()
    if not source.exists():
        detail: Any = f"{noun} does not exist: {path_value}"
        if translated != path_value:
            detail = {"message": f"{noun} does not exist after host path mapping", "input_path": path_value, "mapped_path": translated}
        raise HTTPException(status_code=404, detail=detail)
    roots = get_settings().import_roots
    if roots and not any(source == root or root in source.parents for root in roots):
        raise HTTPException(
            status_code=403,
            detail={
                "message": f"{noun} is outside configured import roots",
                "input_path": path_value,
                "mapped_path": translated,
                "configured_roots": [str(root) for root in roots],
            },
        )
    return source


def describe_import_roots() -> dict[str, Any]:
    settings = get_settings()
    host_root = settings.host_import_root.strip() or None
    container_root = settings.host_import_container_root.strip() or "/host-imports"
    roots = []
    for root in settings.import_roots:
        examples: list[dict[str, str]] = []
        if root.exists():
            examples.extend(_discover_import_examples(root, limit=24))
        roots.append(
            {
                "container_path": str(root),
                "host_path": host_root if str(root) == str(Path(container_root).resolve()) else None,
                "examples": examples,
            }
        )
    return {"container_root": container_root, "host_root": host_root, "roots": roots}


def _discover_import_examples(root: Path, *, limit: int) -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    def add(path: Path, label: str) -> None:
        if len(examples) >= limit:
            return
        examples.append({"path": str(path), "label": label})

    add(root, "导入根目录")
    immediate = sorted(root.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    for item in immediate:
        if len(examples) >= limit:
            break
        if item.is_dir():
            add(item, f"目录：{item.name}")
            for child in sorted(item.iterdir(), key=lambda child: (not child.is_dir(), child.name.lower())):
                if len(examples) >= limit:
                    break
                if child.is_dir():
                    add(child, f"目录：{item.name}/{child.name}")
                elif child.suffix.lower() in MEDIA_EXTENSIONS:
                    add(child, f"文件：{item.name}/{child.name}")
        elif item.suffix.lower() in MEDIA_EXTENSIONS:
            add(item, f"文件：{item.name}")
    return examples
