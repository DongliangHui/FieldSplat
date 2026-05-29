from __future__ import annotations

from typing import Any

from app.config import Settings, get_settings


BASELINE_KEY = "fieldsplat_defaults_v0_1"


def fieldsplat_defaults(settings: Settings | None = None) -> dict[str, Any]:
    config = (settings or get_settings()).engine_config
    defaults = config.get(BASELINE_KEY)
    return defaults if isinstance(defaults, dict) else {}


def default_at(path: str, fallback: Any = None, *, settings: Settings | None = None) -> Any:
    current: Any = fieldsplat_defaults(settings)
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return fallback
        current = current[part]
    return current


def default_int(path: str, fallback: int, *, settings: Settings | None = None) -> int:
    value = default_at(path, fallback, settings=settings)
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def default_float(path: str, fallback: float, *, settings: Settings | None = None) -> float:
    value = default_at(path, fallback, settings=settings)
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def default_bool(path: str, fallback: bool, *, settings: Settings | None = None) -> bool:
    value = default_at(path, fallback, settings=settings)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return fallback
