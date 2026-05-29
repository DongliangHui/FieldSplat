from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings


TOKEN_PERMISSIONS = {
    "admin": {
        "project:create",
        "asset:upload",
        "workflow:start",
        "workflow:read",
        "artifact:read",
        "artifact:download",
        "version:read",
        "issue:write",
        "admin:operator",
    },
    "internal_console": {
        "project:create",
        "asset:upload",
        "workflow:start",
        "workflow:read",
        "artifact:read",
        "artifact:download",
        "version:read",
        "issue:write",
        "admin:operator",
    },
    "external_system": {
        "project:create",
        "asset:upload",
        "workflow:start",
        "workflow:read",
        "artifact:read",
        "artifact:download",
        "version:read",
    },
    "read_only_viewer": {
        "workflow:read",
        "artifact:read",
        "artifact:download",
        "version:read",
    },
}


@dataclass(frozen=True)
class Principal:
    token_type: str
    permissions: set[str]


bearer = HTTPBearer(auto_error=True)


def _token_map(settings: Settings) -> dict[str, Principal]:
    return {
        settings.admin_api_token: Principal("admin", TOKEN_PERMISSIONS["admin"]),
        settings.internal_console_api_token: Principal("internal_console", TOKEN_PERMISSIONS["internal_console"]),
        settings.external_system_api_token: Principal("external_system", TOKEN_PERMISSIONS["external_system"]),
        settings.read_only_viewer_api_token: Principal("read_only_viewer", TOKEN_PERMISSIONS["read_only_viewer"]),
    }


def principal_from_token(token: str, settings: Settings) -> Principal:
    principal = _token_map(settings).get(token)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")
    return principal


def ensure_permissions(principal: Principal, *required: str) -> Principal:
    missing = set(required) - principal.permissions
    if missing:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"message": "Insufficient token permissions", "missing": sorted(missing)},
        )
    return principal


def get_current_principal(
    credentials: HTTPAuthorizationCredentials = Security(bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Bearer token required")
    return principal_from_token(credentials.credentials, settings)


def require_permissions(*required: str):
    def dependency(principal: Principal = Depends(get_current_principal)) -> Principal:
        return ensure_permissions(principal, *required)

    return dependency
