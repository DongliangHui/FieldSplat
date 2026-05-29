from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    api_v1_prefix: str = "/api/v1"
    public_base_url: str = "http://localhost:8000"

    database_url: str = "sqlite:///./reconstruction.db"
    redis_url: str = "redis://localhost:6379/0"

    storage_backend: str = "local"
    storage_local_root: str = "./data/storage"
    s3_endpoint: str = "http://localhost:9000"
    s3_access_key: str = "minio"
    s3_secret_key: str = "minio123456"
    s3_bucket: str = "reconstruction"
    s3_secure: bool = False

    engine_config_path: str = "./configs/engine.yaml"
    workspace_root: str = "./data/workspace"
    keep_failed_workspace: bool = True
    keep_passed_workspace: bool = False

    admin_api_token: str = "dev-admin-token"
    internal_console_api_token: str = "dev-console-token"
    external_system_api_token: str = "dev-external-token"
    read_only_viewer_api_token: str = "dev-readonly-token"

    celery_task_always_eager: bool = False
    webhook_timeout_seconds: int = 10
    asset_import_roots: str = "/host-imports"
    host_import_root: str = ""
    host_import_container_root: str = "/host-imports"
    colmap_fake_runner: bool = False
    nerfstudio_fake_runner: bool = False
    nerfstudio_default_method: str = "splatfacto-big"
    workflow_default_mode: str = "standard"
    nerfstudio_quick_iterations: int = 2000
    nerfstudio_standard_iterations: int = 10000
    nerfstudio_high_iterations: int = 30000
    nerfstudio_smoke_iterations: int = 20
    nerfstudio_video_frame_target: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def engine_config(self) -> dict[str, Any]:
        path = Path(self.engine_config_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        return loaded

    @property
    def import_roots(self) -> list[Path]:
        return [Path(item.strip()).resolve() for item in self.asset_import_roots.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
