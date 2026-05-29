from __future__ import annotations

from fastapi import APIRouter

from app.api import artifacts, assets, capture_assessment, exports, groups, health, issues, optimized_reconstruction, projects, versions, workflows

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(projects.router)
api_router.include_router(assets.router)
api_router.include_router(capture_assessment.router)
api_router.include_router(workflows.router)
api_router.include_router(optimized_reconstruction.router)
api_router.include_router(artifacts.router)
api_router.include_router(versions.router)
api_router.include_router(groups.router)
api_router.include_router(issues.router)
api_router.include_router(exports.router)
