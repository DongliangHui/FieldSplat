# FieldSplat Reconstruction Engine

Dockerized API engine for first-scene digital reconstruction. The system boundary is the `reconstruction-api`; the internal Console is only an API client.

## Architecture

- `reconstruction-api`: FastAPI REST/WebSocket/Webhook boundary.
- `reconstruction-worker-cpu`: Celery worker for registry, preprocess, QC, export queues.
- `reconstruction-worker-colmap`: Celery worker for COLMAP queues.
- `reconstruction-worker-nerfstudio-gpu`: optional GPU worker override for Nerfstudio Splatfacto-big training, single concurrency.
- `reconstruction-worker-gpu`: optional GPU worker override for InstantSplat++ / Gaussian queues.
- `postgres`: metadata store.
- `redis`: queue and worker coordination.
- `minio`: asset and artifact object store.
- `reconstruction-console`: internal React Console. It does not read local files, MinIO buckets, workers, or scripts directly.

## Start

```powershell
docker compose up --build
```

GPU worker:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Default development tokens are in `.env.example`. Replace them before exposing the API outside a local development network.

For the provided local samples, copy `.env.example` to `.env` and set:

```text
HOST_IMPORT_ROOT=F:\video2splat\samples
VITE_SAMPLE_PHOTO_PATH=/host-imports/ai_sample/pic
VITE_SAMPLE_VIDEO_PATH=/host-imports/ai_sample/video/south-building-good.mp4
```

The browser sends `/host-imports/...` to the API. The API container imports only from the configured whitelist and copies files into Storage Service before creating Asset Registry rows.

## API Boundary

All external calls use:

```text
Authorization: Bearer <token>
```

Core API prefix:

```text
/api/v1
```

Minimum loop:

1. `POST /api/v1/projects`
2. `POST /api/v1/projects/{project_id}/assets/upload`
3. `POST /api/v1/projects/{project_id}/workflows`
4. `GET /api/v1/workflows/{workflow_id}`
5. `GET /api/v1/workflows/{workflow_id}/logs`
6. `GET /api/v1/workflows/{workflow_id}/artifacts`
7. `GET /api/v1/artifacts/{artifact_id}/download`
8. `GET /api/v1/projects/{project_id}/current-version`

## Quality Gate

Every workflow emits `quality_report.json`, `run_summary.json`, and `artifacts.json`. The gate blocks command failures, empty artifacts, missing `transforms.json`, zero registered frames, camera mapping errors, and D-grade outputs. D-grade / hard-fail results cannot create current versions or formal viewer exports.

`backend/app/operators/qc/camera_consistency.py` implements the hard gate for InstantSplat++ camera mapping:

- input image count must match camera count;
- input image count must match unique `img_name` count;
- missing crop ids are blocked;
- duplicated `img_name` values are blocked;
- pano crops must preserve `shared_center_group`;
- D-grade / hard-fail outputs cannot create current versions.

The documented failure case, 16 camera records with only 4 unique `img_name` values, is covered by tests.

## Operator Boundary

Algorithm code is wrapped as Operators. The API service only creates metadata records and queues Celery tasks. Long-running CPU/GPU work is executed by workers using registry inputs and workspace-local files, then output is registered back into the Artifact Registry.

The default real training path is `nerfstudio_3dgs_train`, using Nerfstudio `splatfacto-big`, `ns-process-data`, `ns-train`, and `ns-export gaussian-splat`. COLMAP, InstantSplat++, Gaussian/2DGS wrappers are exposed as Operators and report availability through `/api/v1/health/operators` based on mounted binaries/repos/models.

The internal Console covers:

- `/projects`
- `/projects/:projectId`
- `/projects/:projectId/assets`
- `/projects/:projectId/workflows`
- `/workflows/:workflowId/monitor`
- `/versions/:versionId/viewer`
- `/projects/:projectId/issues`
- `/diagnostics/:workflowId`
- `/admin/engine`

## Validation

```powershell
python -m pytest

cd frontend
npm install
npm run build

cd ..
docker compose -f docker-compose.yml -f docker-compose.gpu.yml config --quiet
```
