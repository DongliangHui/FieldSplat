from __future__ import annotations

import json
from pathlib import Path

from conftest import TEST_ROOT

from app.config import get_settings
from app.modules.field_capture_assessment import run_assessment


def _png_header(path: Path, width: int, height: int, payload: bytes = b"field") -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + width.to_bytes(4, "big") + height.to_bytes(4, "big") + payload)


def test_field_capture_assessment_flags_insufficient_site_capture() -> None:
    input_dir = TEST_ROOT / "imports" / "assessment_sparse"
    output_dir = TEST_ROOT / "assessment_output_sparse"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index in range(3):
        _png_header(input_dir / f"rear_missing_{index:03d}.png", 900, 700, payload=bytes([index]))

    result = run_assessment(
        input_dir,
        scene_type="indoor_room",
        target_quality="forensic",
        output_dir=output_dir,
    )

    assert result.report["can_leave_site"] is False
    assert result.report["expected_quality"] in {"C", "D"}
    assert result.report["missing_views"]
    assert result.report["required_reshoot"]
    assert result.report["target_region_marking"]["status"] == "not_marked"
    assert "irrelevant_environment_ratio" in result.report
    assert "subject_coverage_score" in result.report
    assert (output_dir / "capture_assessment_report.json").exists()
    assert (output_dir / "selected_assets_manifest.json").exists()
    saved_report = json.loads((output_dir / "capture_assessment_report.json").read_text(encoding="utf-8"))
    assert saved_report["module"] == "Field Capture Assessment"


def test_field_capture_assessment_selects_usable_assets_for_sufficient_photo_set() -> None:
    input_dir = TEST_ROOT / "imports" / "assessment_dense"
    output_dir = TEST_ROOT / "assessment_output_dense"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index in range(28):
        _png_header(input_dir / f"frame_{index:03d}.png", 2200, 1600, payload=bytes([index]))

    result = run_assessment(
        input_dir,
        scene_type="indoor_room",
        target_quality="standard",
        output_dir=output_dir,
    )

    assert result.report["can_leave_site"] is True
    assert result.report["expected_quality"] in {"A", "B"}
    assert result.manifest["selected_asset_count"] == 28
    assert result.report["selected_assets_manifest"] == "selected_assets_manifest.json"
    assert result.report["background_risk_detection"]["risk_level"] in {"low", "medium", "high"}


def test_capture_assessment_api_runs_from_configured_import_root(client, auth_headers) -> None:
    input_dir = TEST_ROOT / "imports" / "assessment_api"
    output_dir = TEST_ROOT / "assessment_api_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        _png_header(input_dir / f"api_capture_{index:03d}.png", 1800, 1200, payload=bytes([index]))

    response = client.post(
        "/api/capture-assessment/run",
        headers=auth_headers,
        json={
            "input_path": str(input_dir),
            "scene_type": "indoor_room",
            "target_quality": "standard",
            "output_path": str(output_dir),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["asset_scan"]["total_assets"] == 12
    assert body["report"]["selected_assets_manifest"] == "selected_assets_manifest.json"
    assert Path(body["report_path"]).exists()
    assert Path(body["selected_assets_manifest_path"]).exists()


def test_capture_assessment_api_translates_configured_host_path_alias(client, auth_headers, monkeypatch) -> None:
    input_dir = TEST_ROOT / "imports" / "assessment_host_alias"
    output_dir = TEST_ROOT / "assessment_host_alias_output"
    input_dir.mkdir(parents=True, exist_ok=True)
    for index in range(12):
        _png_header(input_dir / f"host_alias_{index:03d}.png", 1800, 1200, payload=bytes([index]))

    monkeypatch.setenv("HOST_IMPORT_ROOT", r"F:\video2splat\samples")
    monkeypatch.setenv("HOST_IMPORT_CONTAINER_ROOT", (TEST_ROOT / "imports").as_posix())
    get_settings.cache_clear()
    try:
        response = client.post(
            "/api/capture-assessment/run",
            headers=auth_headers,
            json={
                "input_path": r"F:\video2splat\samples\assessment_host_alias",
                "scene_type": "indoor_room",
                "target_quality": "standard",
                "output_path": str(output_dir),
            },
        )
    finally:
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json()["report"]["asset_scan"]["total_assets"] == 12


def test_capture_assessment_import_roots_returns_clickable_examples(client, auth_headers) -> None:
    input_dir = TEST_ROOT / "imports" / "000_assessment_examples"
    input_dir.mkdir(parents=True, exist_ok=True)
    _png_header(input_dir / "example.png", 1600, 1200)

    response = client.get("/api/capture-assessment/import-roots", headers=auth_headers)

    assert response.status_code == 200
    roots = response.json()["roots"]
    assert roots
    assert any(example["path"].endswith("000_assessment_examples") for root in roots for example in root["examples"])


def test_capture_assessment_upload_run_hides_local_paths_from_user_flow(client, auth_headers) -> None:
    files = [
        ("files", (f"field/frame_{index:03d}.png", b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (1800).to_bytes(4, "big") + (1200).to_bytes(4, "big"), "image/png"))
        for index in range(12)
    ]
    response = client.post(
        "/api/capture-assessment/upload-run",
        headers=auth_headers,
        data={
            "scene_type": "indoor_room",
            "target_quality": "standard",
            "key_areas": '["door"]',
            "roi_annotations": '{"target_regions":[{"type":"polygon","label":"door","points":[[0,0],[1,0],[1,1]]}],"ignore_regions":[]}',
        },
        files=files,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["asset_scan"]["total_assets"] == 12
    assert body["report"]["roi_annotations"] == "roi_annotations.json"
