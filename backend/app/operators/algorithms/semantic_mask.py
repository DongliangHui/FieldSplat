from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import zlib
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> int:
    args = _parse_args()
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images:
        images = images[: max(1, int(args.max_images))]

    missing = _missing_required(args)
    if missing:
        payload = _unavailable_payload(args, images, missing)
        _write_outputs(args, payload)
        print(f"semantic mask unavailable: missing {missing}", file=sys.stderr)
        return 2

    try:
        result = _run_groundingdino(args, images, masks_dir)
    except Exception as exc:
        payload = _failure_payload(args, images, exc)
        _write_outputs(args, payload)
        print(f"semantic mask failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _write_outputs(args, result)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FieldSplat semantic mask manifests with local GroundingDINO/SAM2 assets.")
    parser.add_argument("--mode", choices=["subject", "dynamic"], required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--masks-dir", required=True)
    parser.add_argument("--output-report")
    parser.add_argument("--output-manifest")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--box-threshold", type=float, default=0.3)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--max-dynamic-ratio", type=float, default=0.35)
    parser.add_argument("--groundingdino-repo", required=True)
    parser.add_argument("--groundingdino-config", required=True)
    parser.add_argument("--groundingdino-checkpoint", required=True)
    parser.add_argument("--groundingdino-checkpoint-min-bytes", type=int, default=0)
    parser.add_argument("--groundingdino-checkpoint-md5", default="")
    parser.add_argument("--text-encoder-path", default="")
    parser.add_argument("--sam2-repo", default="")
    parser.add_argument("--grounded-sam2-repo", default="")
    parser.add_argument("--sam2-config", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--sam2-checkpoint-min-bytes", type=int, default=0)
    return parser.parse_args()


def _missing_required(args: argparse.Namespace) -> list[str]:
    paths = [
        args.groundingdino_repo,
        args.groundingdino_config,
        args.groundingdino_checkpoint,
    ]
    missing = [path for path in paths if not path or not Path(path).exists()]
    if args.groundingdino_checkpoint and Path(args.groundingdino_checkpoint).exists():
        missing.extend(_missing_file_with_min_bytes(args.groundingdino_checkpoint, int(args.groundingdino_checkpoint_min_bytes or 0)))
        missing.extend(_missing_checkpoint_marker(args.groundingdino_checkpoint, args.groundingdino_checkpoint_md5))
    missing.extend(_missing_text_encoder(args.text_encoder_path))
    return missing


def _run_groundingdino(args: argparse.Namespace, images: list[Path], masks_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.groundingdino_repo)))
    try:
        from groundingdino.util.inference import load_image, load_model, predict  # type: ignore
    except ModuleNotFoundError:
        grounding_dino_root = Path(args.groundingdino_repo) / "grounding_dino"
        if grounding_dino_root.exists():
            sys.path.insert(0, str(grounding_dino_root))
        from groundingdino.util.inference import load_image, load_model, predict  # type: ignore

    import torch  # type: ignore

    device = _resolve_device(args.device, torch)
    model = load_model(args.groundingdino_config, args.groundingdino_checkpoint, device=device)
    sam2_predictor, sam2_status = _try_build_sam2(args, device)

    entries: list[dict[str, Any]] = []
    ratios: list[float] = []
    detection_count = 0
    for image_path in images:
        image_source, image_tensor = load_image(str(image_path))
        height, width = _shape_hw(image_source) or _image_size(image_path)
        boxes, confidences, labels = predict(
            model=model,
            image=image_tensor,
            caption=_normalize_prompt(args.prompt),
            box_threshold=float(args.box_threshold),
            text_threshold=float(args.text_threshold),
            device=device,
        )
        pixel_boxes = _boxes_to_pixel_xyxy(boxes, width, height)
        confidences_list = _tensor_to_list(confidences)
        labels_list = [str(label) for label in labels]
        detection_count += len(pixel_boxes)

        if sam2_predictor is not None and pixel_boxes:
            mask_bytes, method = _sam2_mask_bytes(sam2_predictor, image_source, pixel_boxes, width, height)
        else:
            mask_bytes, method = _box_mask_bytes(pixel_boxes, width, height), "groundingdino_box_mask"
        foreground_ratio = sum(1 for value in mask_bytes if value) / max(width * height, 1)
        ratios.append(foreground_ratio)
        mask_path = masks_dir / f"{image_path.stem}.png"
        _write_gray_png(mask_path, width, height, mask_bytes)
        entries.append(
            {
                "image_name": image_path.name,
                "mask_path": str(mask_path),
                "foreground_ratio": round(foreground_ratio, 6),
                "method": method,
                "detections": [
                    {"label": labels_list[index] if index < len(labels_list) else "", "confidence": confidences_list[index] if index < len(confidences_list) else None, "bbox_xyxy": box}
                    for index, box in enumerate(pixel_boxes)
                ],
            }
        )

    foreground_ratio = sum(ratios) / max(len(ratios), 1)
    common = {
        "schema": "fieldsplat.mask_manifest.v1" if args.mode == "subject" else "fieldsplat.dynamic_object_report.v1",
        "operator": "scope.subject_mask_generation" if args.mode == "subject" else "preprocess.dynamic_mask",
        "implementation": "external_command",
        "method": "groundingdino_sam2" if sam2_status == "enabled" else "groundingdino_box_mask",
        "semantic_model_used": True,
        "sam2_status": sam2_status,
        "prompt": _normalize_prompt(args.prompt),
        "input_image_count": len(images),
        "mask_count": len(entries),
        "foreground_ratio": round(foreground_ratio, 6),
        "background_ratio": round(1.0 - foreground_ratio, 6),
        "masks_dir": str(masks_dir),
        "mask_format": "png_full_resolution_binary",
        "images": entries,
        "detections_count": detection_count,
        "models": {
            "groundingdino_config": args.groundingdino_config,
            "groundingdino_checkpoint": args.groundingdino_checkpoint,
            "sam2_config": args.sam2_config or None,
            "sam2_checkpoint": args.sam2_checkpoint or None,
        },
    }
    if args.mode == "dynamic":
        dynamic_ratio = foreground_ratio
        common.update(
            {
                "passed": dynamic_ratio <= float(args.max_dynamic_ratio),
                "hard_fail": dynamic_ratio > float(args.max_dynamic_ratio),
                "dynamic_ratio": round(dynamic_ratio, 6),
                "max_dynamic_ratio": float(args.max_dynamic_ratio),
                "masked_frame_count": sum(1 for item in entries if float(item.get("foreground_ratio") or 0.0) > 0.0),
                "evaluated_frame_count": len(images),
            }
        )
    else:
        common.update(
            {
                "colmap_masking": {
                    "supported": True,
                    "apply_to_colmap": False,
                    "mask_path": str(masks_dir),
                    "reason": "Mask consumption is controlled by reconstruction_scope.apply_masks_to_colmap.",
                },
                "training_masking": {
                    "supported": True,
                    "apply_to_training": False,
                    "reason": "Training mask consumption is controlled by reconstruction_scope.apply_masks_to_training.",
                },
            }
        )
    return common


def _try_build_sam2(args: argparse.Namespace, device: str) -> tuple[Any | None, str]:
    if not args.sam2_checkpoint or not args.sam2_config:
        return None, "not_configured"
    if not Path(args.sam2_checkpoint).exists() or not Path(args.sam2_config).exists():
        return None, "checkpoint_or_config_missing"
    if _missing_file_with_min_bytes(args.sam2_checkpoint, int(args.sam2_checkpoint_min_bytes or 0)):
        return None, "checkpoint_too_small"
    repo = args.grounded_sam2_repo or args.sam2_repo
    if not repo or not Path(repo).exists():
        return None, "repo_missing"
    try:
        sys.path.insert(0, str(Path(repo)))
        previous_cwd = Path.cwd()
        os.chdir(repo)
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

        config_name = str(Path(args.sam2_config).relative_to(Path(repo) / "sam2")) if str(args.sam2_config).startswith(str(Path(repo) / "sam2")) else args.sam2_config
        model = build_sam2(config_name, args.sam2_checkpoint, device=device)
        os.chdir(previous_cwd)
        return SAM2ImagePredictor(model), "enabled"
    except Exception:
        try:
            os.chdir(previous_cwd)  # type: ignore[name-defined]
        except Exception:
            pass
        return None, "build_failed"


def _sam2_mask_bytes(predictor: Any, image_source: Any, pixel_boxes: list[list[int]], width: int, height: int) -> tuple[bytes, str]:
    import numpy as np  # type: ignore

    predictor.set_image(image_source)
    boxes = np.array(pixel_boxes, dtype=np.float32)
    masks, scores, _logits = predictor.predict(point_coords=None, point_labels=None, box=boxes, multimask_output=False)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        if mask.shape[0] != height or mask.shape[1] != width:
            continue
        combined = np.maximum(combined, mask.astype(np.uint8) * 255)
    if combined.max() == 0:
        return _box_mask_bytes(pixel_boxes, width, height), "groundingdino_box_mask_after_empty_sam2"
    return combined.tobytes(), "groundingdino_sam2"


def _box_mask_bytes(boxes: list[list[int]], width: int, height: int) -> bytes:
    mask = bytearray(width * height)
    for x0, y0, x1, y1 in boxes:
        x0 = max(0, min(width, int(x0)))
        y0 = max(0, min(height, int(y0)))
        x1 = max(0, min(width, int(x1)))
        y1 = max(0, min(height, int(y1)))
        if x1 <= x0 or y1 <= y0:
            continue
        for y in range(y0, y1):
            start = y * width + x0
            mask[start : y * width + x1] = b"\xff" * (x1 - x0)
    return bytes(mask)


def _boxes_to_pixel_xyxy(boxes: Any, width: int, height: int) -> list[list[int]]:
    values = _tensor_to_list(boxes)
    pixel_boxes: list[list[int]] = []
    for box in values:
        if len(box) < 4:
            continue
        cx, cy, bw, bh = [float(value) for value in box[:4]]
        x0 = int(round((cx - bw / 2.0) * width))
        y0 = int(round((cy - bh / 2.0) * height))
        x1 = int(round((cx + bw / 2.0) * width))
        y1 = int(round((cy + bh / 2.0) * height))
        pixel_boxes.append([x0, y0, x1, y1])
    return pixel_boxes


def _tensor_to_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "cpu"):
        return value.cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value or [])


def _shape_hw(value: Any) -> tuple[int, int] | None:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[0]), int(shape[1])


def _image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])[1], struct.unpack(">II", data[16:24])[0]
    if data.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(data):
            if data[index] != 0xFF:
                index += 1
                continue
            marker = data[index + 1]
            length = int.from_bytes(data[index + 2 : index + 4], "big")
            if marker in {0xC0, 0xC2}:
                height = int.from_bytes(data[index + 5 : index + 7], "big")
                width = int.from_bytes(data[index + 7 : index + 9], "big")
                return height, width
            index += max(length + 2, 2)
    return 1, 1


def _missing_file_with_min_bytes(value: str, min_bytes: int) -> list[str]:
    path = Path(value)
    if not path.exists():
        return [value]
    if min_bytes > 0 and path.is_file() and path.stat().st_size < min_bytes:
        return [f"{value}:size_bytes={path.stat().st_size}<min_bytes={min_bytes}"]
    return []


def _missing_text_encoder(value: str | None) -> list[str]:
    if not value:
        return []
    path = Path(str(value))
    if not path.exists() or not path.is_dir():
        return [str(value)]
    missing: list[str] = []
    for filename in ["config.json", "vocab.txt"]:
        candidate = path / filename
        if not candidate.exists() or candidate.stat().st_size <= 0:
            missing.append(str(candidate))
    if not any((path / filename).exists() and (path / filename).stat().st_size > 0 for filename in ["model.safetensors", "pytorch_model.bin"]):
        missing.append(f"{path}:missing_model_weights")
    return missing


def _missing_checkpoint_marker(path_value: str, expected_md5: str | None) -> list[str]:
    if not expected_md5:
        return []
    path = Path(path_value)
    marker_path = Path(f"{path}.verified.json")
    if not marker_path.exists():
        return [f"{path}:checkpoint_unverified"]
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        return [f"{path}:checkpoint_marker_invalid"]
    actual_md5 = str(marker.get("md5") or "").upper()
    if actual_md5 != str(expected_md5).upper():
        return [f"{path}:md5={actual_md5}<expected={str(expected_md5).upper()}"]
    if int(marker.get("size_bytes") or -1) != path.stat().st_size or int(marker.get("mtime_ns") or -1) != path.stat().st_mtime_ns:
        return [f"{path}:checkpoint_marker_mismatch"]
    return []


def _write_gray_png(path: Path, width: int, height: int, gray: bytes) -> None:
    rows = [b"\x00" + gray[y * width : (y + 1) * width] for y in range(height)]
    payload = b"".join(rows)
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(payload))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)


def _resolve_device(value: str, torch: Any) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _normalize_prompt(value: str) -> str:
    prompt = " ".join(value.strip().lower().split())
    return prompt if prompt.endswith(".") else f"{prompt}."


def _unavailable_payload(args: argparse.Namespace, images: list[Path], missing: list[str]) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.mask_manifest.v1" if args.mode == "subject" else "fieldsplat.dynamic_object_report.v1",
        "operator": "scope.subject_mask_generation" if args.mode == "subject" else "preprocess.dynamic_mask",
        "implementation": "external_command_unavailable",
        "semantic_model_used": False,
        "passed": args.mode == "dynamic",
        "hard_fail": False,
        "reason": "semantic_mask_dependency_missing",
        "missing_required_paths": missing,
        "input_image_count": len(images),
        "mask_count": 0,
        "foreground_ratio": 0.0,
        "background_ratio": 1.0,
        "images": [],
    }


def _failure_payload(args: argparse.Namespace, images: list[Path], exc: Exception) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.mask_manifest.v1" if args.mode == "subject" else "fieldsplat.dynamic_object_report.v1",
        "operator": "scope.subject_mask_generation" if args.mode == "subject" else "preprocess.dynamic_mask",
        "implementation": "external_command",
        "semantic_model_used": False,
        "passed": False,
        "hard_fail": args.mode == "dynamic",
        "reason": "semantic_mask_command_failed",
        "error_type": type(exc).__name__,
        "error": str(exc)[-1000:],
        "input_image_count": len(images),
        "mask_count": 0,
        "foreground_ratio": 0.0,
        "background_ratio": 1.0,
        "images": [],
    }


def _write_outputs(args: argparse.Namespace, payload: dict[str, Any]) -> None:
    for value in [args.output_manifest, args.output_report]:
        if not value:
            continue
        path = Path(value)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
