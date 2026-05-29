from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Iterator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def main() -> int:
    args = _parse_args()
    images_dir = Path(args.images_dir)
    output_report = Path(args.output_report)
    images = sorted(path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    images = _order_images(images, images_dir=images_dir, image_order_manifest=args.image_order_manifest)
    if args.max_images:
        images = images[: max(1, int(args.max_images))]

    missing = _missing_required(args)
    if missing:
        _write_report(output_report, _unavailable_payload(args, images, missing))
        print(f"local feature matching unavailable: missing {missing}", file=sys.stderr)
        return 2

    if len(images) < 2:
        _write_report(output_report, _skipped_payload(args, images, "not_enough_images"))
        return 0

    try:
        report = _run_lightglue_aliked(args, images)
    except Exception as exc:
        _write_report(output_report, _failure_payload(args, images, exc))
        print(f"local feature matching failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    _write_report(output_report, report)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LightGlue/ALIKED pair matching and write a FieldSplat pre-SfM report.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--lightglue-repo", required=True)
    parser.add_argument("--lightglue-checkpoint", required=True)
    parser.add_argument("--aliked-repo", required=True)
    parser.add_argument("--aliked-checkpoint", required=True)
    parser.add_argument("--aliked-model", default="aliked-n16rot")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--pair-window", type=int, default=8)
    parser.add_argument("--max-num-keypoints", type=int, default=2048)
    parser.add_argument("--min-matches", type=int, default=15)
    parser.add_argument("--image-order-manifest")
    parser.add_argument("--output-colmap-features-dir")
    parser.add_argument("--output-colmap-match-list")
    return parser.parse_args()


def _missing_required(args: argparse.Namespace) -> list[str]:
    paths = [
        args.lightglue_repo,
        Path(args.lightglue_repo) / "lightglue" / "__init__.py",
        Path(args.lightglue_repo) / "lightglue" / "aliked.py",
        args.lightglue_checkpoint,
        args.aliked_repo,
        Path(args.aliked_repo) / "nets" / "aliked.py",
        args.aliked_checkpoint,
    ]
    missing: list[str] = []
    for value in paths:
        path = Path(value)
        if not path.exists():
            missing.append(str(path))
        elif path.is_file() and path.stat().st_size <= 0:
            missing.append(f"{path}:empty")
    return missing


def _run_lightglue_aliked(args: argparse.Namespace, images: list[Path]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(args.lightglue_repo)))
    sys.path.insert(0, str(Path(args.aliked_repo)))

    import torch  # type: ignore
    from lightglue import ALIKED, LightGlue  # type: ignore
    from lightglue.utils import load_image, rbd  # type: ignore

    device = _resolve_device(args.device, torch)
    pairs = _select_pairs(images, max_pairs=int(args.max_pairs or 80), pair_window=int(args.pair_window or 8))

    with _patch_torch_hub_downloads(torch, args):
        extractor = ALIKED(model_name=args.aliked_model, max_num_keypoints=int(args.max_num_keypoints or 2048)).eval().to(device)
        matcher = LightGlue(features="aliked").eval().to(device)

    feature_cache: dict[Path, dict[str, Any]] = {}
    pair_reports: list[dict[str, Any]] = []
    colmap_pairs: list[dict[str, Any]] = []
    total_match_count = 0
    with torch.inference_mode():
        for image0_path, image1_path in pairs:
            feats0 = _features_for(image0_path, extractor, load_image, feature_cache, device)
            feats1 = _features_for(image1_path, extractor, load_image, feature_cache, device)
            matches01 = matcher({"image0": feats0, "image1": feats1})
            feats0_rbd, feats1_rbd, matches01_rbd = [rbd(item) for item in [feats0, feats1, matches01]]
            matches = matches01_rbd["matches"]
            match_count = int(matches.shape[0]) if hasattr(matches, "shape") else len(matches)
            keypoints0 = int(feats0_rbd["keypoints"].shape[0]) if "keypoints" in feats0_rbd else 0
            keypoints1 = int(feats1_rbd["keypoints"].shape[0]) if "keypoints" in feats1_rbd else 0
            total_match_count += match_count
            match_rows = _match_rows(matches)
            if match_count >= int(args.min_matches or 15):
                colmap_pairs.append({"image0": image0_path, "image1": image1_path, "matches": match_rows})
            pair_reports.append(
                {
                    "image0": image0_path.name,
                    "image1": image1_path.name,
                    "match_count": match_count,
                    "keypoints0": keypoints0,
                    "keypoints1": keypoints1,
                }
            )

    mean_matches = total_match_count / max(len(pair_reports), 1)
    colmap_import = _write_colmap_import_outputs(args, images_dir=Path(args.images_dir), feature_cache=feature_cache, colmap_pairs=colmap_pairs)
    integration_status = "colmap_database_import_ready" if colmap_import["import_ready"] else "pre_sfm_pair_match_manifest"
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "operator": "pose.lightglue_aliked_matching",
        "implementation": "external_command",
        "method": "lightglue_aliked",
        "integration_status": integration_status,
        "passed": bool(pair_reports),
        "input_image_count": len(images),
        "evaluated_image_count": len({path for pair in pairs for path in pair}),
        "pair_count": len(pair_reports),
        "total_match_count": total_match_count,
        "mean_matches_per_pair": round(mean_matches, 3),
        "colmap_import": colmap_import,
        "pairs": pair_reports,
        "device": device,
        "models": {
            "lightglue_repo": args.lightglue_repo,
            "lightglue_checkpoint": args.lightglue_checkpoint,
            "aliked_repo": args.aliked_repo,
            "aliked_checkpoint": args.aliked_checkpoint,
            "aliked_model": args.aliked_model,
        },
        "notes": [
            "This runs real ALIKED extraction and LightGlue matching before SfM.",
            "When colmap_import.import_ready is true, COLMAP can consume the exported keypoints and learned matches through feature_importer and matches_importer.",
        ],
    }


def _features_for(path: Path, extractor: Any, load_image: Any, cache: dict[Path, dict[str, Any]], device: str) -> dict[str, Any]:
    if path not in cache:
        image = load_image(str(path)).to(device)
        cache[path] = extractor.extract(image)
    return cache[path]


@contextlib.contextmanager
def _patch_torch_hub_downloads(torch: Any, args: argparse.Namespace) -> Iterator[None]:
    original = torch.hub.load_state_dict_from_url

    def load_state_dict_from_url(url: str, *call_args: Any, **kwargs: Any) -> Any:
        text = str(url)
        map_location = kwargs.get("map_location", "cpu")
        if "aliked_lightglue" in text:
            return torch.load(str(args.lightglue_checkpoint), map_location=map_location)
        if str(args.aliked_model) in text:
            return torch.load(str(args.aliked_checkpoint), map_location=map_location)
        return original(url, *call_args, **kwargs)

    torch.hub.load_state_dict_from_url = load_state_dict_from_url
    try:
        yield
    finally:
        torch.hub.load_state_dict_from_url = original


def _match_rows(matches: Any) -> list[list[int]]:
    if hasattr(matches, "detach"):
        values = matches.detach().cpu().tolist()
    elif hasattr(matches, "tolist"):
        values = matches.tolist()
    else:
        values = list(matches)
    rows: list[list[int]] = []
    for item in values:
        if len(item) < 2:
            continue
        rows.append([int(item[0]), int(item[1])])
    return rows


def _write_colmap_import_outputs(
    args: argparse.Namespace,
    *,
    images_dir: Path,
    feature_cache: dict[Path, dict[str, Any]],
    colmap_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    features_dir_value = getattr(args, "output_colmap_features_dir", None)
    match_list_value = getattr(args, "output_colmap_match_list", None)
    if not features_dir_value or not match_list_value:
        return {
            "import_ready": False,
            "reason": "colmap_import_paths_not_configured",
            "match_type": "raw",
            "feature_file_count": 0,
            "imported_pair_count": 0,
        }
    features_dir = Path(features_dir_value)
    match_list_path = Path(match_list_value)
    features_dir.mkdir(parents=True, exist_ok=True)
    match_list_path.parent.mkdir(parents=True, exist_ok=True)

    image_names = {path: _colmap_image_name(path, images_dir) for path in feature_cache}
    feature_file_count = 0
    for image_path, feats in sorted(feature_cache.items(), key=lambda item: image_names[item[0]]):
        keypoints = _keypoints_for_colmap(feats)
        target = features_dir / f"{image_names[image_path]}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            handle.write(f"{len(keypoints)} 128\n")
            descriptor = " ".join("0" for _ in range(128))
            for x, y in keypoints:
                handle.write(f"{x:.6f} {y:.6f} 1.0 0.0 {descriptor}\n")
        feature_file_count += 1

    imported_pair_count = 0
    with match_list_path.open("w", encoding="utf-8") as handle:
        for pair in colmap_pairs:
            image0 = pair["image0"]
            image1 = pair["image1"]
            matches = pair["matches"]
            if image0 not in image_names or image1 not in image_names or not matches:
                continue
            handle.write(f"{image_names[image0]} {image_names[image1]}\n")
            for match0, match1 in matches:
                handle.write(f"{int(match0)} {int(match1)}\n")
            handle.write("\n")
            imported_pair_count += 1

    return {
        "import_ready": feature_file_count > 1 and imported_pair_count > 0,
        "reason": None if feature_file_count > 1 and imported_pair_count > 0 else "no_pairs_above_min_matches",
        "features_dir": str(features_dir),
        "match_list_path": str(match_list_path),
        "match_type": "raw",
        "feature_format": "colmap_sift_text_keypoints_with_dummy_descriptors",
        "feature_file_count": feature_file_count,
        "imported_pair_count": imported_pair_count,
        "min_matches_per_pair": int(args.min_matches or 15),
    }


def _keypoints_for_colmap(feats: dict[str, Any]) -> list[tuple[float, float]]:
    keypoints = feats.get("keypoints")
    if hasattr(keypoints, "dim") and keypoints.dim() == 3:
        keypoints = keypoints[0]
    elif isinstance(keypoints, list) and keypoints and isinstance(keypoints[0], list) and keypoints[0] and isinstance(keypoints[0][0], list):
        keypoints = keypoints[0]
    if hasattr(keypoints, "detach"):
        values = keypoints.detach().cpu().tolist()
    elif hasattr(keypoints, "tolist"):
        values = keypoints.tolist()
    else:
        values = list(keypoints or [])
    result: list[tuple[float, float]] = []
    for item in values:
        if len(item) < 2:
            continue
        result.append((float(item[0]), float(item[1])))
    return result


def _colmap_image_name(path: Path, images_dir: Path) -> str:
    try:
        return path.relative_to(images_dir).as_posix()
    except ValueError:
        return path.name


def _order_images(images: list[Path], *, images_dir: Path, image_order_manifest: str | None = None) -> list[Path]:
    manifest_path = Path(image_order_manifest) if image_order_manifest else images_dir.parent.parent / "preprocess_metadata.json"
    if not manifest_path.exists():
        return images
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return images
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        return images
    order: dict[str, int] = {}
    for index, value in enumerate(source_files):
        if isinstance(value, str) and value and value not in order:
            order[value.replace("\\", "/")] = index
            order[Path(value).name] = index

    def key(path: Path) -> tuple[int, int, str]:
        image_name = _colmap_image_name(path, images_dir)
        rank = order.get(image_name, order.get(path.name))
        if rank is None:
            return (1, len(order), path.name)
        return (0, rank, path.name)

    return sorted(images, key=key)


def _select_pairs(images: list[Path], *, max_pairs: int, pair_window: int = 8) -> list[tuple[Path, Path]]:
    max_pairs = max(1, int(max_pairs))
    pair_window = max(1, int(pair_window))
    pairs: list[tuple[Path, Path]] = []
    seen: set[tuple[Path, Path]] = set()

    def add_pair(a: Path, b: Path) -> None:
        key = (a, b)
        if a == b or key in seen or len(pairs) >= max_pairs:
            return
        seen.add(key)
        pairs.append(key)

    for offset in range(1, min(pair_window, len(images) - 1) + 1):
        for index in range(len(images) - offset):
            add_pair(images[index], images[index + offset])
            if len(pairs) >= max_pairs:
                return pairs

    stride = max(2, len(images) // max(1, max_pairs // 2))
    for index in range(0, len(images) - stride, stride):
        add_pair(images[index], images[index + stride])
    return pairs


def _resolve_device(value: str, torch: Any) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return value


def _unavailable_payload(args: argparse.Namespace, images: list[Path], missing: list[str]) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "operator": "pose.lightglue_aliked_matching",
        "implementation": "external_command_unavailable",
        "method": "lightglue_aliked",
        "passed": False,
        "reason": "local_feature_matching_dependency_missing",
        "missing_required_paths": missing,
        "input_image_count": len(images),
        "pair_count": 0,
        "total_match_count": 0,
    }


def _skipped_payload(args: argparse.Namespace, images: list[Path], reason: str) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "operator": "pose.lightglue_aliked_matching",
        "implementation": "external_command",
        "method": "lightglue_aliked",
        "passed": False,
        "reason": reason,
        "input_image_count": len(images),
        "pair_count": 0,
        "total_match_count": 0,
    }


def _failure_payload(args: argparse.Namespace, images: list[Path], exc: Exception) -> dict[str, Any]:
    return {
        "schema": "fieldsplat.local_feature_matching.v1",
        "operator": "pose.lightglue_aliked_matching",
        "implementation": "external_command",
        "method": "lightglue_aliked",
        "passed": False,
        "reason": "local_feature_matching_command_failed",
        "error_type": type(exc).__name__,
        "error": str(exc)[-1000:],
        "input_image_count": len(images),
        "pair_count": 0,
        "total_match_count": 0,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
