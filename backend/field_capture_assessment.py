from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.modules.field_capture_assessment import run_assessment


def main() -> int:
    parser = argparse.ArgumentParser(description="Field Capture Assessment / 现场素材采集评估器")
    parser.add_argument("--input", required=True, help="图片文件夹、视频文件或素材目录")
    parser.add_argument("--scene-type", default="indoor_room", help="现场类型，例如 indoor_room / corridor / outdoor_scene / object")
    parser.add_argument("--target-quality", default="standard", help="目标质量，例如 standard / forensic")
    parser.add_argument("--output", required=True, help="报告输出目录")
    parser.add_argument("--no-recursive", action="store_true", help="只扫描输入目录第一层")
    parser.add_argument("--key-area", action="append", default=[], help="关键区域标注，可重复传入")
    args = parser.parse_args()

    result = run_assessment(
        Path(args.input),
        scene_type=args.scene_type,
        target_quality=args.target_quality,
        output_dir=Path(args.output),
        recursive=not args.no_recursive,
        key_areas=args.key_area,
    )
    print(json.dumps({"report_path": str(result.report_path), "selected_assets_manifest_path": str(result.manifest_path), "report": result.report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
