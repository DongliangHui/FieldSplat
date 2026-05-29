from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def evaluate_pose_quality(transforms_path: str | Path) -> dict[str, Any]:
    path = Path(transforms_path)
    if not path.exists():
        return {"passed": False, "registered_frame_count": 0, "reason": "transforms_missing"}
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    return {
        "passed": len(frames) > 0,
        "registered_frame_count": len(frames),
        "trajectory_reasonable": len(frames) > 1,
    }
