#!/usr/bin/env bash
set -euo pipefail

mkdir -p /workspace/locks

TORCH_LIB="$(
python - <<'PY' 2>/dev/null || true
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
if [[ -n "${TORCH_LIB}" && -d "${TORCH_LIB}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"
fi

(
flock -x 9

python - <<'PY'
import importlib
import os
import subprocess
import sys
import shutil
from pathlib import Path

packages = [
    ("simple_knn._C", "/opt/InstantSplatPP/submodules/simple-knn"),
    ("diff_gaussian_rasterization", "/opt/InstantSplatPP/submodules/diff-gaussian-rasterization"),
    ("fused_ssim", "/opt/InstantSplatPP/submodules/fused-ssim"),
]
if os.environ.get("SKIP_INSTANTSPLATPP_EXTENSION_BUILD") == "1":
    print("[gpu-entrypoint] skipping InstantSplat++ CUDA extension build")
    packages = []

missing = []
for module_name, source_dir in packages:
    try:
        importlib.import_module(module_name)
    except Exception:
        missing.append((module_name, Path(source_dir)))

if not missing:
    print("[gpu-entrypoint] InstantSplat++ CUDA extensions already available")
    raise SystemExit(0)

if not Path("/usr/local/cuda-11.8/bin/nvcc").exists() and not Path("/usr/local/cuda/bin/nvcc").exists():
    print("[gpu-entrypoint] nvcc is unavailable; skipping InstantSplat++ CUDA extension build")
    raise SystemExit(0)

env = os.environ.copy()
env.setdefault("CUDA_HOME", "/usr/local/cuda-11.8")
env.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
env.setdefault("MAX_JOBS", "1")

for module_name, source_dir in missing:
    if not source_dir.exists():
        print(f"[gpu-entrypoint] {source_dir} is missing; cannot build {module_name}")
        continue
    build_dir = source_dir / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    print(f"[gpu-entrypoint] building {module_name} from {source_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-cache-dir", str(source_dir)],
        env=env,
    )
    if result.returncode != 0:
        print(f"[gpu-entrypoint] failed to build {module_name}; worker will start with this operator unavailable")

groundingdino_source = Path("/opt/GroundingDINO")
if groundingdino_source.exists():
    sys.path.insert(0, str(groundingdino_source))
    try:
        importlib.import_module("groundingdino._C")
        print("[gpu-entrypoint] GroundingDINO CUDA extension already available")
    except Exception:
        if not Path("/usr/local/cuda-11.8/bin/nvcc").exists() and not Path("/usr/local/cuda/bin/nvcc").exists():
            print("[gpu-entrypoint] nvcc is unavailable; skipping GroundingDINO CUDA extension build")
        elif not (
            Path("/usr/local/cuda-11.8/include/cusparse.h").exists()
            or Path("/usr/local/cuda/include/cusparse.h").exists()
        ):
            print("[gpu-entrypoint] CUDA sparse headers are unavailable; GroundingDINO will use its PyTorch fallback")
        else:
            build_dir = groundingdino_source / "build"
            if build_dir.exists():
                shutil.rmtree(build_dir)
            print(f"[gpu-entrypoint] building groundingdino._C from {groundingdino_source}")
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=str(groundingdino_source),
                env=env,
            )
            if result.returncode != 0:
                print("[gpu-entrypoint] failed to build groundingdino._C; semantic operators may fall back or fail")
else:
    print("[gpu-entrypoint] /opt/GroundingDINO is missing; skipping GroundingDINO CUDA extension build")
PY
) 9>/workspace/locks/instantsplatpp_cuda_extensions.lock

exec "$@"
