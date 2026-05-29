from __future__ import annotations

import importlib.util
from pathlib import Path


PATCH_MARKER = "# FieldSplat patch: preserve COLMAP camera width/height"


def patch_source(source: str) -> tuple[str, bool]:
    if PATCH_MARKER in source:
        updated = source.replace("widths = torch.stack(widths).float()", "widths = torch.stack(widths).int()")
        updated = updated.replace("heights = torch.stack(heights).float()", "heights = torch.stack(heights).int()")
        return updated, updated != source

    updated = source
    updated = updated.replace(
        "        cxs = []\n        cys = []\n        image_filenames = []\n",
        "        cxs = []\n        cys = []\n        widths = []\n        heights = []\n        image_filenames = []\n",
        1,
    )
    updated = updated.replace(
        "            cxs.append(torch.tensor(cam.params[2]))\n            cys.append(torch.tensor(cam.params[3]))\n\n            image_filenames.append",
        "            cxs.append(torch.tensor(cam.params[2]))\n            cys.append(torch.tensor(cam.params[3]))\n            widths.append(torch.tensor(cam.width))\n            heights.append(torch.tensor(cam.height))\n\n            image_filenames.append",
        1,
    )
    updated = updated.replace(
        "        cxs = torch.stack(cxs).float()\n        cys = torch.stack(cys).float()\n\n        all_indices = torch.arange(len(image_filenames))\n",
        "        cxs = torch.stack(cxs).float()\n        cys = torch.stack(cys).float()\n        widths = torch.stack(widths).int()\n        heights = torch.stack(heights).int()\n\n        all_indices = torch.arange(len(image_filenames))\n",
        1,
    )
    updated = updated.replace(
        "        cameras = Cameras(\n            camera_to_worlds=poses[:, :3, :4],\n            fx=fxs,\n            fy=fys,\n            cx=cxs,\n            cy=cys,\n            camera_type=CameraType.PERSPECTIVE,\n        )\n",
        f"        {PATCH_MARKER}\n"
        "        cameras = Cameras(\n"
        "            camera_to_worlds=poses[:, :3, :4],\n"
        "            fx=fxs,\n"
        "            fy=fys,\n"
        "            cx=cxs,\n"
        "            cy=cys,\n"
        "            width=widths,\n"
        "            height=heights,\n"
        "            camera_type=CameraType.PERSPECTIVE,\n"
        "        )\n",
        1,
    )

    if updated == source or PATCH_MARKER not in updated:
        raise RuntimeError("Unable to patch splatfacto-w NerfW dataparser; expected source patterns were not found")
    return updated, True


def patch_installed_package() -> Path:
    spec = importlib.util.find_spec("splatfactow.nerfw_dataparser")
    if spec is None or spec.origin is None:
        raise RuntimeError("Unable to locate installed splatfactow.nerfw_dataparser")
    path = Path(spec.origin)
    source = path.read_text(encoding="utf-8")
    updated, changed = patch_source(source)
    if changed:
        path.write_text(updated, encoding="utf-8")
    return path


def main() -> None:
    path = patch_installed_package()
    print(f"splatfacto-w dataparser width/height patch applied: {path}")


if __name__ == "__main__":
    main()
