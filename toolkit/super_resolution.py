# toolkit/super_resolution.py
# 超分辨率：当前使用双线性插值上采样（非深度学习模型）

from __future__ import annotations

import pathlib

from PIL import Image


def _resample_bilinear():
    try:
        return Image.Resampling.BILINEAR
    except AttributeError:
        return Image.BILINEAR


def run_super_resolution(image_path: str, scale: int = 4) -> str:
    """
    对图像按整数倍上采样，使用 **双线性插值**（Pillow resize）。

    Args:
        image_path: 输入图像路径。
        scale: 宽高放大倍数，如 2、4（>=1）。

    Returns:
        首行为输出 PNG 绝对路径，次行为尺寸与插值说明。
    """
    path = pathlib.Path(image_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = pathlib.Path(image_path).expanduser()

    if not path.is_file():
        return f"[super_resolve] error: file not found: {image_path}"

    try:
        sc = int(scale)
    except (TypeError, ValueError):
        return f"[super_resolve] error: invalid scale={scale!r}"

    if sc < 1:
        return f"[super_resolve] error: scale must be >= 1, got {sc}"
    if sc > 16:
        return (
            f"[super_resolve] error: scale too large ({sc}), "
            "refuse to avoid excessive memory use"
        )

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        return f"[super_resolve] error: cannot open image ({e})"

    try:
        img.seek(0)
    except EOFError:
        pass

    w, h = img.size
    new_w = max(1, w * sc)
    new_h = max(1, h * sc)

    try:
        out_img = img.resize((new_w, new_h), resample=_resample_bilinear())
    except Exception as e:
        return f"[super_resolve] error: resize failed ({e})"

    out = path.parent / f"{path.stem}_sr{sc}x.png"
    n = 0
    while out.exists() and out.resolve() != path.resolve():
        n += 1
        out = path.parent / f"{path.stem}_sr{sc}x_{n}.png"

    try:
        out_img.save(out, format="PNG", optimize=True)
    except Exception as e:
        return f"[super_resolve] error: save failed ({e})"

    line1 = str(out.resolve())
    line2 = (
        f"[super_resolve] bilinear upsample {w}x{h} * {sc} -> {new_w}x{new_h}px"
    )
    return f"{line1}\n{line2}"
