# toolkit/neural_sr.py
# 神经网络超分推理（权重与第三方源码路径由 YAML / 环境变量配置）

from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Optional, Tuple

def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


# Default third-party Real-ESRGAN repo path (relative to GeoMMAgent root); override with GEOMM_SR_REPO.
_DEFAULT_SR_REPO = str(_repo_root() / "Real-ESRGAN-master")

_upsampler_cache: Optional[Tuple[str, Any]] = None


def _sr_repo() -> str:
    return (
        os.environ.get("GEOMM_SR_REPO")
        or os.environ.get("REALESRGAN_REPO")  # 兼容旧名
        or _DEFAULT_SR_REPO
    ).strip()


def _ensure_sr_repo_on_path() -> None:
    repo = _sr_repo()
    p = pathlib.Path(repo)
    if p.is_dir():
        s = str(p.resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def _get_upsampler(model_path: str, tile: int = 0):
    """单例缓存，避免每条样本重复加载权重。"""
    global _upsampler_cache
    if _upsampler_cache is not None and _upsampler_cache[0] == model_path:
        return _upsampler_cache[1]

    _ensure_sr_repo_on_path()
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError as e:
        raise ImportError(
            f"超分依赖未就绪: {e}。请安装 basicsr、opencv、torch 等，"
            "并确保 sr_repo 指向含推理代码的源码目录。"
        ) from e

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"权重文件不存在: {model_path}")

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=4,
    )
    netscale = 4
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None,
    )
    _upsampler_cache = (model_path, upsampler)
    return upsampler


def _next_out_path(image_path: pathlib.Path, outscale: float) -> pathlib.Path:
    stem = f"{image_path.stem}_sr{int(outscale)}x"
    out = image_path.parent / f"{stem}.png"
    n = 0
    while out.exists() and out.resolve() != image_path.resolve():
        n += 1
        out = image_path.parent / f"{stem}_{n}.png"
    return out


def _weights_env() -> str:
    return (
        os.environ.get("GEOMM_SR_WEIGHTS")
        or os.environ.get("GEOMM_REALESRGAN_WEIGHTS")
        or ""
    ).strip()


def _tile_env() -> int:
    raw = os.environ.get("GEOMM_SR_TILE") or os.environ.get(
        "GEOMM_REALESRGAN_TILE", "0"
    )
    try:
        return max(0, int(raw or "0"))
    except ValueError:
        return 0


def run_neural_super_resolution(
    image_path: str,
    outscale: float = 4.0,
    model_path: Optional[str] = None,
    tile: int = 0,
) -> str:
    """
    使用配置的神经网络权重做超分。

    Args:
        image_path: 输入图像路径
        outscale: 放大倍数（常用 2 或 4）
        model_path: 权重 .pth；默认读环境变量 GEOMM_SR_WEIGHTS
        tile: >0 时切块推理，大图可减轻显存压力

    Returns:
        首行为输出 PNG 绝对路径，次行为说明；失败则返回 error 说明
    """
    mp = (model_path or _weights_env()).strip()
    if not mp:
        return (
            "[super_resolve] error: 未配置超分权重。"
            "请在配置中 SuperResolutionAgent 下设置 sr_weights。"
        )

    path = pathlib.Path(image_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = pathlib.Path(image_path).expanduser()

    if not path.is_file():
        return f"[super_resolve] error: file not found: {image_path}"

    try:
        import cv2
    except ImportError as e:
        return f"[super_resolve] error: need opencv-python ({e})"

    try:
        upsampler = _get_upsampler(mp, tile=tile)
    except (ImportError, FileNotFoundError, OSError, RuntimeError) as e:
        return f"[super_resolve] error: 超分模型初始化失败 ({e})"

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return f"[super_resolve] error: cannot read image: {path}"

    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        return (
            f"[super_resolve] error: 超分推理失败 ({e})。"
            "大图可在配置中设置 sr_tile（如 400）。"
        )

    out_file = _next_out_path(path, outscale)
    try:
        cv2.imwrite(str(out_file), output)
    except Exception as e:
        return f"[super_resolve] error: save failed ({e})"

    line1 = str(out_file.resolve())
    line2 = f"[super_resolve] neural SR outscale={outscale} -> {out_file.name}"
    return f"{line1}\n{line2}"


def clear_neural_sr_cache() -> None:
    """测试或切换权重时释放缓存。"""
    global _upsampler_cache
    _upsampler_cache = None
