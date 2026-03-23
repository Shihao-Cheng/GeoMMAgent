# toolkit/general.py
# 通用预处理 / 后处理工具集（General Toolkit）
#
# 包含：格式转换、滤波、缩放、超分辨率

from __future__ import annotations

import os
import pathlib
from typing import List, Tuple

from camel.toolkits import FunctionTool
from PIL import Image, ImageFilter

# ── 格式转换 ────────────────────────────────────────

def _normalize_target_format(target_format: str) -> str:
    """'PNG', '.png', 'jpeg' -> canonical short name for branching."""
    fmt = target_format.strip().lower().lstrip(".")
    aliases = {
        "jpeg": "jpg",
        "tif": "tiff",
    }
    return aliases.get(fmt, fmt)


def _suffix_to_format(suffix: str) -> str | None:
    """从扩展名推断格式标签（用于日志）。"""
    s = suffix.lower()
    mapping = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".tif": "TIFF",
        ".tiff": "TIFF",
        ".bmp": "BMP",
        ".webp": "WEBP",
        ".gif": "GIF",
    }
    return mapping.get(s)


def _prepare_for_save(img, target: str):
    """按目标格式调整 mode（如 JPEG 需 RGB）。"""
    if target in ("jpg", "jpeg"):
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[-1])
            return background
        if img.mode == "P":
            return img.convert("RGBA").convert("RGB")
        if img.mode not in ("RGB", "L"):
            return img.convert("RGB")
        return img
    if target == "png":
        return img
    if target == "tiff":
        return img
    if target in ("bmp", "webp"):
        if img.mode == "P":
            return img.convert("RGBA")
        return img
    return img


def _output_path_for(
    src: pathlib.Path, target: str,
) -> Tuple[pathlib.Path, str]:
    """生成输出路径与 PIL save 的 format 关键字。"""
    ext_map = {
        "png": (".png", "PNG"),
        "jpg": (".jpg", "JPEG"),
        "jpeg": (".jpg", "JPEG"),
        "tiff": (".tif", "TIFF"),
        "tif": (".tif", "TIFF"),
        "bmp": (".bmp", "BMP"),
        "webp": (".webp", "WEBP"),
    }
    ext, pil_fmt = ext_map.get(target, (".png", "PNG"))
    out = src.parent / f"{src.stem}_converted{ext}"
    n = 0
    while out.exists() and out.resolve() != src.resolve():
        n += 1
        out = src.parent / f"{src.stem}_converted_{n}{ext}"
    return out, pil_fmt


def convert_format(image_path: str, target_format: str = "png") -> str:
    """
    读取图像（自动识别编码），并保存为目标格式。

    使用 Pillow 打开文件，真实格式以解码结果为准；扩展名仅作辅助说明。
    多页 TIFF 仅处理第一帧。

    Args:
        image_path (str): 输入图像路径。
        target_format (str): 目标格式，如 'png'（推荐统一管线）、'jpg'、'tiff'。

    Returns:
        str: 首行为输出文件绝对路径；次行起为检测与转换说明。
    """
    path = pathlib.Path(image_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = pathlib.Path(image_path).expanduser()

    if not path.is_file():
        return f"[convert_format] error: file not found: {image_path}"

    target = _normalize_target_format(target_format or "png")
    if target not in ("png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"):
        return (
            f"[convert_format] error: unsupported target_format={target_format!r} "
            f"(use png, jpg, tiff, bmp, webp)"
        )

    ext_hint = _suffix_to_format(path.suffix) or "unknown"

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        return f"[convert_format] error: cannot open image ({e})"

    # 多页 TIFF：取第一帧
    try:
        img.seek(0)
    except EOFError:
        pass

    pil_format = getattr(img, "format", None) or ext_hint
    img = _prepare_for_save(img, target)
    out_path, save_fmt = _output_path_for(path, target)

    try:
        save_kw = {"format": save_fmt}
        if save_fmt == "JPEG":
            save_kw["quality"] = 95
            save_kw["optimize"] = True
        img.save(out_path, **save_kw)
    except Exception as e:
        return f"[convert_format] error: save failed ({e})"

    line1 = str(out_path.resolve())
    line2 = (
        f"[convert_format] path_ext={path.suffix!r} pil_format={pil_format!r} "
        f"target={save_fmt} -> saved"
    )
    return f"{line1}\n{line2}"

# ── 滤波 ───────────────────────────────────────────

def _normalize_filter_method(method: str) -> str:
    """别名归一，便于 VLM 自然语言描述。"""
    m = (method or "gaussian").strip().lower()
    aliases = {
        "gauss": "gaussian",
        "blur": "gaussian",
        "denoise": "median",
        "noise": "median",
        "unsharp": "sharpen",
        "edge": "edge_enhance",
    }
    return aliases.get(m, m)


def _next_filtered_path(src: pathlib.Path) -> pathlib.Path:
    out = src.parent / f"{src.stem}_filtered.png"
    n = 0
    while out.exists() and out.resolve() != src.resolve():
        n += 1
        out = src.parent / f"{src.stem}_filtered_{n}.png"
    return out


def _apply_pil_filter(img: Image.Image, method_key: str) -> Image.Image:
    """在 PIL 图像上应用滤波（不改变 mode，由调用方保证）。"""
    if method_key == "gaussian":
        return img.filter(ImageFilter.GaussianBlur(radius=2))
    if method_key == "median":
        return img.filter(ImageFilter.MedianFilter(size=3))
    if method_key == "sharpen":
        return img.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
    if method_key == "sharp":
        return img.filter(ImageFilter.SHARPEN)
    if method_key == "smooth":
        return img.filter(ImageFilter.SMOOTH)
    if method_key == "edge_enhance":
        return img.filter(ImageFilter.EDGE_ENHANCE)
    raise ValueError(f"unknown method_key={method_key!r}")


def filter_image(image_path: str, method: str = "gaussian") -> str:
    """
    对图像做平滑、去噪或锐化（Pillow ImageFilter），结果保存为 PNG。

    Args:
        image_path (str): 输入图像路径。
        method (str): 滤波类型，支持：
            - gaussian — 高斯模糊（平滑）
            - median — 中值滤波（椒盐噪声）
            - sharpen — Unsharp Mask（细节增强，默认推荐「锐化」）
            - sharp — 核锐化（更强边缘）
            - smooth — 平滑
            - edge_enhance — 边缘增强

    Returns:
        str: 首行为输出 PNG 绝对路径，次行为 method 与说明。
    """
    path = pathlib.Path(image_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = pathlib.Path(image_path).expanduser()

    if not path.is_file():
        return f"[filter_image] error: file not found: {image_path}"

    key = _normalize_filter_method(method)
    allowed = frozenset(
        {"gaussian", "median", "sharpen", "sharp", "smooth", "edge_enhance"}
    )
    if key not in allowed:
        return (
            f"[filter_image] error: unsupported method={method!r} "
            f"(use {sorted(allowed)})"
        )

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        return f"[filter_image] error: cannot open image ({e})"

    try:
        img.seek(0)
    except EOFError:
        pass

    # 调色板 / 非 RGB：先转成 RGB 再滤波，避免模式不一致
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode == "RGBA":
        pass
    elif img.mode not in ("RGB", "L", "LA"):
        img = img.convert("RGB")

    try:
        out_img = _apply_pil_filter(img, key)
    except Exception as e:
        return f"[filter_image] error: filter failed ({e})"

    out_path = _next_filtered_path(path)
    try:
        out_img.save(out_path, format="PNG", optimize=True)
    except Exception as e:
        return f"[filter_image] error: save failed ({e})"

    line1 = str(out_path.resolve())
    line2 = f"[filter_image] method={key!r} -> saved as PNG"
    return f"{line1}\n{line2}"


def _next_scaled_path(src: pathlib.Path) -> pathlib.Path:
    out = src.parent / f"{src.stem}_scaled.png"
    n = 0
    while out.exists() and out.resolve() != src.resolve():
        n += 1
        out = src.parent / f"{src.stem}_scaled_{n}.png"
    return out


def _resize_resample():
    """Pillow 9+ 使用 Resampling.LANCZOS。"""
    try:
        return Image.Resampling.LANCZOS
    except AttributeError:
        return Image.LANCZOS


# ── 缩放 ───────────────────────────────────────────

def scale_image(image_path: str, scale_factor: float = 1.0) -> str:
    """
    将图像按倍率缩放（宽高同比例），使用 LANCZOS 重采样，结果保存为 PNG。

    Args:
        image_path (str): 输入图像路径。
        scale_factor (float): 相对原图的倍率；>1 放大，<1 缩小，=1 仍输出副本。

    Returns:
        str: 首行为输出 PNG 绝对路径，次行为尺寸说明。
    """
    path = pathlib.Path(image_path).expanduser()
    try:
        path = path.resolve()
    except OSError:
        path = pathlib.Path(image_path).expanduser()

    if not path.is_file():
        return f"[scale_image] error: file not found: {image_path}"

    try:
        sf = float(scale_factor)
    except (TypeError, ValueError):
        return f"[scale_image] error: invalid scale_factor={scale_factor!r}"

    if sf <= 0:
        return f"[scale_image] error: scale_factor must be positive, got {sf}"
    if sf > 1e3:
        return (
            f"[scale_image] error: scale_factor too large ({sf}), "
            "refuse to avoid excessive memory use"
        )

    try:
        img = Image.open(path)
        img.load()
    except Exception as e:
        return f"[scale_image] error: cannot open image ({e})"

    try:
        img.seek(0)
    except EOFError:
        pass

    w, h = img.size
    new_w = max(1, int(round(w * sf)))
    new_h = max(1, int(round(h * sf)))

    try:
        out_img = img.resize((new_w, new_h), resample=_resize_resample())
    except Exception as e:
        return f"[scale_image] error: resize failed ({e})"

    out_path = _next_scaled_path(path)
    try:
        out_img.save(out_path, format="PNG", optimize=True)
    except Exception as e:
        return f"[scale_image] error: save failed ({e})"

    line1 = str(out_path.resolve())
    line2 = (
        f"[scale_image] {w}x{h} * {sf} -> {new_w}x{new_h}px, saved as PNG"
    )
    return f"{line1}\n{line2}"


# ── 超分辨率 ───────────────────────────────────────

def super_resolve(image_path: str, scale: int = 4) -> str:
    """
    超分辨率：若已在配置中设置超分权重（GEOMM_SR_WEIGHTS），则走神经网络超分；
    否则回退为双线性插值（toolkit.super_resolution）。

    Args:
        image_path (str): 输入图像路径。
        scale (int): 放大倍数，如 2、4。

    Returns:
        str: 首行为输出 PNG 路径，次行为说明。
    """
    weights = (
        os.environ.get("GEOMM_SR_WEIGHTS", "").strip()
        or os.environ.get("GEOMM_REALESRGAN_WEIGHTS", "").strip()
    )
    if weights and os.path.isfile(weights):
        from toolkit.neural_sr import run_neural_super_resolution

        raw_tile = os.environ.get("GEOMM_SR_TILE") or os.environ.get(
            "GEOMM_REALESRGAN_TILE", "0"
        )
        try:
            tile = int(raw_tile or "0")
        except ValueError:
            tile = 0
        try:
            outscale = float(scale)
        except (TypeError, ValueError):
            outscale = 4.0
        return run_neural_super_resolution(
            image_path,
            outscale=outscale,
            model_path=weights,
            tile=max(0, tile),
        )

    from toolkit.super_resolution import run_super_resolution

    return run_super_resolution(image_path, scale=int(scale) if scale else 4)


# ── 工具注册（按能力拆分，每个 Agent 只挂载对应工具）──────────

def get_format_conversion_tools() -> List[FunctionTool]:
    """格式转换（供 FormatConversionAgent）"""
    return [FunctionTool(convert_format)]


def get_image_filter_tools() -> List[FunctionTool]:
    """滤波（供 ImageFilterAgent）"""
    return [FunctionTool(filter_image)]


def get_scale_tools() -> List[FunctionTool]:
    """缩放（供 ScaleAgent；倍率由 VLM 根据任务决定）"""
    return [FunctionTool(scale_image)]


def get_super_resolution_tools() -> List[FunctionTool]:
    """超分辨率（供 SuperResolutionAgent）"""
    return [FunctionTool(super_resolve)]


def get_general_tools() -> List[FunctionTool]:
    """全部通用预处理工具（聚合，便于调试或旧代码）"""
    return (
        get_format_conversion_tools()
        + get_image_filter_tools()
        + get_scale_tools()
        + get_super_resolution_tools()
    )
