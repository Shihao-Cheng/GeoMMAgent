# toolkit/super_resolution.py
# 超分辨率模型调用（占位文件，待接入具体模型）
#
# 预期接口：
#   run_super_resolution(image_path: str, scale: int = 4) -> str
#
# 推荐可接入的模型：
#   - Real-ESRGAN / HAT / SwinIR（通用超分）
#   - RSISR / TransENet（遥感专用超分）
#
# 实现时请返回超分后图像的保存路径。


def run_super_resolution(image_path: str, scale: int = 4) -> str:
    """
    对遥感图像执行超分辨率增强。

    Args:
        image_path: 输入图像路径。
        scale: 超分倍率。

    Returns:
        超分辨率结果图像的保存路径。
    """
    raise NotImplementedError(
        "Super-resolution model not yet integrated. "
        "Please implement this function with your preferred SR model."
    )
