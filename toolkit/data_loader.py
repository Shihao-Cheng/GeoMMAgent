# toolkit/data_loader.py
# GeoMMBench 数据加载工具：从 parquet 文件中提取问题、选项、图片等信息

import io
import tempfile
import os
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image


@dataclass
class BenchmarkSample:
    """一条 GeoMMBench 样本"""
    index: int
    question: str
    options: dict          # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str            # ground-truth，如 "D"
    image: Optional[Image.Image] = field(repr=False, default=None)
    image_path: Optional[str] = None
    hint: Optional[str] = None
    source: Optional[str] = None
    category: Optional[str] = None

    @property
    def prompt(self) -> str:
        """拼接为可直接送入 Agent 的完整 prompt"""
        lines = [self.question]
        for key in ("A", "B", "C", "D"):
            if key in self.options:
                lines.append(f"{key}. {self.options[key]}")
        return "\n".join(lines)


def load_benchmark(
    parquet_path: str,
    save_images_to: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[BenchmarkSample]:
    """
    从 parquet 文件加载 GeoMMBench 数据集。

    Parameters
    ----------
    parquet_path : str
        parquet 文件路径
    save_images_to : str | None
        若指定目录，则把图片写出为 PNG 文件并填充 sample.image_path；
        若为 None，则使用临时目录。
    limit : int | None
        只加载前 N 条（调试用）

    Returns
    -------
    List[BenchmarkSample]
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if limit is not None:
        df = df.head(limit)

    img_dir = save_images_to or tempfile.mkdtemp(prefix="geomm_imgs_")
    os.makedirs(img_dir, exist_ok=True)

    samples: List[BenchmarkSample] = []
    for _, row in df.iterrows():
        idx = int(row["index"])

        # 解析图片
        pil_img = None
        img_path = None
        img_data = row.get("image")
        if isinstance(img_data, dict) and img_data.get("bytes"):
            pil_img = Image.open(io.BytesIO(img_data["bytes"]))
            img_path = os.path.join(img_dir, f"{idx}.png")
            pil_img.save(img_path)

        sample = BenchmarkSample(
            index=idx,
            question=row["question"],
            options={k: row[k] for k in ("A", "B", "C", "D") if k in row and row[k] is not None},
            answer=row.get("answer", ""),
            image=pil_img,
            image_path=img_path,
            hint=row.get("hint"),
            source=row.get("source"),
            category=row.get("category"),
        )
        samples.append(sample)

    return samples
