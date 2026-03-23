# toolkit/data_loader.py
# Load GeoMMBench-style samples from Parquet (question, options, image, metadata).

import io
import json
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image


@dataclass
class BenchmarkSample:
    """One benchmark row: question, options, label, and optional image path."""
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
        """
        Single-line task string: question ending with ``?``, then ``A:`` … ``D:`` tokens
        with no newline, so the option block can be parsed after the first ``?``.
        """
        parts_opts = []
        for key in ("A", "B", "C", "D"):
            if key in self.options:
                parts_opts.append(f"{key}: {self.options[key]}")
        opt_str = " ".join(parts_opts)
        q = self.question.strip()
        if not q.endswith("?"):
            q = q + "?"
        return f"{q}{opt_str}"


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


def get_benchmark_sample_by_id(parquet_path: str, sample_id: int) -> BenchmarkSample:
    """从 parquet 中取出 index == sample_id 的一条样本。"""
    samples = load_benchmark(parquet_path=parquet_path)
    for s in samples:
        if s.index == sample_id:
            return s
    raise ValueError(f"No row with index={sample_id!r} in {parquet_path}")


def get_benchmark_sample_from_jsonl(
    jsonl_path: str,
    images_root: str | Path,
    sample_id: int,
) -> BenchmarkSample:
    """
    从 query.jsonl 风格文件按 id 取一条；image 字段为相对路径时相对于 images_root。
    """
    p = Path(jsonl_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    root = Path(images_root)
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if int(row.get("id", -1)) != sample_id:
                continue
            rel = row.get("image") or ""
            img_path = Path(rel)
            if not img_path.is_absolute():
                img_path = (root / rel).resolve()
            else:
                img_path = img_path.resolve()
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")
            opts = row.get("options") or {}
            return BenchmarkSample(
                index=int(row["id"]),
                question=row["question"],
                options={k: opts[k] for k in ("A", "B", "C", "D") if k in opts},
                answer=str(row.get("answer", "") or ""),
                image=None,
                image_path=str(img_path),
                hint=row.get("hint"),
                source=row.get("source"),
                category=row.get("category"),
            )
    raise ValueError(f"No id={sample_id} in {jsonl_path}")
