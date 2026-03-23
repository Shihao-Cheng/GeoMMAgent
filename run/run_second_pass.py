#!/usr/bin/env python3
"""
二次评判：读取 results/<run>/self_eval_failed.jsonl，
对每条做一次「推理复核 + 第二轮四维度自评」，写入 self_eval_second_pass.jsonl（每样本最多一次重试）。

用法:
    python run/run_second_pass.py results/20260320_120000
    python run/run_second_pass.py /abs/path/to/results/run_name
"""

import argparse
import os
import pathlib
import sys

base_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base_dir))

from dotenv import load_dotenv

load_dotenv(dotenv_path=str(base_dir / ".env"))


def main():
    p = argparse.ArgumentParser(description="Second-pass self-eval review (reasoning + re-score)")
    p.add_argument(
        "run_dir",
        type=str,
        help="Path to a benchmark run directory (contains self_eval_failed.jsonl)",
    )
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    failed = os.path.join(run_dir, "self_eval_failed.jsonl")
    if not os.path.isfile(failed):
        print(f"No file: {failed}")
        print("Run benchmark with benchmark.self_eval: true or --self-eval first.")
        sys.exit(1)

    from toolkit.second_pass_review import process_failed_jsonl

    n = process_failed_jsonl(run_dir)
    out = os.path.join(run_dir, "self_eval_second_pass.jsonl")
    print(f"Processed {n} record(s) -> {out}")


if __name__ == "__main__":
    main()
