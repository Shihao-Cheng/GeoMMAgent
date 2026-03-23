"""
GeoMMAgent Parallel Benchmark Evaluation

Evaluates GeoMMAgent on GeoMMBench with multi-threaded parallelism.
Each worker thread owns an independent AgentCoordinator to avoid shared-state
issues.

Results are saved to:
    results/<run_name>/
        summary.json       — overall accuracy, timing, config
        agents.jsonl       — one line per sample: {"id": ..., "agents": [...]}
        traces/
            <id>.json      — per-sample full execution trace
        self_eval_failed.jsonl — optional; samples whose 4-dim self-eval failed (for second pass)

Usage:
    python run/run_benchmark_parallel.py /path/to/validation.parquet

    python run/run_benchmark_parallel.py /path/to/validation.parquet \\
        --workers 10 --limit 50 --run-name my_experiment
"""

import argparse
import json
import os
import re
import sys
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────
base_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
load_dotenv(dotenv_path=str(base_dir / ".env"))

from camel.logger import set_log_level

from coordinator import AgentCoordinator, RunTrace
from configs import load_config, build_agents_from_config, create_model_from_config
from toolkit import load_benchmark, BenchmarkSample
from toolkit.self_eval_queue import apply_post_hoc_self_eval

set_log_level("WARNING")

_thread_local = threading.local()
_global_cfg = {}


def get_coordinator() -> AgentCoordinator:
    """Return a thread-local AgentCoordinator (created once per thread)."""
    if not hasattr(_thread_local, "coordinator"):
        cfg = _global_cfg
        coord_cfg = cfg.get("coordinator", {})
        coord_model = create_model_from_config(
            model_type=coord_cfg.get("model", "qwen-vl-max"),
            temperature=coord_cfg.get("temperature", 0),
        )
        coord = AgentCoordinator(model=coord_model)
        coord.register_workers(build_agents_from_config(cfg))
        _thread_local.coordinator = coord
    return _thread_local.coordinator


def extract_option(text: str) -> str:
    """Extract a single option letter (A-D) from model output."""
    if not text:
        return ""
    text = text.strip()
    if len(text) == 1 and text.upper() in "ABCD":
        return text.upper()
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else ""


def _trace_to_record(
    sample: BenchmarkSample,
    trace: RunTrace,
    pred: str,
    elapsed: float,
) -> Dict:
    """Build a serializable record dict from a RunTrace and sample metadata."""
    return {
        "id": sample.index,
        "question": sample.question,
        "options": sample.options,
        "ground_truth": sample.answer,
        "image_path": sample.image_path,
        "category": sample.category,
        "source": sample.source,
        "dispatch_plan": trace.dispatch_plan,
        "agent_trace": [
            {
                "step": s.step,
                "agent_index": s.agent_index,
                "agent_name": s.agent_name,
                "subtask": s.subtask,
                "input_text": s.input_text,
                "trace_only_context": s.trace_only_context,
                "output": s.output,
            }
            for s in trace.agent_steps
        ],
        "final_answer": trace.final_answer,
        "prediction": pred,
        "is_correct": pred == sample.answer,
        "elapsed_s": elapsed,
        "evidence_images": trace.evidence_images,
    }


def _error_record(sample: BenchmarkSample, error: str, elapsed: float) -> Dict:
    return {
        "id": sample.index,
        "question": sample.question,
        "options": sample.options,
        "ground_truth": sample.answer,
        "image_path": sample.image_path,
        "category": sample.category,
        "source": sample.source,
        "dispatch_plan": [],
        "agent_trace": [],
        "final_answer": "",
        "prediction": "",
        "is_correct": False,
        "elapsed_s": elapsed,
        "error": error,
    }


def process_sample(sample: BenchmarkSample) -> Dict:
    """Run the full agent pipeline on a single sample with trace."""
    t0 = time.time()
    coord = get_coordinator()
    try:
        trace = coord.run_with_trace(
            sample.prompt,
            image_path=sample.image_path,
            answer_text=sample.answer,
        )
        pred = extract_option(trace.final_answer)
        return _trace_to_record(sample, trace, pred, round(time.time() - t0, 2))
    except Exception as e:
        return _error_record(sample, str(e), round(time.time() - t0, 2))


def save_trace(record: Dict, traces_dir: str):
    """Save a single sample trace immediately after it completes."""
    path = os.path.join(traces_dir, f"{record['id']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def _agent_names_from_record(record: Dict) -> List[str]:
    """Ordered list of agent class names invoked for this sample (from agent_trace)."""
    return [step["agent_name"] for step in record.get("agent_trace", [])]


def save_summary(results: List[Dict], run_dir: str, num_workers: int) -> Dict:
    """Write the final summary.json after all samples are done."""
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    errors = sum(1 for r in results if r.get("error"))
    avg_time = sum(r["elapsed_s"] for r in results) / total if total else 0

    wrong_ids = [r["id"] for r in results if not r["is_correct"]]
    error_ids = [r["id"] for r in results if r.get("error")]

    se_failed = 0
    if any("self_eval" in r for r in results):
        se_failed = sum(
            1
            for r in results
            if isinstance(r.get("self_eval"), dict)
            and r["self_eval"].get("overall_pass") is False
            and not r["self_eval"].get("error")
        )

    summary = {
        "run_dir": run_dir,
        "timestamp": datetime.now().isoformat(),
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total else 0,
        "errors": errors,
        "avg_elapsed_s": round(avg_time, 2),
        "model": "qwen-vl-max",
        "workers": num_workers,
        "wrong_ids": sorted(wrong_ids),
        "error_ids": sorted(error_ids),
        "self_eval_failed_count": se_failed,
    }
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    agents_path = os.path.join(run_dir, "agents.jsonl")
    with open(agents_path, "w", encoding="utf-8") as f:
        for r in results:
            line = json.dumps(
                {"id": r["id"], "agents": _agent_names_from_record(r)},
                ensure_ascii=False,
            )
            f.write(line + "\n")

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="GeoMMAgent parallel benchmark evaluation",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (default: configs/GeoMMBench.yaml)",
    )
    parser.add_argument(
        "--parquet", type=str, default=None,
        help="Path to GeoMMBench parquet file (overrides config)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (overrides config)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only evaluate the first N samples (overrides config)",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Name for this run (default: timestamp)",
    )
    parser.add_argument(
        "--self-eval",
        action="store_true",
        help="After run: 4-dim self-eval per sample; failures -> self_eval_failed.jsonl",
    )
    return parser.parse_args()


def main():
    global _global_cfg
    args = parse_args()

    cfg = load_config(args.config)
    _global_cfg = cfg
    bench_cfg = cfg.get("benchmark", {})

    parquet_path = args.parquet or bench_cfg.get("parquet_path")
    if not parquet_path:
        print("No parquet path specified (use --parquet or set in config)")
        sys.exit(1)
    num_workers = args.workers or bench_cfg.get("workers", 5)
    limit = args.limit if args.limit is not None else bench_cfg.get("limit")

    try:
        coord_cfg = cfg.get("coordinator", {})
        create_model_from_config(
            model_type=coord_cfg.get("model", "qwen-vl-max"),
            temperature=coord_cfg.get("temperature", 0),
        )
    except Exception as e:
        print(f"Model init failed — check API keys: {e}")
        sys.exit(1)

    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(base_dir / "results" / run_name)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)

    print(f"Loading benchmark from {parquet_path} ...")
    samples = load_benchmark(parquet_path=parquet_path, limit=limit)
    total = len(samples)
    print(f"Loaded {total} samples, launching {num_workers} workers")
    print(f"Results -> {run_dir}\n")

    results: List[Dict] = []
    correct = 0
    t_start = time.time()

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, desc="Evaluating", unit="sample")
    except ImportError:
        progress = None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(process_sample, s): s.index for s in samples
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            save_trace(result, traces_dir)

            if result["is_correct"]:
                correct += 1
            done = len(results)
            acc = correct / done * 100

            if progress:
                progress.set_postfix(acc=f"{acc:.1f}%", correct=correct)
                progress.update(1)
            else:
                tag = "OK" if result["is_correct"] else "FAIL"
                print(
                    f"[{done}/{total}] id={result['id']}  "
                    f"pred={result['prediction']}  gt={result['ground_truth']}  "
                    f"{tag}  acc={acc:.1f}%  ({result['elapsed_s']}s)"
                )

    if progress:
        progress.close()

    results.sort(key=lambda r: r["id"])

    bench_self_eval = bool(bench_cfg.get("self_eval", False)) or bool(args.self_eval)
    apply_post_hoc_self_eval(results, run_dir, bench_self_eval)

    elapsed = time.time() - t_start
    summary = save_summary(results, run_dir, num_workers)

    print(f"\n{'=' * 50}")
    print(f"  Run:       {run_name}")
    print(f"  Samples:   {summary['total']}")
    print(f"  Correct:   {summary['correct']}")
    print(f"  Accuracy:  {summary['accuracy']}%")
    print(f"  Errors:    {summary['errors']}")
    print(f"  Time:      {elapsed:.1f}s  ({summary['avg_elapsed_s']}s/sample)")
    print(f"{'=' * 50}")
    print(f"  Summary:   {run_dir}/summary.json")
    print(f"  Agents:    {run_dir}/agents.jsonl")
    print(f"  Traces:    {run_dir}/traces/")
    if bench_self_eval:
        print(f"  Self-eval: failed queue -> {run_dir}/self_eval_failed.jsonl")
        print(f"             (second pass: python run/run_second_pass.py {run_dir})")


if __name__ == "__main__":
    main()
