"""
GeoMMAgent Demo：展示 coordinator + exec_agents + toolkit 的完整协作流程。

功能：
  1. AgentCoordinator     — 注册专业 exec_agents，分发并执行
  2. 动态 prompt 替换     — 运行时修改规划提示词

运行（在 GeoMMAgent 仓库根目录）：
  python run/run_geomm.py --single "..." --image path/to/image.png

  # 按题号单测（jsonl 或 configs 中 parquet）
  python run/run_geomm.py --id 21 [--jsonl datasets/val/query.jsonl]

  # 加载 benchmark parquet 并逐条跑 pipeline
  python run/run_geomm.py --bench [parquet路径] [--limit N]
"""

import json
import re
import sys
import pathlib
from dotenv import load_dotenv

# ── 路径设置 ────────────────────────────────────
base_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))  # 确保 coordinator/ exec_agents/ toolkit/ 可导入

env_path = base_dir / ".env"
load_dotenv(dotenv_path=str(env_path))

# ── 导入核心模块 ────────────────────────────────
from camel.logger import set_log_level

from coordinator import AgentCoordinator, RunTrace
from configs import load_config, build_agents_from_config, create_model_from_config
from toolkit import (
    load_benchmark,
    get_benchmark_sample_by_id,
    get_benchmark_sample_from_jsonl,
)

set_log_level("WARNING")

# Global config, loaded once in main()
_cfg = {}


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────

def create_coordinator_model():
    coord_cfg = _cfg.get("coordinator", {})
    return create_model_from_config(
        model_type=coord_cfg.get("model", "qwen-vl-max"),
        temperature=coord_cfg.get("temperature", 0),
    )


# ────────────────────────────────────────────────
# Benchmark 批量模式
# ────────────────────────────────────────────────

def run_benchmark(model, parquet_path: str = None, limit: int = None):
    """从 parquet 加载 benchmark 数据，逐条跑完整 pipeline，保存结构化 trace"""
    import json
    import os
    import re
    from datetime import datetime

    bench_cfg = _cfg.get("benchmark", {})
    parquet_path = parquet_path or bench_cfg.get("parquet_path")

    samples = load_benchmark(
        parquet_path=parquet_path,
        limit=limit,
    )
    print(f"\n已加载 {len(samples)} 条 benchmark 样本")

    coord = AgentCoordinator(model=model)
    coord.register_workers(build_agents_from_config(_cfg))

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = str(base_dir / "results" / run_name)
    traces_dir = os.path.join(run_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    print(f"Results -> {run_dir}")

    def _extract(text: str) -> str:
        if not text:
            return ""
        t = text.strip()
        if len(t) == 1 and t.upper() in "ABCD":
            return t.upper()
        m = re.search(r"\b([A-D])\b", t.upper())
        return m.group(1) if m else ""

    correct, total = 0, len(samples)
    all_records = []

    for i, sample in enumerate(samples):
        print(f"\n{'─'*50}")
        print(f"[{i+1}/{total}] index={sample.index}  answer={sample.answer}")

        try:
            trace = coord.run_with_trace(
                sample.prompt,
                image_path=sample.image_path,
                answer_text=sample.answer,
            )
            pred = _extract(trace.final_answer)
            is_ok = pred == sample.answer
            correct += is_ok

            record = {
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
                "is_correct": is_ok,
                "evidence_images": trace.evidence_images,
            }

            flag = "OK" if is_ok else "FAIL"
            agents_used = " -> ".join(s.agent_name for s in trace.agent_steps)
            print(f"  Agents: {agents_used}")
            print(f"  {flag}  pred={pred}  gt={sample.answer}")
        except Exception as e:
            record = {
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
                "error": str(e),
            }
            print(f"  ERROR: {e}")

        all_records.append(record)
        trace_path = os.path.join(traces_dir, f"{sample.index}.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

    errors = sum(1 for r in all_records if r.get("error"))
    wrong_ids = [r["id"] for r in all_records if not r["is_correct"]]
    error_ids = [r["id"] for r in all_records if r.get("error")]

    all_records.sort(key=lambda r: r["id"])

    from toolkit.self_eval_queue import apply_post_hoc_self_eval

    apply_post_hoc_self_eval(
        all_records,
        run_dir,
        bool(bench_cfg.get("self_eval", False)),
    )

    agents_path = os.path.join(run_dir, "agents.jsonl")
    with open(agents_path, "w", encoding="utf-8") as f:
        for r in all_records:
            agents = [step["agent_name"] for step in r.get("agent_trace", [])]
            line = json.dumps({"id": r["id"], "agents": agents}, ensure_ascii=False)
            f.write(line + "\n")

    se_failed = 0
    if any("self_eval" in r for r in all_records):
        se_failed = sum(
            1
            for r in all_records
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
        "model": "qwen-vl-max",
        "wrong_ids": sorted(wrong_ids),
        "error_ids": sorted(error_ids),
        "self_eval_failed_count": se_failed,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"  Accuracy: {correct}/{total} = {summary['accuracy']}%")
    print(f"  Errors:   {errors}")
    print(f"  Summary:  {run_dir}/summary.json")
    print(f"  Agents:   {run_dir}/agents.jsonl")
    print(f"  Traces:   {run_dir}/traces/")
    if bench_cfg.get("self_eval"):
        print(f"  Self-eval failed queue: {run_dir}/self_eval_failed.jsonl")
        print(f"  Second pass: python run/run_second_pass.py {run_dir}")


# ────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────

def _safe_step_filename(agent_name: str) -> str:
    s = re.sub(r"[^\w\-]+", "_", agent_name)
    return (s.strip("_") or "agent")


def write_demo_trace_json(
    trace: RunTrace,
    out_dir: pathlib.Path,
    *,
    sample_id: int | None = None,
    ground_truth: str | None = None,
) -> None:
    """
    将单题 demo 的调度与各 agent 的输入/输出写入 out_dir：
      run.json, trace_full.json, steps/{02}_{AgentName}.json
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_dir = out_dir / "steps"
    steps_dir.mkdir(exist_ok=True)

    run_payload = {
        "sample_id": sample_id,
        "ground_truth": ground_truth,
        "image_path": trace.image_path,
        "original_task": trace.original_task,
        "dispatch_plan": trace.dispatch_plan,
        "final_answer": trace.final_answer,
        "evidence_images": trace.evidence_images,
    }
    with (out_dir / "run.json").open("w", encoding="utf-8") as f:
        json.dump(run_payload, f, ensure_ascii=False, indent=2)

    with (out_dir / "trace_full.json").open("w", encoding="utf-8") as f:
        json.dump(trace.to_dict(), f, ensure_ascii=False, indent=2)

    for s in trace.agent_steps:
        fname = f"{s.step:02d}_{_safe_step_filename(s.agent_name)}.json"
        step_payload = {
            "step": s.step,
            "agent_index": s.agent_index,
            "agent_name": s.agent_name,
            "agent_description": s.agent_description,
            "subtask": s.subtask,
            "input_text": s.input_text,
            "trace_only_context": s.trace_only_context,
            "attached_image_paths": s.attached_image_paths,
            "output": s.output,
        }
        with (steps_dir / fname).open("w", encoding="utf-8") as f:
            json.dump(step_payload, f, ensure_ascii=False, indent=2)


def run_single(
    model,
    question: str,
    image_path: str = None,
    *,
    ground_truth: str | None = None,
    export_dir: str | pathlib.Path | None = None,
    sample_id: int | None = None,
):
    """单条问题测试，输出完整 trace"""
    coord = AgentCoordinator(model=model)
    coord.register_workers(build_agents_from_config(_cfg))

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    if image_path:
        print(f"Image:    {image_path}")
    if ground_truth:
        print(f"Answer (gt): {ground_truth}")
    print(f"{'='*60}\n")

    try:
        trace = coord.run_with_trace(
            question,
            image_path=image_path,
            answer_text=ground_truth,
        )
        print(f"Dispatch plan ({len(trace.dispatch_plan)} steps):")
        for step in trace.dispatch_plan:
            print(f"  {step}")
        print()
        for s in trace.agent_steps:
            print(f"--- Step {s.step}: {s.agent_name} ---")
            print(f"  Subtask: {s.subtask}")
            print(f"  Output:  {s.output[:1024]}{'...' if len(s.output) > 1024 else ''}")
            print()
        print(f"{'='*60}")
        print(f"Final answer: {trace.final_answer}")
        print(f"{'='*60}")
        if export_dir:
            write_demo_trace_json(
                trace,
                pathlib.Path(export_dir),
                sample_id=sample_id,
                ground_truth=ground_truth,
            )
            print(f"JSON trace -> {pathlib.Path(export_dir).resolve()}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def _parse_args():
    """简单的参数解析，不引入额外依赖"""
    args = sys.argv[1:]

    if "--id" in args:
        idx = args.index("--id")
        if idx + 1 >= len(args):
            print("Error: --id 后面必须跟题号（整数）")
            sys.exit(1)
        sample_id = int(args[idx + 1])
        jsonl_path = None
        if "--jsonl" in args:
            ji = args.index("--jsonl")
            jsonl_path = args[ji + 1] if ji + 1 < len(args) else None
        images_root = None
        if "--images-root" in args:
            ki = args.index("--images-root")
            images_root = args[ki + 1] if ki + 1 < len(args) else None
        return "single_id", (sample_id, jsonl_path, images_root), None

    if "--single" in args:
        idx = args.index("--single")
        question = args[idx + 1] if idx + 1 < len(args) else None
        image_path = None
        if "--image" in args:
            img_idx = args.index("--image")
            image_path = args[img_idx + 1] if img_idx + 1 < len(args) else None
        if not question:
            print("Error: --single 后面必须跟问题内容")
            sys.exit(1)
        return "single", (question, image_path), None

    if "--bench" in args:
        args.remove("--bench")
        limit = None
        if "--limit" in args:
            idx = args.index("--limit")
            limit = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        parquet_path = args[0] if args else None
        return "bench", parquet_path, limit

    return "help", None, None


def _resolve_sample_by_id(cfg: dict, sample_id: int, jsonl_path: str | None, images_root: str | None):
    """按题号加载一条 BenchmarkSample：优先 --jsonl，否则使用 benchmark.parquet_path。"""
    from pathlib import Path

    if jsonl_path:
        jp = Path(jsonl_path)
        if not jp.is_absolute():
            jp = base_dir / jp
        root = images_root or jp.parent
        if not Path(root).is_absolute():
            root = base_dir / root
        return get_benchmark_sample_from_jsonl(str(jp), root, sample_id)

    bench_cfg = cfg.get("benchmark", {}) or {}
    parquet_path = bench_cfg.get("parquet_path")
    if not parquet_path:
        raise SystemExit(
            "未配置 benchmark.parquet_path，请用 --jsonl datasets/val/query.jsonl 指定题库"
        )
    pq = Path(parquet_path)
    if not pq.is_absolute():
        pq = base_dir / pq
    if not pq.is_file():
        raise SystemExit(
            f"找不到 parquet: {pq}。请检查 configs 中的 parquet_path，或使用 --jsonl"
        )
    return get_benchmark_sample_by_id(str(pq), sample_id)


def main():
    global _cfg
    mode, arg1, limit = _parse_args()

    _cfg = load_config()

    try:
        model = create_coordinator_model()
    except Exception as e:
        print(f"模型初始化失败，请配置 API KEY: {e}")
        return

    if mode == "bench":
        print("\nGeoMMAgent — Benchmark 模式")
        run_benchmark(model, parquet_path=arg1, limit=limit)
        print("\n完成\n")
    elif mode == "single_id":
        sample_id, jsonl_path, images_root = arg1
        try:
            sample = _resolve_sample_by_id(_cfg, sample_id, jsonl_path, images_root)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"\nGeoMMAgent — 单题 id={sample_id}")
        export_dir = base_dir / "test" / f"id{sample_id}"
        run_single(
            model,
            sample.prompt,
            sample.image_path,
            ground_truth=sample.answer or None,
            export_dir=export_dir,
            sample_id=sample_id,
        )
    elif mode == "single":
        question, image_path = arg1
        run_single(model, question, image_path)
    else:
        print("Usage:")
        print("  按题号单测: python run/run_geomm.py --id <题号>")
        print("              [--jsonl datasets/val/query.jsonl] [--images-root datasets/val]")
        print("              （无 --jsonl 时使用 configs 里 benchmark.parquet_path）")
        print("  单条测试:   python run/run_geomm.py --single '问题内容' [--image /path/to/image.jpg]")
        print("  批量测试:   python run/run_geomm.py --bench [parquet_path] [--limit N]")


if __name__ == "__main__":
    main()
