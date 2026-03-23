# toolkit/self_eval_queue.py
# 自评估未通过样本的队列（JSONL）与 trace 文本格式化

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

FAILED_JSONL = "self_eval_failed.jsonl"
SECOND_PASS_JSONL = "self_eval_second_pass.jsonl"


def format_trace_text(record: Dict[str, Any]) -> str:
    """从 benchmark trace record 拼成供自评/复核用的纯文本。"""
    lines: List[str] = []
    for step in record.get("agent_trace") or []:
        name = step.get("agent_name", "?")
        sn = step.get("step", "?")
        lines.append(f"--- {name} (step {sn}) ---")
        lines.append(f"Subtask: {step.get('subtask', '')}")
        toc = step.get("trace_only_context")
        if toc:
            lines.append(
                "Coordinator context (not sent to model):\n" + str(toc)
            )
        lines.append(f"Output: {step.get('output', '')}")
        lines.append("")
    return "\n".join(lines).strip()


def self_eval_should_queue(first_eval: Dict[str, Any]) -> bool:
    """是否应写入待二次评判队列（API 错误、未跑评的不入队）。"""
    if first_eval.get("error"):
        return False
    if first_eval.get("overall_pass") is None:
        return False
    return first_eval.get("overall_pass") is False


def append_jsonl(run_dir: str, filename: str, obj: Dict[str, Any]) -> None:
    path = os.path.join(run_dir, filename)
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def build_failed_payload(
    record: Dict[str, Any],
    run_dir: str,
    first_eval: Dict[str, Any],
) -> Dict[str, Any]:
    """单行 self_eval_failed.jsonl 结构。"""
    return {
        "id": record.get("id"),
        "question": record.get("question"),
        "options": record.get("options"),
        "ground_truth": record.get("ground_truth"),
        "prediction": record.get("prediction"),
        "image_path": record.get("image_path"),
        "final_answer": record.get("final_answer"),
        "trace_text": format_trace_text(record),
        "first_eval": first_eval,
        "run_dir": run_dir,
    }


def apply_post_hoc_self_eval(
    results: List[Dict[str, Any]],
    run_dir: str,
    enabled: bool,
) -> None:
    """
    对每条成功跑完的 trace 做四维度自评，写回 record['self_eval'] 并重写 traces/<id>.json；
    未通过则追加 self_eval_failed.jsonl。
    """
    if not enabled:
        return

    from toolkit.evaluation_metrics import run_self_evaluation_metrics

    traces_dir = os.path.join(run_dir, "traces")
    for r in results:
        if r.get("error"):
            continue
        tt = format_trace_text(r)
        se_str = run_self_evaluation_metrics(
            r.get("question", ""),
            tt,
            r.get("final_answer", ""),
            r.get("image_path"),
        )
        try:
            se_obj = json.loads(se_str)
        except json.JSONDecodeError:
            se_obj = {"error": "json_parse", "raw": se_str[:500]}
        r["self_eval"] = se_obj
        tid = r.get("id")
        if tid is not None:
            p = os.path.join(traces_dir, f"{tid}.json")
            if os.path.isdir(traces_dir):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(r, f, ensure_ascii=False, indent=2)
        if self_eval_should_queue(se_obj):
            append_jsonl(run_dir, FAILED_JSONL, build_failed_payload(r, run_dir, se_obj))


def load_failed_jsonl(run_dir: str) -> List[Dict[str, Any]]:
    path = os.path.join(run_dir, FAILED_JSONL)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
