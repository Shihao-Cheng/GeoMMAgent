# toolkit/second_pass_review.py
# 二次评判：推理式分析首次失败原因 + 对修订结论再做一轮四维度自评（最多一次）

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

from .evaluation_metrics import run_self_evaluation_metrics, _extract_json_object

SECOND_PASS_REASONING_PROMPT = """You are the Reasoning Agent on a SECOND-REVIEW pass (only one retry allowed in the pipeline).
The first automated self-evaluation (4 dimensions: logic, spatial reasoning, domain validity, accuracy) did NOT all pass.

You receive:
- The multiple-choice question and options (if any)
- The full agent execution trace
- The pipeline's final answer
- The JSON of the first self-evaluation (which dimensions failed and why)

Tasks:
1. **failure_analysis**: Explain likely causes linking failed dimensions to the trace (be specific, cite trace).
2. **revised_answer_explanation**: Re-analyze the image (if provided) and question; give your best reasoning.
3. **recommended_option**: For A/B/C/D questions output exactly one letter; else empty string.

Output ONLY valid JSON (no markdown):
{
  "failure_analysis": "string",
  "revised_answer_explanation": "string",
  "recommended_option": "A"
}
"""


def _openai_client():
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        return None, "QWEN_API_KEY not set"
    from openai import OpenAI

    base_url = os.environ.get(
        "QWEN_API_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return OpenAI(api_key=api_key, base_url=base_url), None


def _image_to_base64_url(image_path: str) -> str:
    import base64

    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext in (".png", "") else "image/jpeg"
    return f"data:{mime};base64,{data}"


def run_reasoning_second_review(
    question: str,
    options: Optional[Dict],
    trace_text: str,
    final_answer: str,
    first_eval: Dict[str, Any],
    image_path: Optional[str] = None,
    model: str = "qwen-vl-max",
) -> Dict[str, Any]:
    """单次推理复核：返回 failure_analysis + revised explanation + recommended_option。"""
    client, err = _openai_client()
    if err:
        return {"error": err}

    opt_str = ""
    if options:
        opt_str = "\n".join(f"{k}: {v}" for k, v in options.items())

    user_text = (
        "## Question\n"
        f"{question}\n\n"
        "## Options\n"
        f"{opt_str}\n\n"
        "## Agent trace\n"
        f"{trace_text}\n\n"
        "## Pipeline final answer\n"
        f"{final_answer}\n\n"
        "## First self-evaluation JSON\n"
        f"{json.dumps(first_eval, ensure_ascii=False, indent=2)}\n\n"
        "Produce the JSON specified in your system instructions."
    )

    if image_path and os.path.isfile(image_path):
        content = [
            {"type": "image_url", "image_url": {"url": _image_to_base64_url(image_path)}},
            {"type": "text", "text": user_text},
        ]
    else:
        content = user_text

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SECOND_PASS_REASONING_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as e:
        return {"error": str(e)}

    parsed = _extract_json_object(raw)
    if parsed is None:
        return {"error": "failed to parse reasoning JSON", "raw_response": raw[:4000]}
    return parsed


def build_second_pass_candidate(reasoning: Dict[str, Any]) -> str:
    """把二次推理结果拼成「候选答案」字符串供第二轮四维度评测。"""
    expl = str(reasoning.get("revised_answer_explanation", "")).strip()
    opt = str(reasoning.get("recommended_option", "")).strip().upper()
    if opt and re.match(r"^[A-D]$", opt):
        return f"{expl}\n【Recommended: {opt}】"
    return expl or str(reasoning.get("failure_analysis", ""))


def run_second_pass_review(
    failed_record: Dict[str, Any],
    model_eval: str = "qwen-vl-max",
) -> Dict[str, Any]:
    """
    对 self_eval_failed.jsonl 中的一条记录做完整二次评判。

    流程：推理复核 → 用修订候选答案再跑一轮 run_self_evaluation_metrics。
    返回结构写入 self_eval_second_pass.jsonl。
    """
    q = failed_record.get("question", "")
    trace_text = failed_record.get("trace_text", "")
    final_answer = failed_record.get("final_answer", "")
    first_eval = failed_record.get("first_eval") or {}
    image_path = failed_record.get("image_path") or None
    options = failed_record.get("options")

    reasoning = run_reasoning_second_review(
        question=q,
        options=options,
        trace_text=trace_text,
        final_answer=final_answer,
        first_eval=first_eval,
        image_path=image_path,
        model=model_eval,
    )

    out: Dict[str, Any] = {
        "id": failed_record.get("id"),
        "run_dir": failed_record.get("run_dir"),
        "first_eval": first_eval,
        "reasoning_second_pass": reasoning,
    }

    if reasoning.get("error"):
        out["second_eval"] = None
        out["still_failed"] = True
        out["note"] = "reasoning step failed"
        return out

    candidate2 = build_second_pass_candidate(reasoning)
    se_str = run_self_evaluation_metrics(
        question=q,
        execution_trace=trace_text,
        candidate_answer=candidate2,
        image_path=image_path,
        model=model_eval,
    )
    try:
        second_eval = json.loads(se_str)
    except json.JSONDecodeError:
        second_eval = {"error": "parse", "raw": se_str[:2000]}

    out["second_pass_candidate_answer"] = candidate2
    out["second_eval"] = second_eval

    passed = False
    if isinstance(second_eval, dict) and not second_eval.get("error"):
        passed = bool(second_eval.get("overall_pass"))

    out["still_failed"] = not passed
    out["retry_exhausted"] = True
    return out


def process_failed_jsonl(run_dir: str) -> int:
    """
    读取 run_dir/self_eval_failed.jsonl，逐条二次评判，追加到 self_eval_second_pass.jsonl。

    Returns:
        处理条数。
    """
    from .self_eval_queue import SECOND_PASS_JSONL, append_jsonl, load_failed_jsonl

    rows = load_failed_jsonl(run_dir)
    n = 0
    for rec in rows:
        out = run_second_pass_review(rec)
        append_jsonl(run_dir, SECOND_PASS_JSONL, out)
        n += 1
    return n
