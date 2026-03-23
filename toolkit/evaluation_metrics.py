# toolkit/evaluation_metrics.py
# 自评估四维度指标（与论文叙事一致，可自动跑分）
#
# Logic: Reasonable?
# Spatial Reasoning: adequate?
# Domain Validity: valid?
# Accuracy: Reliable?

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

# ── 指标定义（供文档 / prompt 复用）────────────────────────────

SELF_EVAL_DIMENSIONS: Dict[str, str] = {
    "logic": "Logic — Is the reasoning chain reasonable and internally consistent?",
    "spatial_reasoning": (
        "Spatial reasoning — For spatial tasks: is layout/location reasoning sound? "
        "For non-spatial tasks: true if N/A or no spatial error."
    ),
    "domain_validity": (
        "Domain validity — Are claims consistent with remote sensing / geoscience knowledge?"
    ),
    "accuracy": (
        "Accuracy / reliability — Is the candidate answer reliable given evidence in the trace?"
    ),
}

EVALUATOR_SYSTEM_PROMPT = """You are a strict evaluator for geoscience and remote sensing QA pipelines.
You must score the candidate answer using the execution trace (and image if provided) on EXACTLY four dimensions:

1. logic.reasonable — Reasoning chain coherent, no fatal contradictions.
2. spatial_reasoning.adequate — Spatial/layout reasoning correct when relevant; otherwise true if N/A.
3. domain_validity.valid — No obvious domain nonsense or contradictions with RS/geoscience facts in the trace.
4. accuracy.reliable — Answer is well supported by the trace (not hallucinated beyond evidence).

Output ONLY valid JSON (no markdown fences):
{
  "logic": {"reasonable": true, "note": "short"},
  "spatial_reasoning": {"adequate": true, "note": "short"},
  "domain_validity": {"valid": true, "note": "short"},
  "accuracy": {"reliable": true, "note": "short"},
  "overall_pass": true,
  "summary": "one sentence"
}
Set overall_pass to true only if logic.reasonable, spatial_reasoning.adequate, domain_validity.valid, and accuracy.reliable are all true.
"""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _image_to_base64_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    ext = os.path.splitext(image_path)[1].lower()
    mime = "image/png" if ext in (".png", "") else "image/jpeg"
    return f"data:{mime};base64,{data}"


def run_self_evaluation_metrics(
    question: str,
    execution_trace: str,
    candidate_answer: str,
    image_path: Optional[str] = None,
    model: str = "qwen-vl-max",
) -> str:
    """
    调用多模态 LLM，对 trace + 候选答案打四维度分数，返回 JSON 字符串。

    环境变量：QWEN_API_KEY（与项目其它模块一致），可选 QWEN_API_BASE_URL。
    """
    api_key = os.environ.get("QWEN_API_KEY", "")
    if not api_key:
        return json.dumps(
            {
                "error": "QWEN_API_KEY not set",
                "logic": {"reasonable": None, "note": "skipped"},
                "spatial_reasoning": {"adequate": None, "note": "skipped"},
                "domain_validity": {"valid": None, "note": "skipped"},
                "accuracy": {"reliable": None, "note": "skipped"},
                "overall_pass": None,
                "summary": "API key missing; cannot run metrics.",
            },
            ensure_ascii=False,
            indent=2,
        )

    base_url = os.environ.get(
        "QWEN_API_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    try:
        from openai import OpenAI
    except ImportError as e:
        return json.dumps({"error": f"openai not installed: {e}"}, ensure_ascii=False)

    client = OpenAI(api_key=api_key, base_url=base_url)

    user_text = (
        "## Question\n"
        f"{question}\n\n"
        "## Execution trace (agent outputs)\n"
        f"{execution_trace}\n\n"
        "## Candidate answer\n"
        f"{candidate_answer}\n\n"
        "Evaluate with the four dimensions and return ONLY the JSON specified in your instructions."
    )

    if image_path and os.path.isfile(image_path):
        content_parts = [
            {"type": "image_url", "image_url": {"url": _image_to_base64_url(image_path)}},
            {"type": "text", "text": user_text},
        ]
    else:
        content_parts = user_text

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                {"role": "user", "content": content_parts},
            ],
            temperature=0,
        )
        raw = completion.choices[0].message.content or ""
    except Exception as e:
        return json.dumps(
            {"error": str(e), "raw": ""},
            ensure_ascii=False,
            indent=2,
        )

    parsed = _extract_json_object(raw)
    if parsed is None:
        return json.dumps(
            {
                "error": "failed to parse JSON from model",
                "raw_response": raw[:4000],
            },
            ensure_ascii=False,
            indent=2,
        )

    # 规范化 overall_pass
    try:
        lr = bool(parsed.get("logic", {}).get("reasonable"))
        sr = bool(parsed.get("spatial_reasoning", {}).get("adequate"))
        dv = bool(parsed.get("domain_validity", {}).get("valid"))
        ar = bool(parsed.get("accuracy", {}).get("reliable"))
        parsed["overall_pass"] = bool(lr and sr and dv and ar)
    except Exception:
        pass

    parsed["_metric_version"] = "geoagent_self_eval_v1"
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def evaluate_trace_with_metrics(
    question: str,
    agent_trace_and_outputs: str,
    candidate_answer: str,
    image_path: str = "",
) -> str:
    """
    FunctionTool 友好签名：空字符串表示无图。

    Returns:
        JSON 字符串（四维度 + overall_pass + summary）。
    """
    ip = image_path.strip() or None
    return run_self_evaluation_metrics(
        question=question,
        execution_trace=agent_trace_and_outputs,
        candidate_answer=candidate_answer,
        image_path=ip,
    )


@dataclass
class SelfEvalMetricSchema:
    """四维度布尔结果（解析 JSON 后可选使用）。"""

    logic_reasonable: bool
    spatial_adequate: bool
    domain_valid: bool
    accuracy_reliable: bool
    overall_pass: bool
    summary: str = ""

    @classmethod
    def from_json_dict(cls, d: Dict[str, Any]) -> Optional["SelfEvalMetricSchema"]:
        try:
            return cls(
                logic_reasonable=bool(d["logic"]["reasonable"]),
                spatial_adequate=bool(d["spatial_reasoning"]["adequate"]),
                domain_valid=bool(d["domain_validity"]["valid"]),
                accuracy_reliable=bool(d["accuracy"]["reliable"]),
                overall_pass=bool(d.get("overall_pass", False)),
                summary=str(d.get("summary", "")),
            )
        except Exception:
            return None


def format_metrics_checklist(d: Dict[str, Any]) -> str:
    """人类可读 checklist（Logic ✓ 形式）。"""
    try:
        lg = d.get("logic") or {}
        sp = d.get("spatial_reasoning") or {}
        dv = d.get("domain_validity") or {}
        ac = d.get("accuracy") or {}

        def chk(v: Any) -> str:
            if v is True:
                return "✓"
            if v is False:
                return "✗"
            return "?"

        op = d.get("overall_pass")
        overall_s = (
            "PASS" if op is True else "FAIL" if op is False else "N/A"
        )
        lines = [
            f"Logic: Reasonable? {chk(lg.get('reasonable'))}",
            f"Spatial Reasoning? {chk(sp.get('adequate'))}",
            f"Domain Validity? {chk(dv.get('valid'))}",
            f"Accuracy: Reliable? {chk(ac.get('reliable'))}",
            f"Overall: {overall_s}",
        ]
        if d.get("summary"):
            lines.append(f"Summary: {d['summary']}")
        return "\n".join(lines)
    except Exception:
        return json.dumps(d, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 无 API 时仍应能跑通（返回缺 key 的 JSON）；有 key 时跑完整评测
    demo = evaluate_trace_with_metrics(
        question="What land cover dominates?",
        agent_trace_and_outputs="ClsAgent: predicted forest 0.9 confidence.",
        candidate_answer="Forest is the dominant class.",
        image_path="",
    )
    print(demo)
    print("--- checklist ---")
    try:
        obj = json.loads(demo)
        print(format_metrics_checklist(obj))
    except Exception:
        pass
