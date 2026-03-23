# coordinator/coordinator.py
# Two-stage pipeline: Dispatch → Execute (sequential with context).
#
# The coordinator LLM reads all registered agent descriptions, decomposes
# the task into subtasks, and assigns each subtask to the most suitable
# agent. Agents are then executed in order, with each agent receiving its
# specific subtask instruction plus all previous agents' outputs as context.
#
# Fully synchronous and thread-safe — no asyncio, compatible with
# ThreadPoolExecutor for parallel benchmark evaluation.
#
# Usage:
#   coord = AgentCoordinator(model=my_model)
#   coord.register_workers([agent.as_worker_dict() for agent in agents])
#   result = coord.run("question text", image_path="img.png")

import base64
import io
import json
import logging
import os
import pathlib
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI
from PIL import Image

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.types import RoleType

from .prompts import DISPATCH_PROMPT

logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    """Record of a single agent invocation within a pipeline run."""
    step: int
    agent_index: int
    agent_name: str
    agent_description: str
    subtask: str
    output: str
    # Full user message text (and image paths) actually sent to the model for this step.
    input_text: str = ""
    attached_image_paths: List[str] = field(default_factory=list)
    # ReasoningAgent: coordinator context aligned with other agents (original task, prior
    # outputs, subtask) recorded for trace only — not included in ``input_text``.
    trace_only_context: Optional[str] = None


@dataclass
class RunTrace:
    """Full execution trace for one coordinator run."""
    original_task: str
    dispatch_plan: List[Dict[str, Any]]
    agent_steps: List[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    image_path: Optional[str] = None
    # SearchAgent evidence entries: label, path, url (replay / debugging).
    evidence_images: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_image(image_path: Optional[str]) -> Optional[List[Image.Image]]:
    if not image_path:
        return None
    img = Image.open(image_path)
    if img.format is None:
        img.format = "PNG"
    return [img]


def _load_image_stack(
    main_path: Optional[str],
    evidence: List[Dict[str, str]],
) -> Optional[List[Image.Image]]:
    """Task image at index 0, then SearchAgent evidence images. Used as ``image_list``."""
    imgs: List[Image.Image] = []
    if main_path:
        try:
            im = Image.open(main_path)
            im.load()
            imgs.append(im)
        except Exception as e:
            logger.warning("Failed to open task image %s: %s", main_path, e)
    for ev in evidence:
        p = ev.get("path")
        if not p:
            continue
        try:
            im = Image.open(p)
            im.load()
            imgs.append(im)
        except Exception as e:
            logger.debug("Failed to open evidence image %s: %s", p, e)
    return imgs if imgs else None


def _multimodal_evidence_instruction(
    main_path: Optional[str],
    evidence: List[Dict[str, str]],
) -> str:
    if not evidence:
        return ""
    lines = [
        "Multimodal images are provided in order (same order as image_list):",
    ]
    if main_path:
        lines.append("  - Index 0: the task / question image.")
        start = 1
    else:
        start = 0
    for j, ev in enumerate(evidence):
        lab = ev.get("label", "")
        idx = start + j
        lines.append(
            f"  - Index {idx}: search evidence — label (search query) = {lab!r}",
        )
    return "\n".join(lines)


def _make_message(
    content: str,
    image_list: Optional[List[Image.Image]] = None,
) -> BaseMessage:
    return BaseMessage(
        role_name="user",
        role_type=RoleType.USER,
        meta_dict={},
        content=content,
        image_list=image_list,
        image_detail="high",
    )


def _project_root() -> pathlib.Path:
    """Repository root (parent of the ``coordinator`` package)."""
    return pathlib.Path(__file__).resolve().parents[1]


def _run_yolo_detection_for_det_step(
    image_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Run YOLO on the task image for a DetAgent execution step.

    Called from :meth:`AgentCoordinator._execute_traced` when the current
    worker is ``DetAgent``, immediately before that agent's LLM is invoked.
    Returns class aggregates and ``detections`` geometry from the tool JSON, or
    ``None`` if weights are missing or inference fails.

    Weight path: environment variable ``GEOMM_DET_YOLO_PATH``, else
    ``<project>/weights/yolo11s-obb.pt``.
    """
    if not image_path or not os.path.isfile(image_path):
        return None
    try:
        wpath = os.environ.get("GEOMM_DET_YOLO_PATH")
        if not wpath:
            wpath = str(_project_root() / "weights" / "yolo11s-obb.pt")
        if not os.path.isfile(wpath):
            logger.warning("YOLO det weights not found: %s", wpath)
            return None
        from toolkit.detection_toolkit import YOLODetectionToolkit

        tk = YOLODetectionToolkit(yolo_model_path=wpath)
        raw = tk.detect_objects_from_image_path(image_path)
        d = json.loads(raw)
        if not d.get("ok"):
            logger.warning("YOLO det inference not ok: %s", d.get("error"))
            return None
        data = d.get("data") or {}
        return {
            "per_class_counts": data.get("per_class_counts"),
            "per_class_avg_confidence": data.get("per_class_avg_confidence"),
            "total_instances": data.get("total_instances"),
            "image_size": data.get("image_size"),
            "detections": data.get("detections"),
        }
    except Exception as e:
        logger.warning("YOLO detection step failed: %s", e)
        return None


def _chat_agent_without_tools(agent: ChatAgent) -> ChatAgent:
    """Same ``system_message`` and ``model`` as ``agent``, with an empty tool list.

    Used for the DetAgent step after :func:`_run_yolo_detection_for_det_step`
    so the LLM cannot issue redundant ``detect_*`` tool calls.
    """
    return ChatAgent(
        system_message=agent.system_message,
        model=agent.model_backend,
        tools=[],
    )


def _detection_summary_block(det_stats: Dict[str, Any]) -> str:
    """Serialize detection aggregates for the DetAgent user message."""
    payload = {
        k: det_stats[k]
        for k in (
            "per_class_counts",
            "per_class_avg_confidence",
            "total_instances",
            "image_size",
            "detections",
        )
        if det_stats.get(k) is not None
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _is_stream_only_agent(agent: ChatAgent) -> bool:
    """Check if an agent's model requires streaming (e.g. qvq-max)."""
    try:
        return agent.model_backend.model_config_dict.get("stream", False)
    except Exception:
        return False


def _image_to_base64_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def _call_stream_model_direct(
    agent: ChatAgent,
    text: str,
    image_path: Optional[str] = None,
    system_prompt: Optional[str] = None,
    extra_image_paths: Optional[List[str]] = None,
) -> str:
    """
    Bypass CAMEL and call a streaming-only model (e.g. qvq-max) via OpenAI
    client directly, collecting both reasoning_content and content.
    Images: task image first, then optional evidence paths (same order as PIL path).
    """
    api_key = os.environ.get("QWEN_API_KEY", "")
    base_url = os.environ.get(
        "QWEN_API_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model_type = getattr(agent.model_backend, "model_type", "qvq-max")

    client = OpenAI(api_key=api_key, base_url=base_url)

    content_parts: list = []
    paths: List[str] = []
    if image_path:
        paths.append(image_path)
    if extra_image_paths:
        paths.extend(extra_image_paths)
    for p in paths:
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": _image_to_base64_url(p)},
        })
    content_parts.append({"type": "text", "text": text})

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})

    completion = client.chat.completions.create(
        model=str(model_type),
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={
            "enable_thinking": True,
            "thinking_budget": 8096,
        },
    )

    reasoning_content = ""
    answer_content = ""

    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if hasattr(delta, "reasoning_content") and delta.reasoning_content:
            reasoning_content += delta.reasoning_content
        if delta.content:
            answer_content += delta.content

    return reasoning_content or answer_content


class AgentCoordinator:
    """
    Two-stage coordinator: Plan then Execute.

    Stage 1 — Plan & Dispatch:
      A coordinator LLM receives the refined task and all registered agent
      descriptions, then outputs a JSON execution plan: an ordered list of
      (subtask, agent_index) pairs.

    Stage 2 — Execute:
      Each agent in the plan is called sequentially via ChatAgent.step().
      It receives its specific subtask instruction, the original image,
      and accumulated outputs from all previously executed agents.

    Extensibility:
      Register new agents via register_workers() at any time.  The
      coordinator LLM discovers them automatically through their
      WORKER_DESCRIPTION — no code changes needed for dispatch.
    """

    def __init__(self, model: BaseModelBackend):
        self.model = model
        self._workers: List[Dict[str, Any]] = []
        self._dispatcher = ChatAgent(
            system_message="You are a multi-agent task coordinator.",
            model=model,
        )

    # ── Worker registration ──────────────────────

    def register_worker(
        self, description: str, agent: ChatAgent, name: str = "",
    ) -> None:
        self._workers.append({
            "description": description,
            "agent": agent,
            "name": name or f"Agent_{len(self._workers)}",
        })

    def register_workers(self, worker_list: List[Dict[str, Any]]) -> None:
        for w in worker_list:
            self.register_worker(
                w["description"], w["agent"], w.get("name", ""),
            )

    # ── Stage 1: Dispatch ─────────────────────────

    def _find_matching_agent_index(self) -> Optional[int]:
        """
        Only match by registered class name.

        Do **not** scan descriptions for the substring \"matching\": ReasoningAgent
        (\"… for the Matching agent\"), ClsAgent (\"best-matching\"), etc. would be
        mistaken for MatchingAgent and break _ensure_matching_last ordering.
        """
        for i, w in enumerate(self._workers):
            if w.get("name") == "MatchingAgent":
                return i
        return None

    def _dispatch(
        self, refined_task: str, image_path: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ask the coordinator LLM to decompose the refined task into subtasks
        and assign each to a specific agent.

        Returns a list of {"agent": int, "subtask": str} dicts.
        Falls back to invoking all agents with the full task if parsing fails.

        Hard guarantee: if a MatchingAgent is registered, the last step in
        the plan is always the MatchingAgent.
        """
        agent_list = "\n".join(
            f"  [{i}] {w['description']}" for i, w in enumerate(self._workers)
        )
        prompt = DISPATCH_PROMPT.format(agent_list=agent_list, task=refined_task)

        image_list = _load_image(image_path)
        msg = _make_message(prompt, image_list=image_list)
        try:
            resp = self._dispatcher.step(msg)
            text = resp.msgs[0].content if resp.msgs else ""
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                plan = json.loads(match.group())
                valid = [
                    step for step in plan
                    if isinstance(step, dict)
                    and "agent" in step
                    and "subtask" in step
                    and 0 <= step["agent"] < len(self._workers)
                ]
                if valid:
                    valid = self._ensure_matching_last(valid)
                    return valid
        except Exception as e:
            logger.warning("Dispatch parsing failed: %s", e)

        return [
            {"agent": i, "subtask": refined_task}
            for i in range(len(self._workers))
        ]

    def _ensure_matching_last(
        self, plan: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Guarantee the MatchingAgent is the final step."""
        midx = self._find_matching_agent_index()
        if midx is None:
            return plan

        if plan[-1]["agent"] == midx:
            return plan

        plan = [s for s in plan if s["agent"] != midx]
        plan.append({
            "agent": midx,
            "subtask": "Match the preceding analysis to the correct option (A/B/C/D). Output a single letter.",
        })
        return plan

    # ── Stage 2: Execute ─────────────────────────

    def _execute_traced(
        self,
        plan: List[Dict[str, Any]],
        original_task: str,
        image_path: Optional[str],
        answer_text: Optional[str] = None,
    ) -> tuple:
        """
        Execute the plan sequentially and return (final_answer, agent_steps).

        Each agent receives:
          - Its assigned subtask instruction
          - The original task and image for full context
          - Accumulated outputs from all previously executed agents
          - After SearchAgent: extra evidence images (same multimodal message, ordered)

        **DetAgent:** When the active step targets ``DetAgent`` and ``image_path`` is set,
        :func:`_run_yolo_detection_for_det_step` runs *before* ``ChatAgent.step``. On
        success, the user message contains the detection summary JSON (including
        ``detections`` coordinates); no detection overlay image is attached. Optional
        SearchAgent evidence images use the same multimodal order as elsewhere. The
        agent is stepped without tools. On failure, the step falls back to the default
        path (task image + registered tools).
        """
        image_tag = f"[Image: {image_path}]\n" if image_path else ""

        agent_outputs: List[str] = []
        steps: List[AgentStep] = []
        evidence_accum: List[Dict[str, str]] = []

        for step_num, step in enumerate(plan, 1):
            idx = step["agent"]
            subtask = step["subtask"]
            trace_only_context: Optional[str] = None
            w = self._workers[idx]
            agent: ChatAgent = w["agent"]
            desc: str = w["description"]
            name: str = w.get("name", f"Agent_{idx}")

            text_block = ""
            if name == "SearchAgent":
                from exec_agents.knowledge.search_agent import (
                    run_search_evidence_pipeline,
                )
                text_block, new_evidence = run_search_evidence_pipeline(
                    agent,
                    subtask,
                    original_task,
                    image_path,
                )
                evidence_accum.extend(new_evidence)

            det_stats: Optional[Dict[str, Any]] = None
            if name == "DetAgent" and image_path:
                det_stats = _run_yolo_detection_for_det_step(image_path)

            if det_stats is not None:
                parts = [
                    "YOLO object detection has **already been run** on the task image; "
                    "do **not** call any detection tools.\n",
                    "Answer using **only** the JSON below (``detections`` includes per-instance "
                    "geometry; ``per_class_counts`` / ``total_instances`` for counts). "
                    "No detection overlay image is provided.\n",
                    "Detection summary:\n" + _detection_summary_block(det_stats),
                    f"Original task: {original_task}",
                ]
                if agent_outputs:
                    parts.append(
                        "Previous agent outputs:\n" + "\n---\n".join(agent_outputs)
                    )
                parts.append(f"Your assigned subtask: {subtask}")
                if evidence_accum:
                    parts.append(
                        _multimodal_evidence_instruction(None, evidence_accum)
                    )
                else:
                    parts.append(
                        "No images are attached for this step (text-only context from JSON)."
                    )
                pil_list = _load_image_stack(None, evidence_accum)
                agent_to_use = _chat_agent_without_tools(agent)
                attached_paths = [e.get("path", "") for e in evidence_accum if e.get("path")]
            else:
                # ReasoningAgent: user message is the MCQ template (+ optional evidence text below).
                # Subtask and prior outputs go to trace_only_context only, not into the model prompt.
                if name == "ReasoningAgent":
                    from exec_agents.reasoning.mcq_match_prompt import (
                        build_reasoning_user_message,
                    )
                    at = answer_text if answer_text is not None else ""
                    trace_only_parts = [
                        f"{image_tag}Original task: {original_task}",
                    ]
                    if agent_outputs:
                        trace_only_parts.append(
                            "Previous agent outputs:\n"
                            + "\n---\n".join(agent_outputs)
                        )
                    trace_only_parts.append(f"Your assigned subtask: {subtask}")
                    trace_only_context = "\n\n".join(trace_only_parts)
                    parts = [build_reasoning_user_message(original_task, at)]
                else:
                    parts = [f"{image_tag}Original task: {original_task}"]
                    if agent_outputs:
                        parts.append(
                            "Previous agent outputs:\n"
                            + "\n---\n".join(agent_outputs)
                        )
                    parts.append(f"Your assigned subtask: {subtask}")
                if name == "SearchAgent":
                    parts.append(text_block)
                if evidence_accum:
                    parts.append(
                        _multimodal_evidence_instruction(image_path, evidence_accum)
                    )
                pil_list = _load_image_stack(image_path, evidence_accum)
                agent_to_use = agent
                attached_paths = []
                if image_path:
                    attached_paths.append(image_path)
                attached_paths.extend(
                    e.get("path", "") for e in evidence_accum if e.get("path")
                )

            agent_input = "\n\n".join(parts)

            try:
                if _is_stream_only_agent(agent_to_use):
                    sys_prompt = getattr(agent_to_use, "system_message", None)
                    if hasattr(sys_prompt, "content"):
                        sys_prompt = sys_prompt.content
                    extra_paths = [e["path"] for e in evidence_accum]
                    if det_stats is not None:
                        output = _call_stream_model_direct(
                            agent_to_use,
                            "\n\n".join(parts),
                            image_path=None,
                            system_prompt=sys_prompt,
                            extra_image_paths=extra_paths or None,
                        )
                    else:
                        output = _call_stream_model_direct(
                            agent_to_use,
                            "\n\n".join(parts),
                            image_path=image_path,
                            system_prompt=sys_prompt,
                            extra_image_paths=extra_paths or None,
                        )
                else:
                    msg = _make_message("\n\n".join(parts), image_list=pil_list)
                    resp = agent_to_use.step(msg)
                    output = resp.msgs[0].content if resp.msgs else ""
            except Exception as e:
                output = f"[Agent error: {e}]"

            agent_outputs.append(f"[{desc[:80]}]\n{output}")
            steps.append(AgentStep(
                step=step_num,
                agent_index=idx,
                agent_name=name,
                agent_description=desc,
                subtask=subtask,
                output=output,
                input_text=agent_input,
                attached_image_paths=list(attached_paths),
                trace_only_context=trace_only_context,
            ))

        if not agent_outputs:
            return "", steps, evidence_accum
        final = agent_outputs[-1].split("\n", 1)[-1]
        return final, steps, evidence_accum

    def _execute(
        self,
        plan: List[Dict[str, Any]],
        original_task: str,
        image_path: Optional[str],
        answer_text: Optional[str] = None,
    ) -> str:
        final, _, _ = self._execute_traced(
            plan, original_task, image_path, answer_text=answer_text,
        )
        return final

    # ── Public API ───────────────────────────────

    def run(
        self,
        task_content: str,
        image_path: Optional[str] = None,
        *,
        answer_text: Optional[str] = None,
    ) -> str:
        """
        Full pipeline: Dispatch → Execute.

        Parameters
        ----------
        task_content : str
            The user question (with options if applicable).
        image_path : str, optional
            Path to the associated image file.
        answer_text : str, optional
            Same as ``run_with_trace(..., answer_text=...)`` for ReasoningAgent.

        Returns
        -------
        str
            The final agent's output.
        """
        if not self._workers:
            raise RuntimeError(
                "No workers registered. Call register_worker() before run()."
            )

        plan = self._dispatch(task_content, image_path=image_path)

        logger.info(
            "Execution plan: %s",
            [(s["agent"], s["subtask"][:60]) for s in plan],
        )

        return self._execute(plan, task_content, image_path, answer_text=answer_text)

    def run_with_trace(
        self,
        task_content: str,
        image_path: Optional[str] = None,
        *,
        answer_text: Optional[str] = None,
    ) -> RunTrace:
        """
        Full pipeline with structured trace output.

        Same as run() but returns a RunTrace containing every intermediate
        result: dispatch plan, per-agent outputs, and the final answer.

        answer_text
            Reference answer string for the MCQ template (``Answer:`` line in Reasoning).
            Pass the dataset ground-truth when benchmarking; if omitted, an empty string is used.
        """
        if not self._workers:
            raise RuntimeError(
                "No workers registered. Call register_worker() before run()."
            )

        plan = self._dispatch(task_content, image_path=image_path)

        logger.info(
            "Execution plan: %s",
            [(s["agent"], s["subtask"][:60]) for s in plan],
        )

        final, steps, evidence = self._execute_traced(
            plan, task_content, image_path, answer_text=answer_text,
        )

        return RunTrace(
            original_task=task_content,
            dispatch_plan=plan,
            agent_steps=steps,
            final_answer=final,
            image_path=image_path,
            evidence_images=evidence,
        )

    # ── Prompt updates ───────────────────────────

    def update_coordinator_prompt(self, prompt: str) -> None:
        self._dispatcher = ChatAgent(
            system_message=prompt,
            model=self.model,
        )
