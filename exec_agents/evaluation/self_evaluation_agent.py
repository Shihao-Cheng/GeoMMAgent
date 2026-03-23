# exec_agents/evaluation/self_evaluation_agent.py
# Self-Evaluation Agent — 验证推理链一致性、证据支撑度，触发重执行

from typing import List, Optional

from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool

from exec_agents.base import BaseExecAgent
from .prompts import SELF_EVAL_SYSTEM_PROMPT, SELF_EVAL_WORKER_DESC


class SelfEvaluationAgent(BaseExecAgent):
    """
    自评估 Agent，对应论文 plan–execute–evaluate 的第三阶段。

    职责：
      - 接收执行日志（perception / knowledge / reasoning 输出）和候选答案
      - 评估逻辑一致性、证据充分性、任务完整性
      - 输出 pass/fail + confidence + error_analysis + revised_plan
      - 若 fail，coordinator 据此触发重执行
    """

    SYSTEM_PROMPT = SELF_EVAL_SYSTEM_PROMPT
    WORKER_DESCRIPTION = SELF_EVAL_WORKER_DESC

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
    ):
        super().__init__(model=model, system_prompt_override=system_prompt_override)

    def get_tools(self) -> List[FunctionTool]:
        tools = []
        try:
            from toolkit.evaluation_metrics import evaluate_trace_with_metrics

            tools.append(FunctionTool(evaluate_trace_with_metrics))
        except Exception:
            pass
        return tools
