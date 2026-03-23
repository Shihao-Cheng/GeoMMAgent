# exec_agents/base.py
# 所有执行单元 agent 的基类，统一接口，支持 prompt 覆盖
#
# 子类只需覆盖：
#   SYSTEM_PROMPT  — 该 agent 的专业提示词
#   get_tools()    — 返回该 agent 使用的工具列表

from typing import List, Optional

from camel.agents import ChatAgent
from camel.models import BaseModelBackend
from camel.toolkits import FunctionTool


class BaseExecAgent:
    """
    执行单元基类。

    子类使用示例
    ------------
    class PerceptionAgent(BaseExecAgent):
        SYSTEM_PROMPT = \"\"\"You are a perception expert...\"\"\"

        def get_tools(self):
            from toolkit.detection import get_detection_tools
            return get_detection_tools()
    """

    # 子类覆盖此类属性，或在 prompts.py 中定义后引用
    SYSTEM_PROMPT: str = "You are a helpful assistant."

    # worker 在 Workforce 中的描述，供 coordinator_agent 做任务分配决策
    WORKER_DESCRIPTION: str = "A helpful assistant."

    def __init__(
        self,
        model: Optional[BaseModelBackend] = None,
        system_prompt_override: Optional[str] = None,
    ):
        """
        参数
        ----
        model : BaseModelBackend
            模型后端；为 None 时使用 CAMEL 默认模型
        system_prompt_override : str
            运行时覆盖 SYSTEM_PROMPT，用于不修改代码的快速调试
        """
        self.model = model
        self._prompt = system_prompt_override or self.SYSTEM_PROMPT

    def get_system_prompt(self) -> str:
        """返回当前使用的 system prompt（可在子类中动态生成）"""
        return self._prompt

    def get_tools(self) -> List[FunctionTool]:
        """子类返回该 agent 绑定的工具列表"""
        return []

    def build(self) -> ChatAgent:
        """组装成 CAMEL ChatAgent，供 Workforce.add_single_agent_worker 使用"""
        kwargs = {"tools": self.get_tools()}
        if self.model is not None:
            kwargs["model"] = self.model
        return ChatAgent(
            system_message=self.get_system_prompt(),
            **kwargs,
        )

    def as_worker_dict(self) -> dict:
        """
        返回 {'name': ..., 'description': ..., 'agent': ChatAgent}，
        方便直接传给 AgentCoordinator.register_workers()
        """
        return {
            "name": type(self).__name__,
            "description": self.WORKER_DESCRIPTION,
            "agent": self.build(),
        }
