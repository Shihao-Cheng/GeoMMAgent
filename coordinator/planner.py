# coordinator/planner.py
# 任务规划模块：TaskSpecifyAgent 细化 + TaskPlannerAgent 拆解 + Task.decompose
#
# 用法：
#   from coordinator.planner import TaskPlanner
#   planner = TaskPlanner(model=my_model)
#   result = planner.plan("用户问题")
#   print(result.specified_task)
#   print(result.subtasks)

from dataclasses import dataclass, field
from typing import List, Optional

from camel.agents import ChatAgent
from camel.agents.task_agent import TaskSpecifyAgent, TaskPlannerAgent
from camel.models import BaseModelBackend
from camel.tasks.task import Task

from .prompts import TASK_SPECIFY_WORD_LIMIT, DECOMPOSE_PROMPT


@dataclass
class PlanResult:
    """规划结果，包含细化任务、子任务列表和根 Task 对象"""
    original_content: str
    specified_task: str
    plan_text: str
    root_task: Task
    subtasks: List[Task] = field(default_factory=list)
    image_path: Optional[str] = None


class TaskPlanner:
    """
    可复用的任务规划器，封装了：
      1. TaskSpecifyAgent — 细化任务描述
      2. TaskPlannerAgent — 生成子任务列表文本
      3. Task.decompose    — 结构化拆解为 Task 对象

    参数
    ----
    model : BaseModelBackend
        模型后端（ModelFactory.create 的返回值）
    word_limit : int
        TaskSpecifyAgent 细化描述的词数上限
    decompose_prompt : str
        传给 decompose 的自定义提示词；为 None 则使用默认
    """

    def __init__(
        self,
        model: BaseModelBackend,
        word_limit: int = TASK_SPECIFY_WORD_LIMIT,
        decompose_prompt: Optional[str] = DECOMPOSE_PROMPT,
    ):
        self.model = model
        self.word_limit = word_limit
        self.decompose_prompt = decompose_prompt

        self._task_specify_agent = TaskSpecifyAgent(
            model=model,
            word_limit=word_limit,
        )
        self._task_planner_agent = TaskPlannerAgent(model=model)

        # decompose 使用的 ChatAgent（可替换 system prompt）
        self._decompose_agent = ChatAgent(
            system_message=decompose_prompt or "You are a helpful task planner.",
            model=model,
        )

    def plan(
        self,
        task_content: str,
        task_id: str = "0",
        image_path: Optional[str] = None,
    ) -> PlanResult:
        """
        对用户输入执行完整规划流程。

        参数
        ----
        task_content : str
            用户问题文本
        task_id : str
            根任务 ID
        image_path : str | None
            关联的图像文件路径（传递给下游 agent 用于 VLM 推理）

        返回 PlanResult
        """
        content_for_plan = task_content
        if image_path:
            content_for_plan = f"[Image: {image_path}]\n{task_content}"

        # 1. 细化任务
        specified = self._task_specify_agent.run(content_for_plan)
        specified_str = str(specified)

        # 2. 生成规划文本
        plan_text = str(self._task_planner_agent.run(specified))

        # 3. 结构化拆解为 Task 对象
        root_task = Task(content=content_for_plan, id=task_id)
        subtasks = root_task.decompose(agent=self._decompose_agent)

        return PlanResult(
            original_content=task_content,
            specified_task=specified_str,
            plan_text=plan_text,
            root_task=root_task,
            subtasks=subtasks,
            image_path=image_path,
        )

    def update_decompose_prompt(self, new_prompt: str) -> None:
        """运行时替换 decompose 使用的提示词"""
        self.decompose_prompt = new_prompt
        self._decompose_agent = ChatAgent(
            system_message=new_prompt,
            model=self.model,
        )
