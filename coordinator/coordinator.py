# coordinator/coordinator.py
# 任务协调模块：构建 Workforce，注册 exec_agents，分发并执行任务
#
# 用法：
#   from coordinator.coordinator import AgentCoordinator
#   coord = AgentCoordinator(model=my_model)
#   coord.register_worker("描述", agent_instance)
#   result = coord.run("用户问题", image_path="path/to/img.png")

from typing import Any, Dict, List, Optional

from PIL import Image

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.societies import Workforce
from camel.tasks.task import Task
from camel.types import RoleType

from .prompts import TASK_AGENT_PROMPT, COORDINATOR_AGENT_PROMPT
from .planner import TaskPlanner, PlanResult


def _load_image(image_path: Optional[str]) -> Optional[List[Image.Image]]:
    """加载图像为 PIL Image 列表，供 BaseMessage.image_list 使用"""
    if not image_path:
        return None
    img = Image.open(image_path)
    if img.format is None:
        img.format = "PNG"
    return [img]


class AgentCoordinator:
    """
    可复用的协调器，封装了：
      - Workforce 的构建（task_agent + coordinator_agent）
      - worker 注册（可动态添加 exec_agents）
      - 完整的 plan → execute 流程（支持图像输入）

    参数
    ----
    model : BaseModelBackend
        规划和协调使用的模型（exec_agents 可使用不同模型）
    task_agent_prompt : str
        覆盖 task_agent 的 system prompt
    coordinator_agent_prompt : str
        覆盖 coordinator_agent 的 system prompt
    """

    def __init__(
        self,
        model: BaseModelBackend,
        task_agent_prompt: str = TASK_AGENT_PROMPT,
        coordinator_agent_prompt: str = COORDINATOR_AGENT_PROMPT,
    ):
        self.model = model
        self.task_agent_prompt = task_agent_prompt
        self.coordinator_agent_prompt = coordinator_agent_prompt

        self._workers: List[Dict[str, Any]] = []  # {description, agent}
        self._planner = TaskPlanner(model=model)
        self._workforce: Optional[Workforce] = None  # 延迟构建

    # ──────────────────────────────────────────
    # 注册 exec_agents
    # ──────────────────────────────────────────

    def register_worker(self, description: str, agent: ChatAgent) -> None:
        """注册一个执行单元 agent，可多次调用插入不同专业 agent"""
        self._workers.append({"description": description, "agent": agent})
        self._workforce = None  # 注册后重建

    def register_workers(self, worker_list: List[Dict[str, Any]]) -> None:
        """批量注册，每个元素需包含 'description' 和 'agent' 键"""
        for w in worker_list:
            self.register_worker(w["description"], w["agent"])

    # ──────────────────────────────────────────
    # 内部：构建 Workforce
    # ──────────────────────────────────────────

    def _build_workforce(self) -> Workforce:
        task_agent = ChatAgent(
            system_message=self.task_agent_prompt,
            model=self.model,
        )
        coordinator_agent = ChatAgent(
            system_message=self.coordinator_agent_prompt,
            model=self.model,
        )
        wf = Workforce(
            "GeoMMAgent Workforce",
            task_agent=task_agent,
            coordinator_agent=coordinator_agent,
        )
        for w in self._workers:
            wf.add_single_agent_worker(w["description"], worker=w["agent"])
        return wf

    # ──────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────

    def run(
        self,
        task_content: str,
        image_path: Optional[str] = None,
    ) -> str:
        """
        完整流程：规划 → Workforce 执行 → 返回结果字符串。

        参数
        ----
        task_content : str
            用户问题文本（含选项等）
        image_path : str | None
            关联的图像文件路径，会以 [Image: path] 标签嵌入 task content，
            同时作为 additional_info 附在 Task 对象上供下游 agent 使用
        """
        if not self._workers:
            raise RuntimeError(
                "No workers registered. Call register_worker() before run()."
            )

        if self._workforce is None:
            self._workforce = self._build_workforce()

        content = task_content
        if image_path:
            content = f"[Image: {image_path}]\n{task_content}"

        task = Task(content=content, additional_info=image_path)
        processed = self._workforce.process_task(task)
        return processed.result or ""

    def run_sequential(
        self,
        task_content: str,
        image_path: Optional[str] = None,
    ) -> str:
        """
        备选流程：顺序调用各 worker，每个 worker 收到含图像的 BaseMessage。

        不依赖 Workforce，直接通过 ChatAgent.step() 逐个执行，
        后续 agent 可看到前序 agent 的输出。
        """
        if not self._workers:
            raise RuntimeError(
                "No workers registered. Call register_worker() before run()."
            )

        image_list = _load_image(image_path)
        image_tag = f"[Image: {image_path}]\n" if image_path else ""

        agent_outputs: List[str] = []
        for w in self._workers:
            agent: ChatAgent = w["agent"]
            desc: str = w["description"]

            context = ""
            if agent_outputs:
                context = (
                    "Previous agent outputs:\n"
                    + "\n---\n".join(agent_outputs)
                    + "\n\n"
                )

            user_content = f"{image_tag}{context}Task: {task_content}"

            msg = BaseMessage(
                role_name="user",
                role_type=RoleType.USER,
                content=user_content,
                image_list=image_list,
                image_detail="high",
            )

            try:
                resp = agent.step(msg)
                output = resp.msgs[0].content if resp.msgs else ""
            except Exception as e:
                output = f"[Agent error: {e}]"

            agent_outputs.append(f"[{desc[:60]}]\n{output}")

        return agent_outputs[-1].split("\n", 1)[-1] if agent_outputs else ""

    def plan_only(
        self,
        task_content: str,
        image_path: Optional[str] = None,
    ) -> PlanResult:
        """
        只做规划，不执行 Workforce。
        适合用于预览任务分解结果，或做自定义执行逻辑。
        """
        return self._planner.plan(task_content, image_path=image_path)

    # ──────────────────────────────────────────
    # 提示词动态更新
    # ──────────────────────────────────────────

    def update_task_agent_prompt(self, prompt: str) -> None:
        self.task_agent_prompt = prompt
        self._workforce = None  # 触发重建

    def update_coordinator_agent_prompt(self, prompt: str) -> None:
        self.coordinator_agent_prompt = prompt
        self._workforce = None

    def update_decompose_prompt(self, prompt: str) -> None:
        self._planner.update_decompose_prompt(prompt)
