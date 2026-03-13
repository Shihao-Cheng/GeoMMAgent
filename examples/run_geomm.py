"""
GeoMMAgent Demo：展示 coordinator + exec_agents + toolkit 的完整协作流程。

功能：
  1. TaskPlanner          — 任务细化 + 结构化分解
  2. AgentCoordinator     — 注册三个专业 exec_agents，分发并执行
  3. 动态 prompt 替换     — 运行时修改规划提示词

运行：
  cd /root/autodl-tmp/owl
  python examples/run_geomm_demo.py [任务描述]

  # 加载 benchmark parquet 并逐条跑 pipeline
  python examples/run_geomm_demo.py --bench [parquet路径] [--limit N]
"""

import sys
import pathlib
from dotenv import load_dotenv

# ── 路径设置 ────────────────────────────────────
base_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))  # 确保 coordinator/ exec_agents/ toolkit/ 可导入

env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

# ── 导入核心模块 ────────────────────────────────
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.logger import set_log_level

from coordinator import TaskPlanner, AgentCoordinator
from exec_agents import (
    GeneralAgent, PerceptionAgent, KnowledgeAgent,
    ReasoningAgent, SelfEvaluationAgent,
)

set_log_level("WARNING")  # 减少 CAMEL 内部日志干扰

DEFAULT_PARQUET_PATH = "/root/autodl-tmp/CVPR/data/validation.parquet"


# ────────────────────────────────────────────────
# 工具函数
# ────────────────────────────────────────────────

def create_model():
    return ModelFactory.create(
        model_platform=ModelPlatformType.QWEN,
        model_type="qwen-max",
        model_config_dict={"temperature": 0},
    )


def demo_planner_only(model, task_content: str):
    """Demo 1：只做规划，查看任务分解结果"""
    print("\n" + "=" * 60)
    print("【1】TaskPlanner — 任务细化与分解")
    print("=" * 60)

    planner = TaskPlanner(model=model)
    result = planner.plan(task_content)

    print(f"原始任务: {result.original_content}")
    print(f"\n细化任务: {result.specified_task}")
    print(f"\n规划文本:\n{result.plan_text}")
    print(f"\n结构化子任务 ({len(result.subtasks)} 个):")
    for st in result.subtasks:
        print(f"  [{st.id}] {st.content}")

    return result


def demo_prompt_override(model, task_content: str):
    """Demo 2：运行时替换 decompose 提示词"""
    print("\n" + "=" * 60)
    print("【2】动态替换 decompose 提示词")
    print("=" * 60)

    custom_prompt = (
        "你是一个遥感图像分析专家。"
        "将以下任务分解为 3 个步骤：首先感知，然后检索知识，最后推理得出答案。"
        "每行一个，格式：'1. xxx'"
    )

    planner = TaskPlanner(model=model, decompose_prompt=custom_prompt)
    result = planner.plan(task_content)

    print(f"自定义提示词生效后的子任务 ({len(result.subtasks)} 个):")
    for st in result.subtasks:
        print(f"  [{st.id}] {st.content}")

    # 运行时再次替换
    planner.update_decompose_prompt(
        "Split the task into exactly 2 steps: analysis and conclusion. "
        "One step per line, prefix with '1.' and '2.'"
    )
    result2 = planner.plan(task_content)
    print(f"\n二次替换后的子任务 ({len(result2.subtasks)} 个):")
    for st in result2.subtasks:
        print(f"  [{st.id}] {st.content}")


def demo_coordinator_full(model, task_content: str):
    """Demo 3：完整协调流程（plan → Workforce 执行）"""
    print("\n" + "=" * 60)
    print("【3】AgentCoordinator — 注册 exec_agents 并执行")
    print("=" * 60)

    coord = AgentCoordinator(model=model)

    coord.register_workers([
        GeneralAgent(model=model).as_worker_dict(),
        PerceptionAgent(model=model).as_worker_dict(),
        KnowledgeAgent(model=model).as_worker_dict(),
        ReasoningAgent(model=model).as_worker_dict(),
        SelfEvaluationAgent(model=model).as_worker_dict(),
    ])

    print(f"已注册 {len(coord._workers)} 个 exec_agents")
    for w in coord._workers:
        print(f"  · {w['description'][:70]}")

    print(f"\n任务: {task_content}")
    print("正在执行（Workforce 分发中）...")

    try:
        result = coord.run(task_content)
        print(f"\n最终结果:\n{result}")
    except Exception as e:
        print(f"\n执行遇到错误: {e}")
        print("（可能是 API 额度、工具占位符等原因）")


# ────────────────────────────────────────────────
# Benchmark 批量模式
# ────────────────────────────────────────────────

def run_benchmark(model, parquet_path: str = None, limit: int = None):
    """从 parquet 加载 benchmark 数据，逐条跑完整 pipeline"""
    from toolkit import load_benchmark

    samples = load_benchmark(
        parquet_path=parquet_path or DEFAULT_PARQUET_PATH,
        limit=limit,
    )
    print(f"\n📂 已加载 {len(samples)} 条 benchmark 样本")

    coord = AgentCoordinator(model=model)
    coord.register_workers([
        GeneralAgent(model=model).as_worker_dict(),
        PerceptionAgent(model=model).as_worker_dict(),
        KnowledgeAgent(model=model).as_worker_dict(),
        ReasoningAgent(model=model).as_worker_dict(),
        SelfEvaluationAgent(model=model).as_worker_dict(),
    ])

    correct, total = 0, len(samples)
    for i, sample in enumerate(samples):
        print(f"\n{'─'*50}")
        print(f"[{i+1}/{total}] index={sample.index}  answer={sample.answer}")
        print(f"Image: {sample.image_path}")
        print(f"Prompt:\n{sample.prompt}")

        try:
            result = coord.run(sample.prompt, image_path=sample.image_path)
            pred = result.strip().upper()[:1] if result else ""
            is_ok = pred == sample.answer
            correct += is_ok
            flag = "✅" if is_ok else "❌"
            print(f"{flag} pred={pred}  gt={sample.answer}  |  raw: {result[:120]}")
        except Exception as e:
            print(f"❌ 执行出错: {e}")

    print(f"\n{'='*50}")
    print(f"📊 准确率: {correct}/{total} = {correct/total*100:.1f}%")


# ────────────────────────────────────────────────
# 主入口
# ────────────────────────────────────────────────

def _parse_args():
    """简单的参数解析，不引入额外依赖"""
    args = sys.argv[1:]
    if "--bench" in args:
        args.remove("--bench")
        limit = None
        if "--limit" in args:
            idx = args.index("--limit")
            limit = int(args[idx + 1])
            args = args[:idx] + args[idx + 2:]
        parquet_path = args[0] if args else None
        return "bench", parquet_path, limit
    task = args[0] if args else None
    return "demo", task, None


def main():
    mode, arg1, limit = _parse_args()

    try:
        model = create_model()
    except Exception as e:
        print(f"⚠️  模型初始化失败，请配置 QWEN_API_KEY: {e}")
        return

    if mode == "bench":
        print("\n🛰  GeoMMAgent — Benchmark 模式")
        run_benchmark(model, parquet_path=arg1, limit=limit)
    else:
        default_task = (
            "这张遥感图像中有几架飞机？图像场景属于什么类型？"
            "请给出最终答案，选项为：A. 3架  B. 5架  C. 7架  D. 10架"
        )
        task_content = arg1 or default_task

        print("\n🛰  GeoMMAgent 模块化 Demo")
        print(f"任务: {task_content}\n")

        demo_planner_only(model, task_content)
        demo_prompt_override(model, task_content)
        demo_coordinator_full(model, task_content)

    print("\n✅ 完成\n")


if __name__ == "__main__":
    main()
