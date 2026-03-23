import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(str(pathlib.Path(__file__).parent / ".env"))

from exec_agents.knowledge.search_agent import SearchAgent
from configs.loader import create_model_from_config

model = create_model_from_config("qwen-vl-max")
agent = SearchAgent(model=model)

tools = agent.get_tools()
print(f"Registered tools: {len(tools)}")
for t in tools:
    print(f"  - {t.get_function_name()}: {t.get_function_description()[:100]}")

chat_agent = agent.build()
print(f"\nChatAgent tools: {len(chat_agent.tool_list)}")
for t in chat_agent.tool_list:
    print(f"  - {t.get_function_name()}")
