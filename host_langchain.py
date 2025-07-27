import os
import json
import re
import asyncio, nest_asyncio
from datetime import datetime
from dotenv import load_dotenv
from anthropic import Anthropic as AnthropicRaw

from langchain.chat_models import ChatAnthropic
from langchain.agents import Tool, AgentExecutor, initialize_agent
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from typing import List, Dict

load_dotenv()
nest_asyncio.apply()

HISTORY_DIR = "history"
LLM_MODEL = "claude-3-7-sonnet-20250219"

def anthropic_cost(input_tokens: int, output_tokens: int, model: str = "claude-3-sonnet") -> float:
    prices = {
        "claude-3-5-haiku-20241022": (0.8, 4),
        "claude-3-7-sonnet-20250219": (3, 15),
    }
    in_price, out_price = prices[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000

def extract_topic(query: str) -> str:
    words = re.findall(r'\b\w+\b', query.lower())
    return "_".join(words[:2])[:32] or "misc"

class MCPAgent:
    def __init__(self):
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: List[Tool] = []
        self.prompts = []
        self.exit_stack = AsyncExitStack()
        self.token_usage = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}

        self.llm = ChatAnthropic(
            model=LLM_MODEL,
            temperature=0.3,
            streaming=False,
        )

    async def connect_all_servers(self):
        with open("./servers/server_config.json", "r") as f:
            servers = json.load(f).get("mcpServers", {})

        for name, config in servers.items():
            await self.connect_server(name, config)

    async def connect_server(self, name, config):
        params = StdioServerParameters(**config)
        stdio = await self.exit_stack.enter_async_context(stdio_client(params))
        read, write = stdio
        session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # Tools
        tool_list = await session.list_tools()
        for tool_info in tool_list.tools:
            self.sessions[tool_info.name] = session

            @tool(name=tool_info.name, description=tool_info.description)
            async def _wrapped_tool(**kwargs):
                result = await session.call_tool(tool_info.name, arguments=kwargs)
                self.save_history(tool_info.name, kwargs, result.content)
                return result.content

            self.tools.append(_wrapped_tool)

        # Prompts (for manual mode)
        prompt_resp = await session.list_prompts()
        if prompt_resp.prompts:
            for p in prompt_resp.prompts:
                self.prompts.append(p)
                self.sessions[p.name] = session

    def save_history(self, tool_name, tool_input, result):
        topic_dir = os.path.join(HISTORY_DIR, extract_topic(str(tool_input)))
        os.makedirs(topic_dir, exist_ok=True)
        file_path = os.path.join(topic_dir, "history_info.json")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data[timestamp] = {
            "Tool": tool_name,
            "Tool_input": tool_input,
            "result": result
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    async def run(self):
        agent_executor = initialize_agent(
            self.tools,
            self.llm,
            agent="chat-conversational-react-description",
            verbose=True,
            handle_parsing_errors=True
        )

        print("🚀 LangChain-MCP Agent Started!")
        print("💬 Type your query or 'quit' to exit.\n")

        while True:
            user_input = input("\nQuery: ").strip()
            if user_input.lower() == 'quit':
                break

            try:
                result = await agent_executor.arun(user_input)
                print(f"\n🧠 Answer: {result}")
            except Exception as e:
                print(f"❌ Error: {e}")

    async def cleanup(self):
        print("Closing resources...")
        await self.exit_stack.aclose()

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    async def main():
        agent = MCPAgent()
        await agent.connect_all_servers()
        try:
            await agent.run()
        finally:
            await agent.cleanup()

    asyncio.run(main())
