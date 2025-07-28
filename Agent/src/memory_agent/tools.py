"""Define the agent's tools, including memory and MCP tools."""

import uuid
import asyncio
import traceback
from typing import Annotated, Optional, List, Union

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore

from memory_agent.configuration import Configuration
from langchain_mcp_adapters.client import MultiServerMCPClient

# === Memory tool ===
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: Optional[uuid.UUID] = None,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedToolArg],
):
    """Upsert a memory in the database."""
    mem_id = memory_id or uuid.uuid4()
    user_id = Configuration.from_runnable_config(config).user_id
    await store.aput(
        ("memories", user_id),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"


# === Load MCP tools ===
def load_mcptools() -> List:
    """Load MCP tools from configured FastMCP servers."""
    try:
        client = MultiServerMCPClient(
            {
                "twitter": {
                    "command": "python",
                    "args": ["./servers/twitter_server_lite.py"],
                    "transport": "stdio",
                },
                "research": {
                    "command": "python",
                    "args": ["./servers/research_server.py"],
                    "transport": "stdio",
                },
            }
        )
        tools = asyncio.run(client.get_tools())
        print(f"✅ Loaded {len(tools)} MCP tools")
        return tools
    except Exception as e:
        print("❌ Failed to load MCP tools")
        traceback.print_exc()
        return []

# Load MCP tools once at import
mcptools = load_mcptools()