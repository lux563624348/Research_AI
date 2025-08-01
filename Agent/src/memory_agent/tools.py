"""Define the agent's tools, including memory and MCP tools."""

import uuid
import asyncio
import traceback
from typing import List

from langchain_mcp_adapters.client import MultiServerMCPClient

# === Load MCP tools ===
def load_mcptools() -> List:
    """Load MCP tools from configured FastMCP servers."""
    try:
        client = MultiServerMCPClient(
            {
                "twitter": {
                    "command": "python",
                    "args": ["./servers/twitter_server.py"],
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