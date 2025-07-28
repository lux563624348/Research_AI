"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Optional, cast
from langchain_tavily import TavilySearch  # type: ignore[import-not-found]
from react_agent.configuration import Configuration
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


client = MultiServerMCPClient(
    {
        "twitter": {
            "command": "uv",
            # Replace with absolute path to your math_server.py file
            "args": ["run", "servers/twitter_server_lite.py"],
            "transport": "stdio",
        },
        "research": {
            # Ensure you start your weather server on port 8000
            "command": "uv",
            "args": ["run", "servers/research_server.py"],
            "transport": "stdio"
        }
    }
)

# Resolve coroutine at import time
all_tools = asyncio.run(client.get_tools())

