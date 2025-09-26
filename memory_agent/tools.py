"""Define the agent's tools, including memory and MCP tools."""

import uuid
import asyncio
import traceback
from typing import List


from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_mcp_adapters.client import MultiServerMCPClient


#define the vectorstore where we will be storing our memories
recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings())

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    ) # k (int) – Number of Documents to return. Defaults to 4.
    return [document.page_content for document in documents]

# === Load MCP tools ===
def load_mcptools() -> List:
    """Load MCP tools from configured FastMCP servers."""
    try:
        client = MultiServerMCPClient(
            {   
                "hubmap": {
                    "command": "python",
                    "args": ["./servers/hubmap_server.py"],
                    "transport": "stdio",
                },
                "cellxgene": {
                    "command": "python",
                    "args": ["./servers/cellxgene_server.py"],
                    "transport": "stdio",
                },
                "research": {
                    "command": "python",
                    "args": ["./servers/research_server.py"],
                    "transport": "stdio",
                },
                "pubmed":{
                    "command": "python",
                    "args": ["./servers/pubmed_server.py"],
                    "transport": "stdio",
                },
                "PDF": {
                    "command": "python",
                    "args": ["./servers/pdf_server.py"],
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

search = TavilySearch(max_results=3)

all_tools = [*mcptools, save_recall_memory, search_recall_memories, search]


