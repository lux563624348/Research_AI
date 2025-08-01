"""Graphs that extract memories on a schedule."""

import asyncio
import logging
from datetime import datetime

from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.store.base import BaseStore

from memory_agent import configuration, tools, utils
from memory_agent.tools import mcptools
from memory_agent.state import State

logging.basicConfig(level=logging.DEBUG)  # or DEBUG for more detail
logger = logging.getLogger(__name__)


# Initialize the language model to be used for memory extraction
llm = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0)


async def call_model(state: State, config: RunnableConfig, *, store: BaseStore) -> dict:
    """Extract the user's state from the conversation and update the memory."""
    configurable = configuration.Configuration.from_runnable_config(config)

    # Retrieve the most recent memories for context
    memories = await store.asearch(
        ("memories", configurable.user_id),
        query=str([m.content for m in state.messages[-3:]]),
        limit=10,
    )

    # Format memories for inclusion in the prompt
    formatted = "\n".join(f"[{mem.key}]: {mem.value} (similarity: {mem.score})" for mem in memories)
    if formatted:
        formatted = f"""
<memories>
{formatted}
</memories>"""

    # Prepare the system prompt with user memories and current time
    # This helps the model understand the context and temporal relevance
    sys = configurable.system_prompt.format(
        user_info=formatted, time=datetime.now().isoformat()
    )

    # Invoke the language model with the prepared prompt and tools
    # "bind_tools" gives the LLM the JSON schema for all tools in the list so it knows how
    # to use them.           #tools.upsert_memory, 
    msg = await llm.bind_tools([tools.upsert_memory, *mcptools]).ainvoke(
        [{"role": "system", "content": sys}, *state.messages],
        {"configurable": utils.split_model_and_provider(configurable.model)},
    )

    # === ✨ Extract tool call results and upsert memory ===
    tool_messages = msg.tool_calls
    
    for tool_call in tool_messages:
        tool_id = tool_call["id"]
        result = msg.tool_call_outputs.get(tool_id) if hasattr(msg, "tool_call_outputs") else None

        if isinstance(result, dict) and "content" in result and "context" in result:
            await tools.upsert_memory(
                content=result["content"],
                context=result["context"],
                config=config,
                store=store,
            )
    return {"messages": [msg]}

async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Extract tool calls from the last message
    tool_calls = state.messages[-1].tool_calls

    # Concurrently execute all upsert_memory calls
    saved_memories = await asyncio.gather(
        *(
            tools.upsert_memory(**tc["args"], config=config, store=store)
            for tc in tool_calls
        )
    )

    # Format the results of memory storage operations
    # This provides confirmation to the model that the actions it took were completed
    results = [
        {
            "role": "tool",
            "content": mem,
            "tool_call_id": tc["id"],
        }
        for tc, mem in zip(tool_calls, saved_memories)
    ]
    return {"messages": results}

async def store_memory(state: State, config: RunnableConfig, *, store: BaseStore):
    # Extract tool calls from the last message
    tool_calls = state.messages[-1].tool_calls
    print("XIANG: ", state.messages[-1].tool_calls)

    results = []

    for tc in tool_calls:
        tool_name = tc.get("name")
        args = tc.get("args", {})
        # Assume this tool just ran and returned its result
        # Check if the result has content/context and store as memory
        if tool_name == "upsert_memory":
            if "content" in args and "context" in args:
                result = await tools.upsert_memory(**args, config=config, store=store)
            else:
                result = f"Skipped upsert_memory: missing fields in {tc['id']}"
        else:
            result = f"Tool {tool_name} called; no memory saved."

        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": result,
        })
    return {"messages": results}


def route_message(state: State):
    """Determine the next step based on the presence of tool calls."""
    msg = state.messages[-1]
    if msg.tool_calls:
        # If there are tool calls, we need to store memories
        return "store_memory"
    # Otherwise, finish; user can send the next message
    return END


# Create the graph + all nodes
builder = StateGraph(State, config_schema=configuration.Configuration)

# Define the flow of the memory extraction process
builder.add_node(call_model)
builder.add_edge("__start__", "call_model")
builder.add_node(store_memory)
builder.add_conditional_edges("call_model", route_message, ["store_memory", END])
# Right now, we're returning control to the user after storing a memory
# Depending on the model, you may want to route back to the model
# to let it first store memories, then generate a response
builder.add_edge("store_memory", "call_model")
graph = builder.compile()
graph.name = "MemoryAgent"


__all__ = ["graph"]
