from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
from datetime import datetime, UTC
import json, os, re
import asyncio
import nest_asyncio
import shlex

nest_asyncio.apply()

load_dotenv()
History_DIR = "history"

def anthropic_cost(input_tokens: int, output_tokens: int, model: str = "claude-3-sonnet") -> float:
    prices = {
        "claude-3-5-haiku-20241022": (0.8, 4),
        "claude-3-7-sonnet-20250219": (3, 15),
    }
    in_price, out_price = prices[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1000000

def serialize_response(response):
    if isinstance(response, str):
        return response
    elif hasattr(response, "text"):
        return response.text
    elif hasattr(response, "content"):
        return response.content
    elif hasattr(response, "__str__"):
        return str(response)
    else:
        return repr(response)  # fallback for debugging

def extract_top_words(query: str, max_len: int = 66) -> str:
    """
    Extracts the first 2 alphanumeric words from the query, lowercase and joined by underscores.
    Falls back to a short hash if nothing valid is found.
    """
    words = re.findall(r'\b\w+\b', query.lower())
    topic = "_".join(words[:4])
    topic = topic[:max_len]  # truncate if too long
    return topic

class MCP_ChatBot:

    def __init__(self):
        self.token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0
        }
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        # Tools list required for Anthropic API
        self.available_tools = []
        # Prompts list for quick display 
        self.available_prompts = []
        # Sessions dict maps tool/prompt names or resource URIs to MCP client sessions
        self.sessions = {}
        self.llm_model = 'claude-3-7-sonnet-20250219'#os.getenv("LLM_MODEL")
        #
        #self.llm_model = "claude-3-5-haiku-20241022"#

    async def connect_to_servers(self): # new
        """Connect to all configured MCP servers."""
        try:
            with open("./servers/server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def connect_to_server(self, server_name, server_config):
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            
            
            try:
                # List available tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
            
                # List available prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List available resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
            
            except Exception as e:
                print(f"Error {e}")
                
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")

    async def process_query(self, query):
        # Get the current UTC time in ISO 8601 format (Anthropic likes clarity)
        current_time = datetime.now(UTC).isoformat()
        system_prompt = f"The current time is {current_time}."
        messages = [{'role':'user', 'content':query}]
        
        while True:
            response = self.anthropic.messages.create(
                max_tokens = 2024,
                model = self.llm_model, 
                system=system_prompt,
                tools = self.available_tools,
                messages = messages
            )
            ## calculate cost
            usage = response.usage
            input_tokens = usage.input_tokens or 0
            output_tokens = usage.output_tokens or 0
            cost = anthropic_cost(input_tokens, output_tokens, model=self.llm_model)

            self.token_usage["input_tokens"] += input_tokens
            self.token_usage["output_tokens"] += output_tokens
            self.token_usage["cost"] += cost
            #print(f"This query costs: ${cost:.5f}")
            assistant_content = []
            has_tool_use = False
            
            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    assistant_content.append(content)
                    messages.append({'role':'assistant', 'content':assistant_content})
                    
                    # Get session and call tool
                    session = self.sessions.get(content.name)
                    if not session:
                        print(f"Tool '{content.name}' not found.")
                        break
                    

                    print(f"Calling Tool: ${content.name}, ${content.input}")
                    result = await session.call_tool(content.name, arguments=content.input)
                    messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })

                    ## Save results
                    # Create directory for this topic
                    path = os.path.join(History_DIR, extract_top_words(query))
                    os.makedirs(path, exist_ok=True)
                    file_path = os.path.join(path, "history_info.json")

                    # Try to load existing history info
                    try:
                        with open(file_path, "r") as json_file:
                            history_info = json.load(json_file)
                    except (FileNotFoundError, json.JSONDecodeError):
                        history_info = {}

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    history_info[timestamp] = {
                        "Tool": content.name if isinstance(content.name, str) else str(content.name),
                        "Tool_input": content.input if isinstance(content.input, str) else str(content.input),
                        "result": result.content if isinstance(result.content, str) else str(result.content)
                    }

                    # Save updated papers_info to json file
                    with open(file_path, "w") as json_file:
                        json.dump(history_info, json_file, indent=2)

            # Exit loop if no tool was used
            if not has_tool_use:
                break  
    
    async def get_resource(self, resource_uri):
        session = self.sessions.get(resource_uri)
        
        # Fallback for history URIs - try any history resource session
        if not session and resource_uri.startswith("history://"):
            for uri, sess in self.sessions.items():
                if uri.startswith("history://"):
                    session = sess
                    break
            
        if not session:
            print(f"Resource '{resource_uri}' not found.")
            return
        
        try:
            result = await session.read_resource(uri=resource_uri)
            if result and result.contents:
                print(f"\nResource: {resource_uri}")
                print("Content:")
                print(result.contents[0].text)
            else:
                print("No content available.")
        except Exception as e:
            print(f"Error: {e}")

    async def list_prompts(self):
        """List all available prompts."""
        if not self.available_prompts:
            print("No prompts available.")
            return
        
        print("\nAvailable prompts:")
        for prompt in self.available_prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
            if prompt['arguments']:
                print(f"  Arguments:")
                for arg in prompt['arguments']:
                    arg_name = arg.name if hasattr(arg, 'name') else arg.get('name', '')
                    print(f"    - {arg_name}")

    async def execute_prompt(self, prompt_name, args):
        """Execute a prompt with the given arguments."""
        session = self.sessions.get(prompt_name)
        if not session:
            print(f"Prompt '{prompt_name}' not found.")
            return
        
        try:
            result = await session.get_prompt(prompt_name, arguments=args)
            if result and result.messages:
                prompt_content = result.messages[0].content
                
                # Extract text from content (handles different formats)
                if isinstance(prompt_content, str):
                    text = prompt_content
                elif hasattr(prompt_content, 'text'):
                    text = prompt_content.text
                else:
                    # Handle list of content items
                    text = " ".join(item.text if hasattr(item, 'text') else str(item) 
                                  for item in prompt_content)
                
                print(f"\nExecuting prompt '{prompt_name}'...")
                await self.process_query(text)
        except Exception as e:
            print(f"Error: {e}")

    async def chat_loop(self):
        print("\n🚀 MCP Agent for Bioinformatics Research AI Started!")
        print("💬 Type your queries or 'quit' to exit.")
        print("📚 Use @history to see available topics")
        print("🔍 Use @<topic> to search history in that topic")
        print("🧠 Use /prompts to list available prompts")
        print("⚙️ Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if not query:
                    continue
                if query.lower() == 'quit':
                    break

                # Check for @resource syntax first
                if query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "history":
                        resource_uri = "history://history"
                    else:
                        resource_uri = f"history://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if query.startswith('/'):
                    #parts = query.split()
                    parts = shlex.split(query)
                    command = parts[0].lower()
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        print (args)
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue
                
                await self.process_query(query)

            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self): # new
        """Cleanly close all resources using AsyncExitStack."""
        #print("---- Token Usage Summary ----")
        #print(f"Input Tokens   : {self.token_usage['input_tokens']}")
        #print(f"Output Tokens  : {self.token_usage['output_tokens']}")
        #print(f"Total Cost     : ${self.token_usage['cost']:.5f}")
        await self.exit_stack.aclose()


async def main():
    chatbot = MCP_ChatBot()
    try:
        # the mcp clients and sessions are not initialized using "with"
        # like in the previous lesson
        # so the cleanup should be manually handled
        await chatbot.connect_to_servers() 
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()  


if __name__ == "__main__":
    asyncio.run(main())