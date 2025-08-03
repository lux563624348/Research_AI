import os,json
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP(name="twitter-server")
model = "claude-3-7-sonnet-20250219"

History_DIR = "history"

async def make_twitter_endpoint_request(endpoint: str, query_param: dict(), ctx: Context | None = None) -> str:
    """
    Query Twitter API via api.twitterapi.io and return.

    Args:
        query (str): A Twitter search query (e.g., "from:elonmusk #AI").
        ctx (Optional[Context]): FastMCP context object for tracing/logging.

    Returns:
        
    """
    api_key = os.getenv("TWITTER_API_KEY")
    if not api_key:
        if ctx:
            await ctx.error("❌ Missing TWITTER_API_KEY in environment.")
        return None

    try:
        async with httpx.AsyncClient() as client:
            headers = {"X-API-Key": api_key}
            base_url = "https://api.twitterapi.io/twitter"
            #endpoint = 
            url = f"{base_url}/{endpoint}"
            
            response = await client.get(url, headers=headers, params=query_param) 
            response.raise_for_status()
            return response.json()
    except Exception as e:
        if ctx:
            await ctx.error(f"❌ Twitter API request failed: {e}")
        return None

async def _extract_with_anthropic(llm_text: str, api_key: str, ctx: Context | None = None) -> str:
    client = AsyncAnthropic(api_key=api_key)

    try:
        response = await client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"You are a helpful assistant that converts natural language to TwitterAPI.Io advanced search queries. Only return the query string.\n\nExtract query: {llm_text}"
                }
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        if ctx:
            await ctx.error(f"Anthropic extraction failed: {e}")
        raise

async def extract_search_query_from_llm_text(llm_text: str, ctx: Context | None = None) -> str:
    llm_key = os.getenv("ANTHROPIC_API_KEY")

    if llm_key:
        try:
            return await _extract_with_anthropic(llm_text, llm_key, ctx)
        except Exception:
            pass  # Fallback to rule-based
    else:
        await ctx.warn("❗ No LLM API key found. Using rule-based fallback.")
    # Fallback
    return llm_text.strip() if "from:" in llm_text else f"'{llm_text.strip()}'"

def structured_response(data=None, message="success", status="success"):
    """
    Create a consistent response format for MCP tools.

    Args:
        data (Any): The actual data payload (list, dict, str, etc.)
        message (str): A short message about the result.
        status (str): Either 'success' or 'error'.

    Returns:
        dict: Standardized response object.
    """
    return {
        "status": status,
        "context": message,
        "content": data if data is not None else [],
    }

def format_tweet_str(tweet: dict, users: dict = None) -> str:
    author = tweet.get("author", {})
    author_id = author.get("id")
    username = author.get("username", "unknown")
    name = author.get("name", "Unknown User")

    if users and author_id in users:
        user = users[author_id]
        username = user.get("username", username)
        name = user.get("name", name)

    created_at = tweet.get("createdAt", "Unknown time")
    text = tweet.get("text", "No text available")

    metrics = tweet.get("public_metrics", {})
    likes = metrics.get("like_count", 0)
    retweets = metrics.get("retweet_count", 0)
    replies = metrics.get("reply_count", 0)

    return f"""@{username} ({name})\nPosted: {created_at}\n{text}\nLikes: {likes} | Retweets: {retweets} | Replies: {replies}\n"""

def extract_tweets(data: dict) -> list:
    """
    Safely extract a list of tweets from the response data,
    whether it's in data["tweets"] or data["data"]["tweets"].
    """
    if "tweets" in data and isinstance(data["tweets"], list):
        return data["tweets"]
    elif "data" in data and isinstance(data["data"], dict):
        return data["data"].get("tweets", [])
    else:
        return []

def simple_tweet_fields(tweets: list) -> list:
    """
    Convert a list of full tweet objects into a simplified format to save tokens.
    """
    def simplify(tweet: dict) -> dict:
        return {
            "id": tweet.get("id"),
            "text": tweet.get("text"),
            "createdAt": tweet.get("createdAt"),
            "author": {
                "userName": tweet.get("author", {}).get("userName"),
                "id": tweet.get("author", {}).get("id"),
            },
            "quoted_tweet": tweet.get("quoted_tweet", {}),
            "retweeted_tweet": tweet.get("retweeted_tweet", {})
        }

    return [simplify(tweet) for tweet in tweets]

def format_user(u: dict) -> str:
    """
    Format a user dictionary from TwitterAPI.io into a readable string.
    """
    name = u.get("name", "N/A")
    username = u.get("screen_name", "N/A")
    bio = u.get("description", "")
    followers = u.get("followers_count", 0)
    following = u.get("friends_count", 0)
    location = u.get("location", "")
    verified = u.get("verified", False)
    created = u.get("created_at", "")

    verified_badge = "✅" if verified else ""

    return (
        f"👤 @{username} {verified_badge}\n"
        f"Name: {name}\n"
        f"Bio: {bio or '—'}\n"
        f"Followers: {followers:,} | Following: {following:,}\n"
        f"Location: {location or '—'}\n"
        f"Created: {created}"
    )

@mcp.tool()
async def get_user_last_tweets(userId: str) -> dict:
    """Fetch last tweets of a specified userId."""  #martinshkreli #312653402  1859681514415607930
    data = await make_twitter_endpoint_request("user/last_tweets", {"userId": userId})
    tweets = extract_tweets(data)
    sim_tweets = simple_tweet_fields(tweets)
    if not tweets:
        return structured_response([], f"🔍 No tweets found for user: {userId}", "error")
    return structured_response(sim_tweets, f"get_user_last_tweets: {len(sim_tweets)}")

@mcp.tool()
async def advanced_search_twitter(llm_text: str) -> dict:
    """Search latest Twitter using advanced search operators based on LLM-extracted query.
    Args:
        query: Natural language or Twitter search query (supports operators like from:, to:, #hashtag, etc.)
    """
    formatted_query = await extract_search_query_from_llm_text(llm_text)
    data = await make_twitter_endpoint_request("tweet/advanced_search", {"queryType": "Top", "query": formatted_query})

    if not data:
        return structured_response([],"❌ Failed to fetch tweets. Check API key or network.", "error")
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)
    return structured_response(tweets, f"advanced_search {len(tweets)} tweets in thread for {formatted_query}")

@mcp.tool()
async def get_info_by_username(userName: str) -> dict:
    """Get user info by userName"""  #martinshkreli 
    data = await make_twitter_endpoint_request("user/info", {"userName": userName})
    if not data:
        return structured_response({}, "❌ Failed to fetch user info.", "error")
    user_info = data.get("data")
    return structured_response(user_info, f"fetched user batch inf")


@mcp.resource("history://history")
def get_available_history() -> str:
    """
    List all available topic folders in the history directory.
    
    This resource provides a simple list of all available history topic folders.
    """
    folders = []

    # Get all topic directories
    if os.path.exists(History_DIR):
        for topic_dir in os.listdir(History_DIR):
            topic_path = os.path.join(History_DIR, topic_dir)
            if os.path.isdir(topic_path):
                history_file = os.path.join(topic_path, "history_info.json")
                if os.path.exists(history_file):
                    folders.append(topic_dir)

    # Create a simple markdown list
    content = "# Available Topics\n\n"
    if folders:
        for folder in folders:
            content += f"- {folder}\n"
        content += f"\nUse @{folder} to access history in that topic.\n"
    else:
        content += "No topics found.\n"

    return content

@mcp.resource("history://{topic}")
def get_topic_history(topic: str) -> str:
    """
    Get detailed information about query/response history on a specific topic.

    Args:
        topic: The topic to retrieve history for
    """
    topic_dir = topic.lower().replace(" ", "_")
    history_file = os.path.join(History_DIR, topic_dir, "history_info.json")

    if not os.path.exists(history_file):
        return f"# No history found for topic: {topic}\n\nTry submitting a query in this topic first."

    try:
        with open(history_file, 'r') as f:
            history_data = json.load(f)

        # Create markdown content with history details
        content = f"# History for {topic.replace('_', ' ').title()}\n\n"
        content += f"Total entries: {len(history_data)}\n\n"

        for entry_time, record in history_data.items():
            content += f"## {entry_time}\n"
            content += f"- **Query**: {record.get('query', 'N/A')}"
            content += f"- **Tool**: {record.get('Tool', 'N/A')} {record.get('Tool_input', 'N/A')} \n"
            content += f"- **Result**:\n```\n{record.get('result', '')[:1000]}\n```\n"
            content += "---\n"

        return content
    except json.JSONDecodeError:
        return f"# Error reading history data for {topic}\n\nThe history data file is corrupted."



if __name__ == "__main__":
    mcp.run(transport="stdio")
