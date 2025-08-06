import os, json
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from openai import AsyncOpenAI
#from anthropic import AsyncAnthropic
from datetime import datetime

# Load environment variables
load_dotenv()
History_DIR = "history"

# Initialize FastMCP
mcp = FastMCP(name="social-server")
model = os.getenv("LLM_MODEL")
#"claude-3-5-haiku-20241022"
#"claude-3-7-sonnet-20250219"

simplify_tweet_flag = False

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

async def _extract_with_openai(llm_text: str, api_key: str, ctx: Context | None = None) -> str:
    client = AsyncOpenAI(api_key=api_key)

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
        return response.choices[0].message.content[0].text.strip()
    except Exception as e:
        if ctx:
            await ctx.error(f"Anthropic extraction failed: {e}")
        raise

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
    llm_key = os.getenv("OPENAI_API_KEY")

    if llm_key:
        try:
            return await _extract_with_openai(llm_text, llm_key, ctx)
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
        "message": message,
        "data": data if data is not None else [],
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
        if (simplify_tweet_flag) == True:
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
        else: return tweet

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

### User Endpoint
@mcp.tool()
async def get_batch_user_info_by_ids(userIds: list) -> dict:
    """Fetch user info for multiple user IDs."""
    # [44196397, 312653402] #elon, martin
    data = await make_twitter_endpoint_request("user/batch_info_by_ids", {
        "userIds": ",".join(str(uid) for uid in userIds)
    })
    users = data.get("users", [])
    return structured_response(users, f"get_batch_user_info_by_ids")

@mcp.tool()
async def get_info_by_username(userName: str) -> dict:
    """Get user info by userName"""  #martinshkreli 
    data = await make_twitter_endpoint_request("user/info", {"userName": userName})
    if not data:
        return structured_response({}, "❌ Failed to fetch user info.", "error")
    user_info = data.get("data")
    return structured_response(user_info, f"fetched user batch inf")

@mcp.tool()
async def get_user_last_tweets(userId: str) -> dict:
    """Fetch last tweets of a specified userId."""  #martinshkreli #312653402  1859681514415607930
    data = await make_twitter_endpoint_request("user/last_tweets", {"userId": userId})
    tweets = extract_tweets(data)
    sim_tweets = simple_tweet_fields(tweets)
    if not tweets:
        return structured_response([], f"🔍 No tweets found for user: {userId}", "error")
    return structured_response(sim_tweets, f"get_user_last_tweets: {len(sim_tweets)}")

 #"\n---\n".join(format_tweet(t, {}) for t in tweets)

@mcp.tool()
async def search_user_by_keyword(keyword: str) -> dict:
    """Search users by keyword (e.g. name, bio, etc.)."""
    data = await make_twitter_endpoint_request("user/search", {"query": keyword})
    users = data.get("users", [])
    if not users:
        return structured_response([], f"❌ No users found for keyword: {keyword}", "error")
    return structured_response(users, f"Found {len(users)} users for keyword: {keyword}") # "\n".join(format_user(u) for u in users)

### Tweet Endpoint
    """
    Get the full thread context for a given tweet ID.
    Returns sorted tweet thread as a list of tweet objects.
    """ #1949929951244792263 from @biotech2k1 (1244781749458153480)
    data = await make_twitter_endpoint_request("tweet/thread_context", {"tweetId": tweetId})
    thread_tweets = data.get("tweets", [])

    if not thread_tweets:
        return structured_response([], f"❌ No thread context found for tweet: {tweetId}", "error")

    sorted_tweets = sort_tweets_by_created_at(thread_tweets)
    thread_data = merge_thread_into_single_tweet(sorted_tweets)
    return structured_response(thread_data, f"Found {len(thread_tweets)} tweets in thread for {tweetId}")

@mcp.tool()
async def get_article(tweetId: str) -> dict:
    """Extract article or summary from tweet if available."""
    data = await make_twitter_endpoint_request("article", {"tweet_id": tweetId})
    article = data.get("article")
    if not article:
        return structured_response(None, f"❌ No article content found for tweet: {tweetId}", "error")
    return structured_response(article, f"Extracted article for tweet: {tweetId}")
    """Fetch recent tweets from a Twitter Community."""
    data = await make_twitter_endpoint_request("community/tweets", {"community_id": communityId})
    tweets = data.get("tweets", [])
    if not tweets:
        return structured_response([], f"❌ No tweets found for community: {communityId}", "error")
    return structured_response(tweets, f"Found {len(tweets)} tweets in community: {communityId}")

### Trend Endpoint

@mcp.tool()
async def get_trends(woeid: int = 1) -> dict:
    """
    Get trending topics by WOEID (default: 1 = Worldwide).
    23424977 = United States
    Returns structured trend data directly from API.
    """
    data = await make_twitter_endpoint_request("trends", {"woeid": str(woeid)})
    trends = data.get("trends", [])
    if not trends:
        return structured_response([], f"❌ No trends found for WOEID {woeid}", "error")

    valid_trends = [
        t for t in trends
        if isinstance(t, dict) and "trend" in t and isinstance(t["trend"], dict)
    ]
    return structured_response(valid_trends, f"Found {len(valid_trends)} trending topics for WOEID {woeid}")

@mcp.prompt()
def deep_search(topic: str) -> str:
    """
    Prompt template for analyzing a user query using KOL opinions, FRED data, archives, and economic reasoning and return in Chinese.
    """
    return f"""
**"{topic}"** analysis:

Return in Chinese

Please perform the following steps in order:
---
### 1. 🧵 Identify Key Twitter Opinions
- Search for the most relevant and recent tweets from **trusted KOLs** on the topic.
- Include only those with high engagement (retweets, likes) or influence.
- Display **up to 3 tweets** with:
  - Tweet content
  - Username
  - Timestamp

### 2. 📊 Extract Economic Data (FRED)
- Identify related **economic indicators** (GDP, inflation, unemployment, interest rates, etc.).
- Retrieve and summarize the latest data from **FRED** relevant to the query.
- Present them in a table or list with:
  - Indicator name
  - Value
  - Date
  - Source link (FRED series ID)

### 3. 📚 Retrieve Related Archived Insights (if applicable)
- If the question is **not time-sensitive** (e.g. not about current markets or breaking news), search scholarly archives, reports, or previous analyses.
- Return **up to 3 studies or articles** with:
  - Title
  - Source
  - Short summary
  - Link (if available)

### 4. 📈 Form a Conclusion
- Provide a reasoned conclusion or summary using:
  - Insights from KOL tweets
  - FRED economic data
  - Archived knowledge (if used)
- If applicable, **explain using economic models** (e.g. supply/demand, monetary policy, cost-benefit, game theory, etc.).
- Clearly state any **assumptions** or uncertainties.
---
Output the result as structured JSON:

```json
{{
  "tweets": [{{ "username": "...", "text": "...", "timestamp": "..." }}],
  "economic_data": [{{ "indicator": "...", "value": "...", "date": "...", "source": "..." }}],
  "archived_insights": [{{ "title": "...", "summary": "...", "source": "...", "link": "..." }}],
  "conclusion": "..."
}}"""

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
