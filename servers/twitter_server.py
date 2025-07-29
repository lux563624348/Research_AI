import os, json
import httpx
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from anthropic import AsyncAnthropic
from datetime import datetime

# Load environment variables
load_dotenv()
History_DIR = "history"

# Initialize FastMCP
mcp = FastMCP(name="twitter-server")
model = os.getenv("LLM_MODEL")
#"claude-3-5-haiku-20241022"
#"claude-3-7-sonnet-20250219"

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
async def advanced_search_twitter(llm_text: str) -> str:
    """Search latest Twitter using advanced search operators based on LLM-extracted query.
    Args:
        query: Natural language or Twitter search query (supports operators like from:, to:, #hashtag, etc.)
    """
    formatted_query = await extract_search_query_from_llm_text(llm_text)
    data = await make_twitter_endpoint_request("advanced_search", {"queryType": "Top", "query": formatted_query})

    if not data:
        return "❌ Failed to fetch tweets. Check API key or network."
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)

    return {
        "status": "success",
        "msg": f"advanced_search {len(tweets)} tweets in thread for {formatted_query}",
        "tweets": tweets
    }

### User Endpoint
@mcp.tool()
async def get_batch_user_info_by_ids(userIds: list) -> dict:
    """Fetch user info for multiple user IDs."""
    # [44196397, 312653402] #elon, martin
    data = await make_twitter_endpoint_request("user/batch_info_by_ids", {
        "userIds": ",".join(str(uid) for uid in userIds)
    })
    users = data.get("users", [])
    return {
        "status": "success",
        "msg": "fetched user batch info",
        "users": users
    }

@mcp.tool()
async def get_info_by_username(userName: str) -> dict:
    """Get user info by userName"""  #martinshkreli 
    data = await make_twitter_endpoint_request("user/info", {"userName": userName})
    if not data:
        return "❌ Failed to fetch user info."
    user_info = data.get("data")
    return {
        "status": "success",
        "msg": "fetched user batch info",
        "user_info": user_info
    }

@mcp.tool()
async def get_user_last_tweets(userId: str) -> dict:
    """Fetch last tweets of a specified userId."""  #martinshkreli #312653402  1859681514415607930
    data = await make_twitter_endpoint_request("user/last_tweets", {"userId": userId})
    tweets = extract_tweets(data)
    sim_tweets = simple_tweet_fields(tweets)
    if not tweets:
        return f"🔍 No tweets found for user: {userId}"
    return {
        "status": "success",
        "msg": "get_user_last_tweets",
        "tweets": sim_tweets
    }
 #"\n---\n".join(format_tweet(t, {}) for t in tweets)

@mcp.tool()
async def get_user_followers(userName: str, pageSize: int = 20) -> dict:
    """Fetch a page of followers for a given userName. """
    data = await make_twitter_endpoint_request("user/followers", 
                    {"userName": userName, "pageSize": pageSize})
    followers = data.get("followers", [])
    if not followers:
        return structured_response([], f"❌ No followers found for user: {userName}", "error")
    return structured_response(followers, f"Found {len(followers)} followers for {userName}") #"\n".join(format_user(u) for u in followers)

@mcp.tool()
async def get_user_followings(userName: str, pageSize: int = 20) -> dict:
    """Fetch a page of followings (people the user follows)."""
    data = await make_twitter_endpoint_request("user/followings",
                    {"userName": userName, "pageSize": pageSize})
    followings = data.get("followings", [])
    if not followings:
        return structured_response([], f"❌ No followings found for user: {userName}", "error")
    return structured_response(followings, f"Found {len(followings)} followings for {userName}") #"\n".join(format_user(u) for u in followings)

@mcp.tool()
async def get_user_mentions(userName: str) -> dict:
    """get tweet mentions by user userName""" ##martinshkreli 
    data = await make_twitter_endpoint_request("user/mentions", {"userName": userName})
    tweets = data.get("tweets", [])
    if not tweets:
        return structured_response([], f"❌ No mentions found for user: {userName}", "error")
    return structured_response(tweets, f"Found {len(tweets)} mentions for {userName}") # "\n---\n".join(format_tweet(t, {}) for t in tweets)

@mcp.tool()
async def check_follow_relationship(sourceUserName: str, targetUserName: str) -> str:
    """
    Check whether one user follows another and vice versa.

    Returns a human-readable summary like:
    - ✅ @sourceUserName follows @targetUserName
    - ❌ @targetUserName does not follow @sourceUserName
    """
    #realdonaldtrump
    #elonmusk
    data = await make_twitter_endpoint_request("user/check_follow_relationship", {
        "source_user_name": sourceUserName,
        "target_user_name": targetUserName
    })

    if not data:
        return structured_response({}, f"❌ Failed to check relationship between @{sourceUserName} and @{targetUserName}", "error")

    result = {
        "source_follows_target": data.get("following", False),
        "target_follows_source": data.get("followed_by", False)
    }
    return structured_response(result, f"Relationship fetched for {sourceUserName} ↔ {targetUserName}") # "\n".join(result_lines)

@mcp.tool()
async def search_user_by_keyword(keyword: str) -> str:
    """Search users by keyword (e.g. name, bio, etc.)."""
    data = await make_twitter_endpoint_request("user/search", {"query": keyword})
    users = data.get("users", [])
    if not users:
        return structured_response([], f"❌ No users found for keyword: {keyword}", "error")
    return structured_response(users, f"Found {len(users)} users for keyword: {keyword}") # "\n".join(format_user(u) for u in users)

### Tweet Endpoint

@mcp.tool()
async def get_tweets_by_IDs(tweetIds: list[str]) -> str:
    """Fetch tweet details by a list of tweet IDs."""
    data = await make_twitter_endpoint_request("tweets", {
        "tweet_ids": ",".join(str(tid) for tid in tweetIds)
    })
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)

    if not tweets:
        return structured_response([], "❌ No tweets found for the given IDs.", "error")
    return structured_response(tweets, f"Found {len(tweets)} tweets")

@mcp.tool()
async def get_tweet_replies(tweetId: str) -> str:
    """Get replies to a specific tweet.  Must be an original tweet (not a reply to another tweet) and should be the first tweet in a thread"""
    data = await make_twitter_endpoint_request("tweet/replies", {"tweetId": tweetId})
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)
    if not tweets:
        return structured_response([], f"❌ No replies found for tweet: {tweetId}", "error")
    return structured_response(tweets, f"Found {len(tweets)} replies") 

@mcp.tool()
async def get_tweet_quotations(tweetId: str) -> str:
    """Get quote tweets referencing a specific tweet."""
    data = await make_twitter_endpoint_request("tweet/quotes", {"tweetId": tweetId})
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)
    if not tweets:
        return structured_response([], f"❌ No quotations found for tweet: {tweetId}", "error")
    return structured_response(tweets, f"Found {len(tweets)} quotations")

@mcp.tool()
async def get_tweet_retweeters(tweetId: str) -> str:
    """Get users who retweeted a given tweet."""
    data = await make_twitter_endpoint_request("tweet/retweeters", {"tweetId": tweetId})
    users = data.get("users", [])
    if not users:
        return structured_response([], f"❌ No retweeters found for tweet: {tweetId}", "error")
    return structured_response(users, f"Found {len(users)} retweeters")

## formating of thread with context
def sort_tweets_by_created_at(tweets: list) -> list:
    # Identify author ID of the first tweet
    first_author_id = tweets[0].get("author", {}).get("id")
    # Filter tweets by same author
    same_author_tweets = [
        tweet for tweet in tweets
        if tweet.get("author", {}).get("id") == first_author_id
    ]
    # Sort by createdAt
    return sorted(
        same_author_tweets,
        key=lambda x: datetime.strptime(x["createdAt"], "%a %b %d %H:%M:%S %z %Y")
    )

def merge_thread_into_single_tweet(sorted_thread: list) -> dict:
    base = sorted_thread[0]
    # Concatenate all tweet texts
    full_text = "\n\n".join(tweet.get("text", "") for tweet in sorted_thread)
    full_id = "\n".join(tweet.get("id", "") for tweet in sorted_thread)
    merged = {
        "type": base.get("type"),
        "id": full_id,
        "url": base.get("twitterUrl") or base.get("url"),
        "text": full_text,
        "createdAt": base.get("createdAt"),
        "lang": base.get("lang"),
        "bookmarkCount": base.get("bookmarkCount"),
        "isReply": base.get("isReply"),
        "inReplyToId": base.get("inReplyToId"),
        "conversationId": base.get("conversationId"),
        "inReplyToUserId": base.get("inReplyToUserId"),
        "inReplyToUsername": base.get("inReplyToUsername"),
        "author": base.get("author"),
        "extendedEntities": base.get("extendedEntities"),
        "card": base.get("card"),
        "place": base.get("place"),
        "entities": base.get("entities"),
        "quoted_tweet": base.get("quoted_tweet"),
        "retweeted_tweet": base.get("retweeted_tweet")
    }

    return {"data": [merged]}

@mcp.tool() ##!! the real reponse 200 is not same as claimed
async def get_tweet_thread_context(tweetId: str) -> dict:
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

### Community Endpoint

@mcp.tool()
async def get_community_info_by_id(communityId: str) -> dict:
    """Fetch metadata about a Twitter Community by its ID."""
    data = await make_twitter_endpoint_request("community/info", {"community_id": communityId})
    info = data.get("community_info", {})
    if not info:
        return structured_response({}, f"❌ No metadata found for community: {communityId}", "error")
    return structured_response(info, f"Fetched metadata for community: {communityId}")

@mcp.tool()
async def get_community_members(communityId: str) -> dict:
    """Fetch members of a Twitter Community."""
    data = await make_twitter_endpoint_request("community/members", {"community_id": communityId})
    members = data.get("members", [])
    if not members:
        return structured_response([], f"❌ No members found in community: {communityId}", "error")
    return structured_response(members, f"Found {len(members)} members in community: {communityId}")

@mcp.tool()
async def get_community_moderators(communityId: str) -> dict:
    """Get moderators of a Twitter Community."""
    data = await make_twitter_endpoint_request("community/moderators", {"community_id": communityId})
    moderators = data.get("moderators", [])
    if not moderators:
        return structured_response([], f"❌ No moderators found for community: {communityId}", "error")
    return structured_response(moderators, f"Found {len(moderators)} moderators for community: {communityId}")

@mcp.tool()
async def get_community_tweets(communityId: str) -> dict:
    """Fetch recent tweets from a Twitter Community."""
    data = await make_twitter_endpoint_request("community/tweets", {"community_id": communityId})
    tweets = data.get("tweets", [])
    if not tweets:
        return structured_response([], f"❌ No tweets found for community: {communityId}", "error")
    return structured_response(tweets, f"Found {len(tweets)} tweets in community: {communityId}")

### List Endpoint

@mcp.tool()
async def get_list_tweets(listId: str) -> dict:
    """Get tweets from a specific Twitter List."""
    data = await make_twitter_endpoint_request("list/tweets", {"listId": listId})
    tweets = extract_tweets(data)
    tweets = simple_tweet_fields(tweets)

    if not tweets:
        return structured_response([], f"❌ No tweets found in list: {listId}", "error")
    return structured_response(tweets, f"Found {len(tweets)} tweets in list: {listId}")

@mcp.tool()
async def get_list_followers(listId: str) -> dict:
    """Get followers of a Twitter List."""
    data = await make_twitter_endpoint_request("list/followers", {"list_id": listId})
    followers = data.get("followers", [])
    if not followers:
        return structured_response([], f"❌ No followers found for list: {listId}", "error")
    return structured_response(followers, f"Found {len(followers)} followers in list: {listId}")

@mcp.tool()
async def get_list_members(listId: str) -> dict:
    """Fetch members of a Twitter List."""
    data = await make_twitter_endpoint_request("list/members", {"list_id": listId})
    members = data.get("members", [])
    if not members:
        return structured_response([], f"❌ No members found in list: {listId}", "error")
    return structured_response(members, f"Found {len(members)} members in list: {listId}")

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
