import os
import httpx
from typing import Any
from dotenv import load_dotenv
from fastmcp import FastMCP, Context
from anthropic import AsyncAnthropic

# Load environment variables
load_dotenv()

# Initialize FastMCP
mcp = FastMCP(name="twitter-server")
model = "claude-3-7-sonnet-20250219"


async def make_twitter_request(query: str, ctx: Context | None = None) -> list[dict[str, Any]] | None:
    """
    Query Twitter API via api.twitterapi.io and return formatted tweet texts.

    Args:
        query (str): A Twitter search query (e.g., "from:elonmusk #AI").
        ctx (Optional[Context]): FastMCP context object for tracing/logging.

    Returns:
        Optional[list[str]]: List of formatted tweet strings, or None on failure.
    """
    api_key = os.getenv("TWITTER_API_KEY")
    if not api_key:
        if ctx:
            await ctx.error("❌ Missing TWITTER_API_KEY in environment.")
        return None

    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.twitterapi.io/twitter/tweet/advanced_search"
            headers = {"X-API-Key": api_key}
            params = {"queryType": "Top", "query": query}
            response = await client.get(url, headers=headers, params=params)
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


def format_tweet(tweet: dict, users: dict = None) -> str:
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


@mcp.tool()
async def search_latest_twitter(llm_text: str, ctx: Context) -> str:
    """Search latest Twitter using advanced search operators based on LLM-extracted query.
    Args:
        query: Natural language or Twitter search query (supports operators like from:, to:, #hashtag, etc.)
    """
    formatted_query = await extract_search_query_from_llm_text(llm_text, ctx)
    
    await ctx.info(f"📡 Querying Twitter with: {formatted_query}")

    data = await make_twitter_request(formatted_query, ctx)
    if not data:
        return "❌ Failed to fetch tweets. Check API key or network."

    tweets = data.get("tweets", [])
    if not tweets:
        return f"🔍 No tweets found for: {llm_text}"

    users_by_id = {}
    for tweet in tweets:
        author = tweet.get("author", {})
        author_id = author.get("id")
        if author_id:
            users_by_id[author_id] = author

    results = [format_tweet(tweet, users_by_id) for tweet in tweets]
    await ctx.info(f"✅ Retrieved {len(results)} tweets.")
    return "\n---\n".join(results)


if __name__ == "__main__":
    mcp.run(transport="stdio")
