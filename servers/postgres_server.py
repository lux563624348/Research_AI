#!/usr/bin/env python3
import os
import json
import logging
import asyncpg
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from fastmcp import FastMCP
from db_config import DB_CONFIG

# ── Logging ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Database URL ──
DATABASE_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ── Config ──
class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str = Field(default=DATABASE_URL, description="PostgreSQL connection URL")
    max_rows: int = Field(default=int(os.getenv("MAX_ROWS", "1000")), description="Max rows per query")
    query_timeout: int = Field(default=int(os.getenv("QUERY_TIMEOUT", "30")), description="Query timeout in seconds")


# ── DB Connection ──

class DatabaseConnection:
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def init_pool(self):
        if not self.pool:
            self.pool = await asyncpg.create_pool(
                dsn=self.config.url,
                min_size=1,
                max_size=10,
                command_timeout=self.config.query_timeout,
            )
            logger.info("Database connection pool initialized.")

    async def close_pool(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed.")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        if not self.pool:
            await self.init_pool()
        async with self.pool.acquire() as conn:
            yield conn


# ── MCP Server ──
mcp = FastMCP("PostgreSQL MCP Server")
db_config = DatabaseConfig()
db_connection = DatabaseConnection(db_config)

@mcp.tool()
async def get_userid_by_username(username: str) -> str:
    """
    Look up the internal twitter_user.id given a username (case-insensitive).

    Args:
        username: Twitter username to search for.

    Returns:
        The user ID as a string if found, or "Username not found".
        On error, returns a JSON string with error details.
    """
    query = """
    SELECT user_id FROM "twitter_user"
    WHERE LOWER(username) = LOWER($1)
    LIMIT 1;
    """
    try:
        async with db_connection.get_connection() as conn:
            row = await conn.fetchrow(query, username.strip())
            if row:
                return row["user_id"]
            return "Username not found"
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "username": username.strip(),
        })


@mcp.tool()
async def get_sample_tweets_user(username: str, limit: int = 3) -> str:
    """
    Retrieve sample tweets for a given Twitter username.

    Args:
        username: Twitter username to search for (case-insensitive).
        limit: Maximum number of tweets to return (default: 3, max: 10).

    Returns:
        JSON string containing tweet data with columns and rows,
        or an error/message if no tweets found or on failure.
    """
    limit = min(limit, 10)
    tweet_query = """
    SELECT * FROM twitter
    WHERE LOWER(username) = LOWER($1)
    ORDER BY twitter_datetime DESC
    LIMIT $2;
    """
    try:
        async with db_connection.get_connection() as conn:
            tweet_rows = await conn.fetch(tweet_query, username, limit)

            if not tweet_rows:
                return json.dumps({
                    "username": username,
                    "row_count": 0,
                    "message": "No tweets found for this user"
                })

            columns = list(tweet_rows[0].keys())
            rows = [
                [
                    col.isoformat() if hasattr(col, "isoformat") else col
                    for col in (row[col_name] for col_name in columns)
                ]
                for row in tweet_rows
            ]

            return json.dumps({
                "username": username,
                "row_count": len(rows),
                "columns": columns,
                "rows": rows,
            }, indent=2)
    except Exception as e:
        return json.dumps({
            "username": username,
            "error": str(e),
        })

if __name__ == "__main__":
    mcp.run(transport='stdio')
