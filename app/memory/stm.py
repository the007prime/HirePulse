import logging
from typing import Dict, List, Optional
from typing_extensions import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.message import add_messages
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


# 1. State Schema for Short-Term / Working Memory
class convo(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    portfolio_target: Optional[str]
    route: str
    summarized_index: int


# 2. Checkpointer setup (PostgreSQL persistence scoped to thread_id)
pool = ConnectionPool(
    conninfo=DATABASE_URL,
    max_size=20,
    open=True,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
conn = pool  # Backwards compatibility

checkpointer = PostgresSaver(conn=pool)


def init_db() -> None:
    """Initialize PostgresSaver checkpoint migrations and metadata tables."""
    try:
        checkpointer.setup()
        with pool.connection() as c:
            with c.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS hidden_threads (
                        thread_id TEXT PRIMARY KEY,
                        hidden_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );"""
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        email TEXT UNIQUE NOT NULL,
                        hashed_password TEXT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );"""
                )
                cur.execute(
                    """
                
                    CREATE TABLE IF NOT EXISTS user_threads (
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        thread_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
    except Exception as exc:
        logger.warning("PostgresSaver initialization warning: %s", exc)


init_db()


# 3. Token estimation helper
def count_tokens(messages: list[BaseMessage]) -> int:
    """Rough token estimator: ~4 characters per token + 4 per message."""
    total = 0
    for m in messages:
        c = m.content if isinstance(m.content, str) else str(m.content)
        total += len(c) // 4 + 4
    return total


# 4. Progressive thread summarizer
def update_summary(existing_summary: str, new_msgs: list[BaseMessage], model) -> str:
    """Incrementally update the rolling summary of the current thread."""
    convo_lines = []
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            convo_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and m.content:
            convo_lines.append(f"Assistant: {m.content}")

    convo_text = "\n".join(convo_lines).strip()
    if not convo_text:
        return existing_summary

    prompt = (
        "Summarize the key facts, user preferences, and context from this conversation concisely (under 150 words):\n"
    )
    if existing_summary:
        prompt = (
            f"Current conversation summary:\n{existing_summary}\n\n"
            f"Update the summary with these new messages, keeping it concise (under 150 words):\n{convo_text}"
        )
    else:
        prompt += convo_text

    try:
        res = model.invoke([SystemMessage(content=prompt)])
        return res.content.strip() if isinstance(res.content, str) else str(res.content)
    except Exception:
        return existing_summary


# 5. Sliding window token trimmer
def safe_trim(msgs: list[BaseMessage], max_tokens: int = 2500) -> list[BaseMessage]:
    """Trim message list to stay strictly within token budget while preserving recent context."""
    try:
        trimmed = trim_messages(
            msgs,
            max_tokens=max_tokens,
            strategy="last",
            token_counter=count_tokens,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
        if trimmed:
            return trimmed
    except Exception:
        pass
    # Fallback: keep last 8 messages if trimming fails
    return msgs[-8:] if len(msgs) >= 8 else msgs


# 6. Thread retrieval & soft-delete helpers
def hide_thread(thread_id: str) -> None:
    """Soft-delete: hide a thread from the frontend while preserving history in the database."""
    with pool.connection() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                INSERT INTO hidden_threads (thread_id) VALUES (%s)
                ON CONFLICT (thread_id) DO NOTHING
                """,
                (str(thread_id),),
            )


def get_hidden_threads() -> set[str]:
    """Retrieve all thread IDs that have been hidden/removed from the frontend."""
    with pool.connection() as c:
        with c.cursor() as cur:
            cur.execute("SELECT thread_id FROM hidden_threads")
            rows = cur.fetchall()
            return {r["thread_id"] if isinstance(r, dict) else r[0] for r in rows}


def get_threads(include_hidden: bool = False) -> list[str]:
    """List unique thread IDs recorded in the checkpointer, ordered by latest activity."""
    hidden = set() if include_hidden else get_hidden_threads()
    with pool.connection() as c:
        with c.cursor() as cur:
            cur.execute(
                """
                SELECT thread_id, MAX(checkpoint->>'ts') as latest_ts
                FROM checkpoints
                GROUP BY thread_id
                ORDER BY latest_ts DESC NULLS LAST
                """
            )
            rows = cur.fetchall()
            all_threads = [
                r["thread_id"] if isinstance(r, dict) else r[0]
                for r in rows
            ]
            return [tid for tid in all_threads if tid not in hidden]


def _extract_text_content(content) -> str:
    """Normalize message content to a safe string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get("text", "") or str(b))
            else:
                parts.append(str(b))
        return "".join(parts)
    return str(content) if content is not None else ""


def get_thread_messages(chatbot, thread_id: str) -> list[dict]:
    """Retrieve and format all messages for a given thread."""
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    raw_messages = state.values.get("messages", []) if state and state.values else []
    formatted = []
    for msg in raw_messages:
        if isinstance(msg, HumanMessage):
            text = _extract_text_content(msg.content)
            formatted.append({"role": "user", "content": text})
        elif isinstance(msg, AIMessage) and msg.content:
            text = _extract_text_content(msg.content)
            if text:
                formatted.append({"role": "assistant", "content": text})
    return formatted

def register_user_thread(user_id: str, thread_id: str) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_threads (user_id, thread_id)
                VALUES (%s, %s)
                ON CONFLICT (thread_id) DO UPDATE SET created_at = NOW()
                """,
                (str(user_id), str(thread_id)),
            )

def is_thread_owner(user_id: str, thread_id: str) -> bool:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) AS cnt FROM user_threads 
                WHERE user_id = %s AND thread_id = %s""",
                (str(user_id), str(thread_id))
            )
            row = cur.fetchone()
            return row["cnt"] > 0

def get_user_threads(user_id: str) -> list[str]:
    hidden = get_hidden_threads()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT thread_id FROM user_threads 
                WHERE user_id = %s ORDER BY created_at DESC""",
                (str(user_id),)
            )
            rows = cur.fetchall()
            return [r["thread_id"] for r in rows if r["thread_id"] not in hidden]

