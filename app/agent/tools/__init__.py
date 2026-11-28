from app.agent.tools.rag_tools import search_knowledge_base
from app.agent.tools.memory_tools import save_user_memory, recall_user_memories
from app.agent.tools.web_tools import (
    browse_webpage,
    evaluate_github_profile,
    scrape_web_content,
    search_latest_news,
)

# Default toolset available to the agent
tools = [
    search_knowledge_base,
    save_user_memory,
    recall_user_memories,
    search_latest_news,
    browse_webpage,
    scrape_web_content,
    evaluate_github_profile,
]

__all__ = [
    "tools",
    "search_knowledge_base",
    "save_user_memory",
    "recall_user_memories",
    "search_latest_news",
    "browse_webpage",
    "scrape_web_content",
    "evaluate_github_profile",
]
