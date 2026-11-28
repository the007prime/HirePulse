"""HirePulse Chatbot Agent Module.

Provides the compiled LangGraph chatbot and conversational memory interfaces.
"""

from app.agent.graph import (
    chatbot,
    get_thread_messages,
    graph,
    model,
    model_with_tools,
    Chatnode,
)
from app.memory.stm import (
    checkpointer,
    convo,
    count_tokens,
    get_threads,
    safe_trim,
    update_summary,
)
from app.agent.tools import tools
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.guardrails import (
    evaluate_scope,
    STANDARD_REFUSAL_MESSAGE,
    JAILBREAK_REFUSAL_MESSAGE,
)

__all__ = [
    "chatbot",
    "checkpointer",
    "get_thread_messages",
    "get_threads",
    "model",
    "model_with_tools",
    "tools",
    "graph",
    "Chatnode",
    "convo",
    "count_tokens",
    "safe_trim",
    "update_summary",
    "SYSTEM_PROMPT",
    "evaluate_scope",
    "STANDARD_REFUSAL_MESSAGE",
    "JAILBREAK_REFUSAL_MESSAGE",
]

if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    assert count_tokens([HumanMessage(content="test")]) > 0
    assert len(safe_trim([HumanMessage(content="a"), AIMessage(content="b")], max_tokens=100)) == 2

    usr_inp = input("What is your question: ")
    res = chatbot.invoke(
        {"messages": [HumanMessage(content=usr_inp)]},
        config={"configurable": {"thread_id": "test_cli"}},
    )
    print("Response:", res["messages"][-1].content)
