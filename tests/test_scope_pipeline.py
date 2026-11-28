"""Exercise the branch's real graph and SSE handler without external services."""
import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime(monkeypatch):
    def module(name, **attrs):
        mod = ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, mod)
        spec.loader.exec_module(mod)
        return mod

    class State(MessagesState, total=False):
        portfolio_target: str | None
        linked_pages: list[dict]
        route: str
        summary: str
        summarized_index: int

    files, memory = Mock(return_value=[]), Mock(return_value=[])
    monkeypatch.setenv("OPENR_API", "offline-test-key")
    module("app.agent", __path__=[str(ROOT / "app/agent")])
    module("app.agent.tools", __path__=[str(ROOT / "app/agent/tools")], tools=[])
    module("app.memory.stm", convo=State, checkpointer=None, get_threads=Mock(),
           update_summary=Mock(), safe_trim=lambda msgs, **kw: msgs, get_thread_messages=Mock(),
           is_thread_owner=Mock(return_value=True), register_user_thread=Mock())
    module("app.memory.ltm", ltm_store=SimpleNamespace(search_memories=memory))
    module("app.rag.pipeline", rag_system=SimpleNamespace(get_thread_files=files))
    module("app.auth.security", get_current_user=Mock())
    module("app.api", __path__=[str(ROOT / "app/api")])
    module("app.api.schemas", __path__=[str(ROOT / "app/api/schemas")])
    from app.agent import guardrails
    from app.agent.tools import scraper, web_tools
    from app.agent import graph
    chat = load("standalone_chat_route", "app/api/routes/chat.py")
    real_client = guardrails._decider_client
    router = Mock()
    def post(url, *, headers, json):
        label = router.invoke(json)["label"]
        response = Mock()
        scores = {name: float(name == label) for name in json["questions"]["scope"]["criteria"]}
        response.json.return_value = {"answers": {"scope": {"type": "choice", "choice": label, "probabilities": scores}}}
        return response

    monkeypatch.setattr(guardrails, "_decider_client", lambda: SimpleNamespace(post=post))
    answer = Mock()
    answer.invoke.return_value = AIMessage(content="Career response")
    monkeypatch.setattr(graph, "model_with_tools", answer)
    yield SimpleNamespace(**locals())
    # Isolate graph dependencies between cases.
    for name in ("app.agent.guardrails", "app.agent.graph", "app.agent.tools.web_tools",
                 "app.agent.tools.scraper"):
        sys.modules.pop(name, None)



@pytest.mark.parametrize("route", ["out_of_scope", "unavailable"])
def test_rejection_precedes_graph(runtime, monkeypatch, route):
    r = runtime
    monkeypatch.setattr(r.chat, "classify_turn", lambda messages: route)
    stream = Mock(side_effect=AssertionError("Rejected turn reached the graph"))
    monkeypatch.setattr(r.chat.chatbot, "stream", stream)
    response = r.chat.chat_stream(r.chat.ChatRequest(message="Weather at https://example.org"), {"id": "u"})
    async def collect():
        return [json.loads(frame.removeprefix("data: ")) async for frame in response.body_iterator]
    events = asyncio.run(collect())
    assert [e["type"] for e in events] == ["meta", "token", "done"]
    assert events[1]["content"] == r.guardrails.REPLIES[route]
    stream.assert_not_called()


def test_accepted_portfolio_fetches_one_copy_per_link(runtime, monkeypatch):
    r = runtime
    url = "https://example.org/work"
    other = "https://www.example.net"
    query = f"Compare my portfolio [{url}]({url}) with www.example.net."
    fetch = Mock(return_value=[SimpleNamespace(url=url, content="My projects", ok=True)])
    monkeypatch.setattr(r.graph, "scrape_urls", fetch)
    r.graph.chatbot.invoke({"messages": [HumanMessage(content=query)]},
                           {"configurable": {"validated_scope": "career"}})
    assert fetch.call_args.args[0] == [url, other]
    data = json.loads(r.answer.invoke.call_args.args[0][-1].content)
    assert data["linked_pages"] == [{"url": url, "content": "My projects"}, {"url": other, "content": None}]
    r.router.invoke.assert_not_called()
