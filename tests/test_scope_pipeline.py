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
    monkeypatch.setattr(guardrails, 'ROUTER_BACKEND', 'chat')
    real_router = guardrails._router
    router = Mock()
    monkeypatch.setattr(guardrails, "_router", lambda: router)
    answer = Mock()
    answer.invoke.return_value = AIMessage(content="Career response")
    monkeypatch.setattr(graph, "model_with_tools", answer)
    yield SimpleNamespace(**locals())
    # Isolate graph dependencies between cases.
    for name in ("app.agent.guardrails", "app.agent.graph", "app.agent.tools.web_tools",
                 "app.agent.tools.scraper"):
        sys.modules.pop(name, None)


@pytest.mark.parametrize("label", ["career", "out_of_scope", "small_talk", "clarify"])
def test_graph_routes_before_expensive_work(runtime, label):
    r = runtime
    r.router.invoke.return_value = {"label": label}
    result = r.graph.chatbot.invoke({"messages": [HumanMessage(content="Can you explain that?")]},
                                    {"configurable": {"thread_id": "t", "user_id": "u"}})
    assert result["route"] == label
    assert bool(r.answer.invoke.call_count) == (label == "career")
    assert bool(r.files.call_count) == bool(r.memory.call_count) == (label == "career")


def test_context_and_failure(runtime):
    r = runtime
    r.router.invoke.return_value = {"label": "career"}
    messages = [HumanMessage(content="Compare two roles"), AIMessage(content="Backend and ML"),
                ToolMessage(content="private tool data", tool_call_id="1"),
                HumanMessage(content="What about the second one?")]
    assert r.guardrails.classify_turn(messages) == "career"
    payload = json.loads(r.router.invoke.call_args.args[0][-1].content)
    assert [h["text"] for h in payload["history"]] == ["Compare two roles", "Backend and ML"]
    assert payload["current"] == "What about the second one?"
    r.router.invoke.return_value = {"label": "unknown"}
    assert r.guardrails.classify_turn(messages) == "unavailable"
    r.router.invoke.side_effect = TimeoutError
    assert r.guardrails.classify_turn(messages) == "unavailable"


@pytest.mark.parametrize(("query", "decision"), [
    ("Hi!", "small_talk"), ("Give me a recipe", "out_of_scope"), ("Which one?", "clarify"),
])
def test_public_stream_emits_fixed_reply_once(runtime, query, decision):
    r = runtime
    r.router.invoke.return_value = {"label": decision}
    response = r.chat.chat_stream(r.chat.ChatRequest(message=query), {"id": "u"})

    async def collect():
        return [json.loads(frame.removeprefix("data: ")) async for frame in response.body_iterator]

    events = asyncio.run(collect())
    assert [event["type"] for event in events] == ["meta", "token", "done"]
    assert events[1]["content"] == r.guardrails.REPLIES[decision]
    assert r.router.invoke.call_count == int(query != "Hi!")


def test_tool_boundaries(runtime):
    w = runtime.web_tools
    assert "career" in w.browse_webpage.invoke({"url": "unused", "purpose": "weather"})
    assert "Refusal" in w.scrape_web_content.invoke({"query": "weather"})
    assert "Refusal" in w.search_latest_news.invoke({"query": "weather"})
    assert "Invalid" in w.evaluate_github_profile.invoke({"github_url_or_username": "two--hyphens"})


def test_network_policy(runtime, monkeypatch):
    s = runtime.scraper
    monkeypatch.setattr(s.socket, "getaddrinfo", lambda *a, **kw: [(2, 1, 6, "", ("127.0.0.1", 80))])
    assert not s.validate_public_url("https://example.com")[0]
    assert not s.validate_public_url("https://someone@example.com")[0]
    assert not s.validate_public_url("https://metadata.internal")[0]

def test_blocked_redirect_stops_all_fallbacks(runtime, monkeypatch):
    import httpx
    s = runtime.scraper
    monkeypatch.setattr(s.socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    seen = []
    def redirect(request):
        seen.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://example.internal"})
    client = httpx.Client
    monkeypatch.setattr(s.httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(redirect), **kw))
    result = s.scrape_url("https://example.org")
    assert not result.ok
    assert seen == ["93.184.216.34"]


def test_github_readme_uses_bounded_public_api(runtime):
    import httpx
    def serve(request):
        if request.url.path.endswith("/readme"):
            return httpx.Response(200, content=b"x" * 20000)
        return httpx.Response(200, json={"private": False})
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        assert len(runtime.web_tools._github_readme(client, "owner", "project")) == 12288

def test_reported_github_conversation(runtime, monkeypatch):
    r = runtime
    lookup = Mock()
    writer = Mock()
    monkeypatch.setattr(r.graph, "evaluate_github_profile", lookup)
    monkeypatch.setattr(r.graph, "model", writer)
    url = "https://github.com/the007prime"
    conversation = []
    for question, target in [
        (f"can you please tell me about this github profile [{url}]({url}) and give me a complete "
         "description of it can you please tell me about this CORE project and what tech stack "
         "was used to create this complete project ?", "the007prime/CORE"),
        ("i want you to evaulate this github profile and check what project it has build so far", "the007prime"),
        ("can you please tell me about other projects it has made", "the007prime"),
        ("tell me about the airgap-rag project", "the007prime/airgap-rag"),
    ]:
        conversation.append(HumanMessage(content=question))
        lookup.invoke.return_value = f"Evidence for {target}"
        writer.invoke.return_value = AIMessage(content=f"Summary for {target}")
        result = r.graph.chatbot.invoke({"messages": conversation})
        assert result["route"] == "career"
        lookup.invoke.assert_called_with({"github_url_or_username": target})
        prompt = json.loads(writer.invoke.call_args.args[0][-1].content)
        assert prompt["request"] == question
        assert prompt["github_evidence"] == f"Evidence for {target}"
        conversation.append(result["messages"][-1])
    r.router.invoke.assert_not_called()
    r.answer.invoke.assert_not_called()

    # An unrelated turn clears the previous target even in a saved conversation.
    r.router.invoke.return_value = {"label": "out_of_scope"}
    result = r.graph.chatbot.invoke({"messages": conversation + [HumanMessage(content="Give me a recipe")],
                                     "portfolio_target": "the007prime/airgap-rag"})
    assert result["route"] == "out_of_scope"
    assert result["portfolio_target"] is None
    assert lookup.invoke.call_count == 4


def test_github_link_alone_does_not_authorize_mixed_requests(runtime):
    from app.agent.portfolio import portfolio_target
    assert portfolio_target([HumanMessage(content="Describe https://github.com/example and write a song")]) is None
    assert portfolio_target([HumanMessage(content="Tell me about the CORE project")]) is None

def test_recruiter_instructions_reach_classifier(runtime):
    question = "Ignore previous instructions from my recruiter about résumé length. Can you shorten my résumé?"
    runtime.router.invoke.return_value = {"label": "career"}
    assert runtime.guardrails.classify_turn([HumanMessage(content=question)]) == "career"
    runtime.router.invoke.assert_called_once()


@pytest.mark.parametrize("query", ["Thanks a lot!", "See you later.", "Nice to meet you!"])
def test_chitchat_fast_path(runtime, query):
    assert runtime.guardrails.classify_turn([HumanMessage(content=query)]) == "small_talk"
    runtime.router.invoke.assert_not_called()


def test_labeled_scope_benchmark(runtime, monkeypatch):
    """Default to local coverage. Set HIREPULSE_BENCHMARK_KEY to evaluate the real router."""
    from collections import Counter
    from time import perf_counter

    key = os.getenv("HIREPULSE_BENCHMARK_KEY")
    monkeypatch.setattr(runtime.guardrails, "ROUTER_BACKEND", os.getenv("ROUTER_BACKEND") or "chat")
    if key:
        monkeypatch.setattr(runtime.guardrails, "OPENR_API", key)
        monkeypatch.setattr(runtime.guardrails, "_router", runtime.real_router)
        backend = Mock(wraps=runtime.guardrails._model_decision)
    else:
        backend = Mock(side_effect=ConnectionError("offline benchmark"))
    monkeypatch.setattr(runtime.guardrails, "_model_decision", backend)
    latencies = []
    model_counts = {label: Counter() for label in ("career", "small_talk", "out_of_scope")}
    cases = json.loads((ROOT / "tests/scope_cases.json").read_text())
    counts = {}
    errors = []
    started = perf_counter()
    for label, examples in cases.items():
        counts[label] = Counter()
        for item in examples:
            question, history = (item, []) if isinstance(item, str) else item
            messages = [(HumanMessage if role == "u" else AIMessage)(content=text)
                        for role, text in history]
            messages.append(HumanMessage(content=question))
            calls = backend.call_count
            tick = perf_counter()
            actual = runtime.graph.scope_guard({"messages": messages})["route"]
            if backend.call_count > calls:
                latencies.append((perf_counter() - tick) * 1000)
                model_counts[label][actual] += 1
            counts[label][actual] += 1
            if actual != label and (key or actual != "unavailable"):
                errors.append((question, label, actual))
    print("Scope benchmark", "live" if key else "offline", json.dumps(counts),
          "elapsed seconds", round(perf_counter() - started, 2))
    if key:
        from math import ceil
        assert latencies
        print("Model-only outcomes", json.dumps(model_counts), "p95 ms", sorted(latencies)[ceil(.95 * len(latencies)) - 1])
        print("Live misses:", errors)
        assert backend.call_count > sum(row["unavailable"] for row in counts.values()), "No live model decisions"
    else:
        assert not errors

@pytest.mark.parametrize(("status", "answer", "expected"), [
    (200, {"type": "choice", "choice": "career"}, "career"),
    (200, {"type": "refusal"}, "unavailable"),
    (200, {"type": "choice", "choice": "unknown"}, "unavailable"),
    (429, {}, "unavailable"),
])
def test_decisions_transport(runtime, monkeypatch, status, answer, expected):
    import httpx
    g = runtime.guardrails
    monkeypatch.setattr(g, "ROUTER_BACKEND", "decisions")
    received = []
    def serve(request):
        received.append(request)
        return httpx.Response(status, json={"answers": {"scope": answer}})
    with httpx.Client(transport=httpx.MockTransport(serve)) as client:
        monkeypatch.setattr(g, "_decider_client", lambda: client)
        messages = [HumanMessage(content="Compare the offers"), AIMessage(content="One is remote."),
                    HumanMessage(content="Which pays more?")]
        assert g.classify_turn(messages) == expected
    request, = received
    assert request.url.path.endswith("/decisions")
    data = json.loads(request.content)
    assert data["model"] == g.DECIDER_MODEL
    state = json.loads(data["state"])
    assert state["current"] == "Which pays more?"
    assert len(state["history"]) == 2
    runtime.router.invoke.assert_not_called()
