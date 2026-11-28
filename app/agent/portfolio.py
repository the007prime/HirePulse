"""Resolve common public portfolio requests to an owner or owner/repository."""
import re

from langchain_core.messages import HumanMessage

OWNER = r"[a-z0-9](?:[a-z0-9-]{0,37}[a-z0-9])?"
REPO = r"[a-z0-9_][a-z0-9._-]{0,99}"
LINK = re.compile(rf"(?<![\w./])(?:https://)?github\.com/(?P<owner>{OWNER})(?:/(?P<repo>{REPO}))?/?(?=$|[\s)\],!?])", re.I)
PREFIX = r"(?:(?:can|could|would) you |i want you to )?(?:please )?"
ASK = PREFIX + r"(?:tell me about|describe|summarize|review|evaluate|evaulate|explain|list|show me)"
DETAIL = r"(?: and (?:give me )?(?:a )?(?:complete |detailed )?(?:description|overview)(?: of it)?)?"
STACK = r"(?: and (?:what|which) (?:tech stack|technologies)(?: (?:was|were) used)?(?: to (?:create|build) (?:this|the)(?: complete)? project)?)?"
PROJECT = ASK + rf" (?:the |this |their )?(?P<name>{REPO}) (?:project|repo|repository)" + STACK
PROFILE = ASK + r" (?:the |this |their )?(?:github (?:profile|portfolio) ?)?<github>" + DETAIL
OTHER = ASK + r" (?:the |its |their |his |her )?(?:other|more|remaining|additional) (?:projects|repos|repositories)(?: (?:it|they|he|she) (?:has|have) (?:made|built))?"
PROFILE_ONLY = ASK + r" (?:the |this |their )?github profile(?: and (?:check|show) (?:what|which) projects? (?:it|they) (?:has|have) (?:build|built|made)(?: so far)?)?"


def _previous_owner(messages):
    # User-supplied links are authoritative, including in existing checkpoints.
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            owners = {m["owner"].lower() for m in LINK.finditer(message.content)}
            if owners:
                return owners.pop() if len(owners) == 1 else None
    return None


def portfolio_target(messages):
    if not messages or not isinstance(messages[-1], HumanMessage):
        return None
    raw = messages[-1].content
    if not isinstance(raw, str):
        return None
    raw = re.sub(r"\[([^\]]*)\]\((https://github\.com/[^)\s]+)\)", r"\2", raw, flags=re.I)
    raw = " ".join(raw.strip().rstrip(" ?!.,").split())
    links = list(LINK.finditer(raw))
    if len(links) > 1:
        return None
    request = LINK.sub("<github>", raw)
    if links:
        link = links[0]
        if re.fullmatch(PROFILE, request, re.I):
            return "/".join(filter(None, (link["owner"], link["repo"])))
        combined = re.fullmatch(PROFILE + r"(?:[.?!]\s*| (?:and )?)" + PROJECT, request, re.I)
        if combined and not link["repo"]:
            return f"{link['owner']}/{combined['name']}"
        return None
    named = re.fullmatch(PROJECT, request, re.I)
    if named or re.fullmatch(OTHER, request, re.I) or re.fullmatch(PROFILE_ONLY, request, re.I):
        owner = _previous_owner(messages[:-1])
        if owner:
            return f"{owner}/{named['name']}" if named else owner
    return None
