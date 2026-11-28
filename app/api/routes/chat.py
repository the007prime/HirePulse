import json
import uuid
from fastapi import APIRouter
from fastapi import Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.memory.stm import is_thread_owner, register_user_thread
from app.auth.security import get_current_user
from app.agent.guardrails import REPLIES, classify_turn
from app.agent.graph import chatbot, langfuse_handler
from app.api.schemas.chat import ChatRequest

router = APIRouter()


@router.post("/chat")
def chat_stream(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    thread_id = req.thread_id or str(uuid.uuid4())
    if req.thread_id and not is_thread_owner(current_user["id"], thread_id):
        raise HTTPException(status_code=403, detail="Unauthorized access")

    register_user_thread(current_user["id"], thread_id)
    config = {
        "configurable": {
            "thread_id": thread_id,
            "user_id": current_user["id"]
        }
    }

    def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'meta', 'thread_id': thread_id})}\n\n"
            incoming = HumanMessage(content=req.message)
            prior = chatbot.get_state(config).values.get("messages", []) if req.thread_id else []
            route = classify_turn([*prior, incoming])
            if route in {"out_of_scope", "unavailable"}:
                yield f"data: {json.dumps({'type': 'token', 'content': REPLIES[route]})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
                return
            config["configurable"]["validated_scope"] = route
            for mode, event in chatbot.stream(
                {"messages": [incoming]},
                config=config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "updates":
                    reply = (event.get("scope_reply") or {}).get("messages", [])
                    if reply:
                        yield f"data: {json.dumps({'type': 'token', 'content': reply[0].content})}\n\n"
                    continue
                chunk, metadata = event
                node = (metadata or {}).get("langgraph_node")
                if node != "Chatnode" or "career_answer" not in metadata.get("tags", []):
                    continue
                if isinstance(chunk, (AIMessageChunk, AIMessage)) and chunk.content:
                    content = chunk.content
                    if isinstance(content, list):
                        text = "".join(
                            b if isinstance(b, str) else b.get("text", "")
                            for b in content
                            if isinstance(b, (str, dict))
                        )
                    else:
                        text = str(content)
                    if text:
                        yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
