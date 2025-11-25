# main.py
import os
import json
from typing import List, Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools import TavilyTool, LLMTool, TurnMemory, JsonMemory
from agent import create_agent

load_dotenv()


# -------------------------
# FastAPI init
# -------------------------
app = FastAPI(title="Tavily Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Global objects
# -------------------------
turn_mem = TurnMemory(max_turns=40)
json_mem = JsonMemory(max_items=20)

tavily_tool = TavilyTool()
llm_tool = LLMTool()
agent = create_agent(tavily_tool, llm_tool)

MAX_TURNS_CONTEXT = 12


# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    user_id: str
    session_id: str
    message: str


# -------------------------
# Helpers
# -------------------------
def build_chat_template(history: List[Dict[str, str]], latest_query: str, max_turns: int = 12) -> str:
    """
    Simple text template for the LLM router – no markdown.
    """
    pruned = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    for m in pruned:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    lines.append(f"User: {latest_query}")
    return "\n".join(lines)


def get_json_history_str(user_id: str, session_id: str) -> str:
    hist = json_mem.get_history(user_id, session_id)
    if not hist:
        return ""
    try:
        return json.dumps(hist[-10:], ensure_ascii=False)
    except Exception:
        return ""


# -------------------------
# Endpoint
# -------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    # 1) store user turn
    turn_mem.save_turn(req.user_id, req.session_id, "user", req.message)
    recent = turn_mem.load_recent(req.user_id, req.session_id, n=MAX_TURNS_CONTEXT)
    chat_template = build_chat_template(recent, req.message, max_turns=MAX_TURNS_CONTEXT)

    json_history = get_json_history_str(req.user_id, req.session_id)

    state_in = {
        "user_id": req.user_id,
        "session_id": req.session_id,
        "llm_prompt": chat_template,          # for Groq router
        "search_query": req.message,          # for Tavily
        "latest_user_input": req.message,
        "last_structured_output": json_history,
    }

    try:
        out = agent.invoke(state_in)
        structured = out.get("structured_output") or out
    except Exception as e:
        print("AGENT ERROR:", repr(e))
        structured = {
            "query": req.message,
            "final_answer": "Sorry, something went wrong while processing that request.",
            "citations": [],
            "meta": {"engine": "error", "error": str(e)},
        }

    # 2) persist structured JSON in JSON-memory
    json_mem.append(req.user_id, req.session_id, structured)

    # 3) store assistant turn for conversational memory
    turn_mem.save_turn(
        req.user_id,
        req.session_id,
        "assistant",
        structured.get("final_answer", ""),
    )

    return structured


# -------------------------
# Local runner
# -------------------------
def run_api(host: str | None = None, port: str | None = None):
    import uvicorn

    host = host or os.getenv("HOST", "127.0.0.1")
    port = int(port or os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import sys

    h = sys.argv[1] if len(sys.argv) > 1 else None
    p = sys.argv[2] if len(sys.argv) > 2 else None
    run_api(h, p)
