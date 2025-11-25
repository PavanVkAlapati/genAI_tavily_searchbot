from langgraph.graph import StateGraph, END
from tavily import TavilyClient
import os
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, AnyUrl, ValidationError
import re
from textwrap import shorten
import json
import asyncio

# ---------------------------
# Strict output schema
# ---------------------------
class Citation(BaseModel):
    title: Optional[str] = None
    url: AnyUrl
    snippet: Optional[str] = None
    score: Optional[float] = None

class AgentOutput(BaseModel):
    query: str
    final_answer: str
    citations: List[Citation] = []
    meta: Dict[str, Any] = {}

# ---------------------------
# Shared graph state
# ---------------------------
class AgentState(TypedDict, total=False):
    question: str
    search_results: dict
    used_sources: List[Dict[str, Any]]
    structured_output: Dict[str, Any]
    # kept only for backward compatibility (we don't render markdown)
    answer: str

# ---------------------------
# Helpers
# ---------------------------
def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _normalize_sources(results: List[Dict[str, Any]], max_sources: int = 6) -> List[Dict[str, Any]]:
    seen, passages = set(), []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = r.get("title") or url
        snippet = (r.get("raw_content") or r.get("content") or r.get("snippet") or "").strip()
        # Remove social metrics & markdown artifacts
        snippet = re.sub(r"\b\d+\s*(subscribers?|likes?|views?|posted)\b", "", snippet, flags=re.IGNORECASE)
        snippet = re.sub(r"(###|##|#)\s*\b(description|video|channel)\b.*?(?=\s|$)", "", snippet, flags=re.IGNORECASE)
        snippet = _strip_ws(snippet)
        snippet = shorten(snippet, width=1200, placeholder="…")
        passages.append({"url": url, "title": title, "snippet": snippet})
        if len(passages) >= max_sources:
            break
    return passages

# ---------------------------
# Nodes
# ---------------------------
def search_web(state: AgentState) -> AgentState:
    client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    search_results = client.search(
        query=state["question"],
        max_results=6,
        include_answer=False,
        include_raw_content=True
    )
    return {"search_results": search_results}

def prepare_sources(state: AgentState) -> AgentState:
    results = (state.get("search_results") or {}).get("results", []) or []
    passages = _normalize_sources(results, max_sources=6)
    state["used_sources"] = passages
    return state

def finalize_answer(state: AgentState) -> AgentState:
    question = state.get("question") or state.get("query") or ""
    assert question, "state requires 'question' (or 'query')"

    llm_answer = state.get("answer") or ""
    if llm_answer:
        # scrub headings to keep it plain text
        plain = "\n".join(line.lstrip("# ").rstrip() for line in llm_answer.splitlines())
    else:
        passages = state.get("used_sources") or []
        plain = (
            "Here is a concise summary based on top sources. See citations for details."
            if passages else "No results found. Try a broader query."
        )

    citations: List[Citation] = []
    for s in state.get("used_sources") or []:
        try:
            citations.append(Citation(
                title=s.get("title") or None,
                url=s.get("url"),
                snippet=s.get("snippet") or None,
                score=None
            ))
        except ValidationError:
            # skip malformed entries (e.g., invalid URL)
            continue

    payload = AgentOutput(
        query=question,
        final_answer=plain,
        citations=citations,
        meta={
            "engine": "tavily",
            "graph": "langgraph",
            "num_sources": len(citations)
        }
    )

    # IMPORTANT: Make it JSON-serializable (avoids AnyUrl serialization error)
    state["structured_output"] = json.loads(payload.model_dump_json())
    return state

# ---------------------------
# Build graph
# ---------------------------
def create_agent():
    workflow = StateGraph(AgentState)
    workflow.add_node("search", search_web)
    workflow.add_node("prepare_sources", prepare_sources)
    workflow.add_node("finalize", finalize_answer)

    workflow.set_entry_point("search")
    workflow.add_edge("search", "prepare_sources")
    workflow.add_edge("prepare_sources", "finalize")
    workflow.add_edge("finalize", END)
    return workflow.compile()

agent = create_agent()

# ---------------------------
# Async-safe runner
# ---------------------------
async def main():
    q = "what are the latest updates on delhi red fort blast"
    out = agent.invoke({"question": q})
    structured = out.get("structured_output") or out
    print(json.dumps(structured, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
