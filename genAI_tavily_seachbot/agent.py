# agent.py
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, AnyUrl, ValidationError
from langgraph.graph import StateGraph, END
import re
from textwrap import shorten

from tools import TavilyTool, LLMTool


class Citation(BaseModel):
    title: Optional[str] = None
    url: AnyUrl
    snippet: Optional[str] = None
    score: Optional[float] = None
    image_url: Optional[AnyUrl] = None


class AgentOutput(BaseModel):
    query: str
    final_answer: str
    citations: List[Citation] = []
    meta: Dict[str, Any] = {}


class AgentState(TypedDict, total=False):
    user_id: str
    session_id: str

    llm_prompt: str
    search_query: str
    latest_user_input: str

    last_structured_output: str

    raw_llm_answer: str
    needs_web: bool

    search_results: Dict[str, Any]
    used_sources: List[Dict[str, Any]]
    web_summary: str

    structured_output: Dict[str, Any]


def _strip_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _normalize_sources(results: List[Dict[str, Any]], max_sources: int = 6) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for r in results:
        url = (r.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = r.get("title") or url
        snippet = (
            r.get("raw_content")
            or r.get("content")
            or r.get("snippet")
            or ""
        ).strip()

        snippet = re.sub(r"\b\d+\s*(subscribers?|likes?|views?|posted)\b", "", snippet, flags=re.IGNORECASE)
        snippet = re.sub(r"(###|##|#)\s*\b(description|video|channel)\b.*?(?=\s|$)", "", snippet, flags=re.IGNORECASE)
        snippet = _strip_ws(snippet)
        snippet = shorten(snippet, width=1200, placeholder="…")

        out.append(
            {
                "url": url,
                "title": title,
                "snippet": snippet,
                "image_url": r.get("thumbnail") or r.get("image_url"),
            }
        )
        if len(out) >= max_sources:
            break
    return out


def llm_router(state: AgentState, llm: LLMTool) -> AgentState:
    latest = _strip_ws(state.get("latest_user_input") or "")
    prompt_text = state.get("llm_prompt") or latest
    json_history = state.get("last_structured_output") or ""

    system_msg = (
        "You are a research assistant. You can answer from your knowledge, "
        "and if you are not confident or the question is clearly about current events, "
        "you must respond with exactly the token CALL_TAVILY.\n\n"
        "You also receive a compact JSON history of previous answers. "
        "Use it to answer follow-ups when possible to avoid new web searches.\n\n"
        "JSON_HISTORY:\n"
        f"{json_history[:2000]}\n"
        "END_JSON_HISTORY\n\n"
        "If the question is clearly about *today's* news, dates, very recent events, "
        "or if you lack enough info, reply with only CALL_TAVILY."
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt_text},
    ]

    answer = llm.chat(messages, max_tokens=600)
    needs_web = answer.strip() == "CALL_TAVILY"

    return {
        "raw_llm_answer": "" if needs_web else answer,
        "needs_web": needs_web,
    }


def search_web(state: AgentState, tools: TavilyTool) -> AgentState:
    query = _strip_ws(state.get("search_query") or state.get("latest_user_input") or "")
    if not query:
        return {"search_results": {"results": [], "answer": ""}}

    sr = tools.search(
        query=query,
        max_results=6,
        include_answer=True,
        include_raw_content=True,
    )
    return {"search_results": sr}


def prepare_sources(state: AgentState) -> AgentState:
    results = (state.get("search_results") or {}).get("results", []) or []
    passages = _normalize_sources(results, max_sources=6)
    state["used_sources"] = passages
    return state


def summarize_with_llm(state: AgentState, llm: LLMTool) -> AgentState:
    """
    Take Tavily results + user query + JSON history and ask the LLM for a clean answer.

    Style rules:
    - For incident / news / 'latest updates' questions: 2–4 short paragraphs + optional bullet list
      with key points or timeline (no markdown headings).
    - For simple factual questions (e.g., 'who is X', 'what is', 'when was'): 1 short sentence
      plus 2–5 bullet points with the core facts.
    - Never hallucinate beyond the Tavily docs; if something isn't in the docs, say you couldn't find it.
    """
    query = state.get("search_query") or state.get("latest_user_input") or ""
    tavily_answer = (state.get("search_results") or {}).get("answer") or ""
    docs = state.get("used_sources") or []
    json_history = state.get("last_structured_output") or ""

    corpus_parts = []
    for i, d in enumerate(docs, start=1):
        corpus_parts.append(f"[{i}] {d.get('title')}\nURL: {d.get('url')}\n{d.get('snippet')}")
    corpus = "\n\n".join(corpus_parts)

    sys = (
        "You are a precise research summarizer. You must ONLY use the Tavily documents and factual info.\n"
        "If something is not in the sources, clearly say you couldn't find it.\n\n"
        "Answer style rules:\n"
        "- If the question asks for 'latest updates', 'what happened', 'give me the details', "
        "or otherwise describes an incident or ongoing case, write 2–4 short paragraphs in a clear, "
        "story-like style. After that, you may add a short bullet list with the most important facts "
        "or a brief timeline. No markdown headings.\n"
        "- If the question is a direct factual query like 'who is the doctor arrested', "
        "'who is X', 'what is', 'when was', respond with one concise sentence plus 2–5 bullet points "
        "highlighting the key facts. Keep it tight, no fluff.\n"
        "- Never use markdown headings such as '#', '##', '###'. Bullets with '-' or '•' are fine.\n\n"
        "You are also given a compact JSON history of previous answers; use it for context and consistency, "
        "but do not invent facts beyond the Tavily sources.\n\n"
        f"JSON_HISTORY:\n{json_history[:2000]}\nEND_JSON_HISTORY"
    )

    user = (
        f"User question: {query}\n\n"
        f"Tavily suggested answer (may be rough):\n{tavily_answer}\n\n"
        f"DOCUMENT CORPUS:\n{corpus}\n\n"
        "Now write the final answer following the style rules above."
    )

    messages = [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

    summary = llm.chat(messages, max_tokens=900)
    state["web_summary"] = summary
    return state


def finalize_chat(state: AgentState) -> AgentState:
    query = state.get("latest_user_input") or ""
    web_summary = state.get("web_summary")
    raw_llm_answer = state.get("raw_llm_answer")

    if web_summary:
        final = web_summary
        engine = "llm+tavily"
    else:
        final = raw_llm_answer or "I couldn't find detailed information for that request."
        engine = "llm_only"

    citations: List[Citation] = []
    for s in state.get("used_sources") or []:
        try:
            citations.append(
                Citation(
                    title=s.get("title") or None,
                    url=s.get("url"),
                    snippet=s.get("snippet") or None,
                    image_url=s.get("image_url") or None,
                )
            )
        except ValidationError:
            continue

    payload = AgentOutput(
        query=query,
        final_answer=final,
        citations=citations,
        meta={
            "engine": engine,
            "num_sources": len(citations),
        },
    )
    state["structured_output"] = payload.model_dump()
    return state


def route_decision(state: AgentState) -> str:
    return "search_web" if state.get("needs_web") else "finalize"


def create_agent(tavily: TavilyTool, llm: LLMTool) -> Any:
    workflow = StateGraph(AgentState)

    workflow.add_node("llm_router", lambda s: llm_router(s, llm))
    workflow.add_node("search_web", lambda s: search_web(s, tavily))
    workflow.add_node("prepare_sources", prepare_sources)
    workflow.add_node("summarize_with_llm", lambda s: summarize_with_llm(s, llm))
    workflow.add_node("finalize", finalize_chat)

    workflow.set_entry_point("llm_router")
    workflow.add_conditional_edges(
        "llm_router",
        route_decision,
        {
            "search_web": "search_web",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("search_web", "prepare_sources")
    workflow.add_edge("prepare_sources", "summarize_with_llm")
    workflow.add_edge("summarize_with_llm", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()
