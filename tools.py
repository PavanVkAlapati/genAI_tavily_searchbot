# tools.py
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv
from tavily import TavilyClient
from groq import Groq

load_dotenv()


# -------------------------
# Tavily wrapper
# -------------------------
@dataclass
class TavilyTool:
    client: TavilyClient

    def __init__(self) -> None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError("TAVILY_API_KEY not set")
        self.client = TavilyClient(api_key=api_key)

    def search(
        self,
        query: str,
        max_results: int = 6,
        include_answer: bool = True,
        include_raw_content: bool = True,
    ) -> Dict[str, Any]:
        return self.client.search(
            query=query,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )


# -------------------------
# Groq LLM wrapper
# -------------------------
@dataclass
class LLMTool:
    client: Groq
    model: str = "llama-3.1-8b-instant"

    def __init__(self, model: str = "llama-3.1-8b-instant") -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()


# -------------------------
# Turn memory (for chat history)
# -------------------------
class TurnMemory:
    """
    Very simple in-process memory of last N turns per (user_id, session_id).
    Stored as a list[{"role": "user"|"assistant", "content": str}].
    """

    def __init__(self, max_turns: int = 24) -> None:
        self.store: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
        self.max_turns = max_turns

    def save_turn(self, user_id: str, session_id: str, role: str, content: str) -> None:
        key = (user_id, session_id)
        convo = self.store.setdefault(key, [])
        convo.append({"role": role, "content": content})
        if len(convo) > self.max_turns:
            # keep only the most recent
            self.store[key] = convo[-self.max_turns :]

    def load_recent(self, user_id: str, session_id: str, n: int) -> List[Dict[str, str]]:
        key = (user_id, session_id)
        convo = self.store.get(key, [])
        return convo[-n:]


# -------------------------
# JSON memory (for structured outputs)
# -------------------------
class JsonMemory:
    """
    Keeps a list of past structured outputs (the JSON we send back) per conversation.
    """

    def __init__(self, max_items: int = 20) -> None:
        self.store: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        self.max_items = max_items

    def get_history(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        return self.store.get((user_id, session_id), [])

    def append(self, user_id: str, session_id: str, item: Dict[str, Any]) -> None:
        key = (user_id, session_id)
        lst = self.store.setdefault(key, [])
        lst.append(item)
        if len(lst) > self.max_items:
            self.store[key] = lst[-self.max_items :]
