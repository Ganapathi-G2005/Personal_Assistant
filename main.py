"""
Sidekick AI — main.py
FastAPI + LangGraph 1.x + LangChain 1.x agentic backend.

Persistence (April 2026):
  • RAG chunks indexed to SQLite (`.sidekick_data/rag.sqlite`) — survives restarts.
  • LangGraph checkpoints via AsyncSqliteSaver (`.sidekick_data/checkpoints.sqlite`).
  • PDF text extraction uses `pypdf` (declare in dependencies).

Fixes applied (April 2026):
  • Playwright launch wrapped in asyncio.wait_for + sandbox flags;
    falls back to tavily-only if Playwright fails — server always starts.
  • graph / worker_llm_with_tools None-guard before every stream.
  • sanitize_tool_call_history drops dangling tool_calls to prevent OpenAI 400s.
  • aget_state wrapped in asyncio.wait_for — no more checkpoint deadlocks.
  • invoke_input correctly scoped: full init on new thread, lightweight on continue.
  • retry_count hard ceiling (MAX_RETRIES=5) prevents infinite eval loops.
  • SSE always emits a final 'done' event — frontend never hangs.
  • RAG upgraded to BM25-style scoring (replaces naive intersection ratio).
  • Structured logging replaces bare print() calls.
  • /api/reset accepts optional old thread_id to clean up uploaded docs.
  • Evaluator prompt tightened; worker prompt structured into clear sections.
"""

import asyncio
import logging
import math
import sqlite3
import sys
import json
import os
import re
import uuid
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.tools import tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from tavily import TavilyClient
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)

# ─────────────────────────────── Logging ─────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger("sidekick")

# ─────────────────────────────── Constants ───────────────────────────────────

MAX_RETRIES = 5
PLAYWRIGHT_TIMEOUT = 30   # seconds
STATE_TIMEOUT = 10        # seconds for aget_state
MAX_FILE_BYTES = 5 * 1024 * 1024
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
RAG_TOP_K = 4
MAX_CHUNKS_PER_SOURCE = 2
DEFAULT_SUCCESS_CRITERIA = (
    "Provide a helpful, accurate response to the user's latest request. "
    "Ask a clarifying question only if required details are missing."
)

# Local persistence (RAG chunks + LangGraph checkpoints). Survives process restarts.
DATA_DIR = Path(__file__).resolve().parent / ".sidekick_data"
RAG_DB_PATH = DATA_DIR / "rag.sqlite"
CHECKPOINT_DB_PATH = DATA_DIR / "checkpoints.sqlite"

ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv",
    ".html", ".htm", ".js", ".ts", ".css", ".log",
    ".pdf", ".docx",
}

# File types we explicitly don't attempt to parse as text.
# We "ignore" these with a reason so uploads never fail silently.
IGNORED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".exe", ".dll", ".so", ".dylib",
    ".xlsx", ".xls", ".pptx", ".ppt", ".doc", ".rtf",
}


def format_assistant_markdown(raw: str) -> str:
    """
    Normalize assistant markdown into a concise, consistent structure.

    Style goals:
      - Prefer `##` section headings for top-level sections.
      - Preserve existing markdown headings when already present.
      - Convert ad-hoc numeric / bold-only title lines into `##`.
      - Keep warnings visible using blockquotes.
    """
    text = (raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    lines = text.split("\n")
    normalized: List[str] = []

    heading_from_number = re.compile(r"^\s*\d+[\)\.\-:]\s+(.+?)\s*$")
    heading_from_bold = re.compile(r"^\s*\*\*(.+?)\*\*\s*$")
    warning_line = re.compile(
        r"^\s*(warning|important|caution|medical disclaimer)\s*:\s*(.+)$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            normalized.append("")
            continue

        # Preserve explicit markdown headings.
        if stripped.startswith("#"):
            normalized.append(stripped)
            continue

        number_match = heading_from_number.match(stripped)
        if number_match:
            normalized.append(f"## {number_match.group(1).strip()}")
            continue

        bold_match = heading_from_bold.match(stripped)
        if bold_match:
            normalized.append(f"## {bold_match.group(1).strip()}")
            continue

        warning_match = warning_line.match(stripped)
        if warning_match:
            label = warning_match.group(1).capitalize()
            body = warning_match.group(2).strip()
            normalized.append(f"> **{label}:** {body}")
            continue

        # Normalize list markers while preserving nesting/indentation.
        if stripped.startswith("* "):
            indent = len(line) - len(line.lstrip(" "))
            normalized.append(f"{' ' * indent}- {stripped[2:].strip()}")
            continue

        normalized.append(line.rstrip())

    # Clean up excessive blank lines (max one blank line between blocks).
    cleaned: List[str] = []
    prev_blank = False
    for line in normalized:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank

    return "\n".join(cleaned).strip()


# ─────────────────────────────── Pydantic Schemas ────────────────────────────


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(
        description="Whether the success criteria have been met"
    )
    user_input_needed: bool = Field(
        description=(
            "True if more input is needed from the user, or the assistant is "
            "stuck / repeating the same mistake"
        )
    )


class ChatRequest(BaseModel):
    message: str
    success_criteria: Optional[str] = None
    thread_id: Optional[str] = None


# ─────────────────────────────── LangGraph State ─────────────────────────────


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: NotRequired[Optional[str]]
    success_criteria_met: bool
    user_input_needed: bool
    retry_count: NotRequired[int]
    rag_context: NotRequired[str]


# ─────────────────────────────── RAG Pipeline ────────────────────────────────

thread_documents: Dict[str, List[Dict[str, str]]] = {}


def init_rag_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(RAG_DB_PATH), check_same_thread=False)
    try:
        conn.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                source TEXT NOT NULL,
                text TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rag_thread ON rag_chunks(thread_id);
            """
        )
        conn.commit()
    finally:
        conn.close()


def load_rag_from_disk() -> None:
    """Load indexed chunks into memory (used for BM25 retrieval)."""
    global thread_documents
    if not RAG_DB_PATH.exists():
        thread_documents = {}
        return
    conn = sqlite3.connect(str(RAG_DB_PATH), check_same_thread=False)
    try:
        cur = conn.execute("SELECT thread_id, source, text FROM rag_chunks ORDER BY id")
        loaded: Dict[str, List[Dict[str, str]]] = {}
        for row in cur:
            tid, source, text = row[0], row[1], row[2]
            loaded.setdefault(tid, []).append({"source": source, "text": text})
        thread_documents = loaded
        log.info("Loaded RAG index: %d thread(s), %d chunk(s)", len(loaded), sum(len(v) for v in loaded.values()))
    finally:
        conn.close()


def persist_rag_chunks(thread_id: str, filename: str, chunks: List[str]) -> None:
    if not chunks:
        return
    conn = sqlite3.connect(str(RAG_DB_PATH), check_same_thread=False)
    try:
        conn.executemany(
            "INSERT INTO rag_chunks (thread_id, source, text) VALUES (?, ?, ?)",
            [(thread_id, filename, c) for c in chunks],
        )
        conn.commit()
    finally:
        conn.close()


def delete_rag_for_thread(thread_id: str) -> None:
    global thread_documents
    if thread_id in thread_documents:
        del thread_documents[thread_id]
    if not RAG_DB_PATH.exists():
        return
    conn = sqlite3.connect(str(RAG_DB_PATH), check_same_thread=False)
    try:
        conn.execute("DELETE FROM rag_chunks WHERE thread_id = ?", (thread_id,))
        conn.commit()
    finally:
        conn.close()


def purge_checkpoint_thread(thread_id: str) -> None:
    """Remove LangGraph checkpoints for a thread (e.g. on reset)."""
    if not CHECKPOINT_DB_PATH.exists():
        return
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), timeout=30.0)
    try:
        conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        conn.commit()
    except sqlite3.OperationalError:
        pass
    finally:
        conn.close()


def chunk_text(text: str) -> List[str]:
    chunks: List[str] = []
    text = (text or "").strip()
    if not text:
        return chunks

    # Prefer paragraph/sentence-aware chunking to avoid cutting context mid-thought.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        # Paragraph alone is still too long; split by sentence boundaries first.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        if not sentences:
            sentences = [para]

        sentence_acc = ""
        for sentence in sentences:
            s_candidate = f"{sentence_acc} {sentence}".strip() if sentence_acc else sentence
            if len(s_candidate) <= CHUNK_SIZE:
                sentence_acc = s_candidate
                continue
            if sentence_acc:
                chunks.append(sentence_acc)
            sentence_acc = sentence
        if sentence_acc:
            chunks.append(sentence_acc)

    if current:
        chunks.append(current)

    # Add overlap for continuity between adjacent chunks.
    if CHUNK_OVERLAP > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-CHUNK_OVERLAP:].strip()
            if prev_tail:
                merged = f"{prev_tail}\n{chunks[i]}".strip()
                overlapped.append(merged[: CHUNK_SIZE + CHUNK_OVERLAP])
            else:
                overlapped.append(chunks[i])
        chunks = overlapped

    return [c for c in chunks if c.strip()]


def looks_like_meaningful_text(text: str) -> bool:
    """Heuristic to avoid indexing mostly-binary/garbage content."""
    t = (text or "").strip()
    if not t:
        return False
    # Check a small sample for letter density.
    sample = t[:5000]
    letters = sum(1 for ch in sample if ch.isalpha())
    if letters >= 20:
        return True
    # Fallback: alphanumeric density.
    alnum = sum(1 for ch in sample if ch.isalnum())
    return alnum >= 50


def _tokenize(text: str) -> List[str]:
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in text)
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "you", "your", "about", "into", "than", "then", "they", "them", "their",
        "have", "has", "had", "not", "but", "can", "could", "would", "should",
        "what", "when", "where", "which", "will", "just", "over", "under", "after",
    }
    tokens = [t for t in cleaned.split() if len(t) >= 2]
    return [t for t in tokens if t not in stopwords]


def _bm25_score(
    q_tokens: List[str],
    doc_tokens: List[str],
    idf_by_term: Dict[str, float],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """BM25 with per-thread IDF for better lexical relevance."""
    if not q_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    freq: Dict[str, int] = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in q_tokens:
        f = freq.get(qt, 0)
        if f == 0:
            continue
        idf = idf_by_term.get(qt, 0.0)
        if idf <= 0:
            continue
        score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(avg_dl, 1)))
    return score


def retrieve_thread_context(thread_id: str, query: str) -> str:
    docs = thread_documents.get(thread_id, [])
    if not docs:
        return ""
    q_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(d["text"]) for d in docs]
    avg_dl = sum(len(t) for t in tokenized_docs) / max(len(tokenized_docs), 1)

    # No lexical query tokens (or everything filtered): return representative
    # chunks while preserving source diversity.
    if not q_tokens:
        selected: List[tuple[float, Dict[str, str]]] = []
        per_source: Dict[str, int] = {}
        for doc in docs:
            src = doc["source"]
            if per_source.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            selected.append((0.0, doc))
            per_source[src] = per_source.get(src, 0) + 1
            if len(selected) >= RAG_TOP_K:
                break
        final_docs = selected
    else:
        n_docs = len(tokenized_docs)
        df: Dict[str, int] = {}
        for dt in tokenized_docs:
            for t in set(dt):
                df[t] = df.get(t, 0) + 1
        idf_by_term = {
            t: math.log(1 + ((n_docs - dft + 0.5) / (dft + 0.5)))
            for t, dft in df.items()
        }

        scored = [
            (_bm25_score(q_tokens, td, idf_by_term, avg_dl), doc)
            for td, doc in zip(tokenized_docs, docs)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        positive_scored = [(s, d) for s, d in scored if s > 0]
        candidate_docs = positive_scored if positive_scored else scored

        # Keep top-k relevant chunks but avoid single-source domination.
        selected: List[tuple[float, Dict[str, str]]] = []
        per_source: Dict[str, int] = {}
        for score, doc in candidate_docs:
            src = doc["source"]
            if per_source.get(src, 0) >= MAX_CHUNKS_PER_SOURCE:
                continue
            selected.append((score, doc))
            per_source[src] = per_source.get(src, 0) + 1
            if len(selected) >= RAG_TOP_K:
                break

        # Backfill if diversity constraints underfill top-k.
        if len(selected) < min(RAG_TOP_K, len(candidate_docs)):
            seen_ids = {id(d) for _, d in selected}
            for score, doc in candidate_docs:
                if id(doc) in seen_ids:
                    continue
                selected.append((score, doc))
                if len(selected) >= RAG_TOP_K:
                    break
        final_docs = selected

    snippets = [
        f"[Source {i}: {item['source']}]\n{item['text']}"
        for i, (_, item) in enumerate(final_docs, start=1)
    ]
    return "\n\n".join(snippets)


# ─────────────────────────────── Tools ───────────────────────────────────────

_playwright_context = None
async_browser = None
toolkit = None
tools: List[Any] = []

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


@tool
def tavily_search(query: str) -> str:
    """Search the web for up-to-date information using Tavily."""
    response = tavily_client.search(query=query, search_depth="basic")
    return str(response["results"])


# ─────────────────────────────── LLMs ────────────────────────────────────────

worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
worker_llm_with_tools: Optional[Any] = None  # bound post Playwright init

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)


# ─────────────────────────────── Message Utilities ───────────────────────────


def content_to_text(raw: Any) -> str:
    """Coerce any LangChain message content shape to a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: List[str] = []
        for item in raw:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(p for p in parts if p).strip()
    return str(raw)


def _role(msg: Any) -> str:
    r = (
        msg.get("role", msg.get("type", ""))
        if isinstance(msg, dict)
        else getattr(msg, "type", "")
    )
    return {"ai": "assistant", "human": "user"}.get(r, r)


def _tool_calls(msg: Any) -> List[Dict[str, Any]]:
    return (
        msg.get("tool_calls", []) or []
        if isinstance(msg, dict)
        else getattr(msg, "tool_calls", []) or []
    )


def _tool_call_id(msg: Any) -> str:
    return (
        msg.get("tool_call_id", "") or ""
        if isinstance(msg, dict)
        else getattr(msg, "tool_call_id", "") or ""
    )


def sanitize_tool_call_history(messages: List[Any]) -> List[Any]:
    """
    Remove assistant tool-call messages that have no matching tool responses.
    Prevents OpenAI 400 'invalid_request_error' for orphaned tool_calls.
    """
    sanitized: List[Any] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        if _role(msg) != "assistant":
            sanitized.append(msg)
            i += 1
            continue

        calls = _tool_calls(msg)
        if not calls:
            sanitized.append(msg)
            i += 1
            continue

        required_ids = {
            c["id"] for c in calls if isinstance(c, dict) and c.get("id")
        }
        j = i + 1
        seen_ids: set = set()
        tool_msgs: List[Any] = []

        while j < len(messages) and _role(messages[j]) == "tool":
            tid = _tool_call_id(messages[j])
            if tid:
                seen_ids.add(tid)
            tool_msgs.append(messages[j])
            j += 1

        if required_ids and required_ids.issubset(seen_ids):
            sanitized.append(msg)
            sanitized.extend(tool_msgs)
        else:
            log.warning(
                "Dropped dangling tool_call assistant turn (missing ids: %s)",
                required_ids - seen_ids,
            )
        i = j

    return sanitized


def format_conversation(messages: List[Any]) -> str:
    lines = ["Conversation history:\n"]
    for msg in messages:
        role = _role(msg)
        txt = content_to_text(
            msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", None)
        )
        if role == "user":
            lines.append(f"User: {txt}")
        elif role == "assistant" and txt and not txt.startswith("Evaluator Feedback:"):
            lines.append(f"Assistant: {txt or '[Tool use]'}")
    return "\n".join(lines)


# ─────────────────────────────── Graph Nodes ─────────────────────────────────


async def worker(state: State) -> Dict[str, Any]:
    log.info("worker() — state messages: %d", len(state.get("messages", [])))

    if worker_llm_with_tools is None:
        raise RuntimeError("worker_llm_with_tools is None — server initialisation incomplete.")

    # Build the system prompt in clear, structured sections
    sections = [
        (
            "You are a precise, helpful assistant that uses tools to complete tasks.\n"
            "Keep working until the success criteria is met or you need user clarification.\n"
            "IMPORTANT: If the user asks about an uploaded document, file, or PDF, its contents are ALREADY "
            "provided to you in the 'Retrieved File Context' section below. Read the context and answer directly, "
            "do NOT claim you cannot access or read the file."
        ),
        f"## Success Criteria\n{state['success_criteria']}",
        (
            "## Response format\n"
            "- If you need clarification: start with 'Question: <your question>'\n"
            "- Otherwise: reply with your final answer directly — no preamble."
        ),
    ]

    if state.get("rag_context"):
        sections.append(
            f"## Retrieved File Context\nUse the following when relevant to the request.\n\n{state['rag_context']}"
        )

    if state.get("feedback_on_work"):
        sections.append(
            f"## Previous Attempt Feedback\nYour last response did not satisfy the criteria.\n"
            f"Feedback: {state['feedback_on_work']}\n"
            "Address this feedback explicitly."
        )

    system_message = "\n\n".join(sections)
    sanitized = sanitize_tool_call_history(state["messages"])

    # Build the final message list for the LLM: Always one fresh system prompt at the top.
    final_messages = [{"role": "system", "content": system_message}]
    for msg in sanitized:
        is_sys = (
            getattr(msg, "type", "") == "system"
            or (isinstance(msg, dict) and msg.get("type") == "system")
        )
        if not is_sys:
            final_messages.append(msg)

    log.debug("worker() — invoking LLM...")
    response = await worker_llm_with_tools.ainvoke(final_messages)
    log.debug("worker() — LLM done.")
    return {"messages": [response]}


async def worker_router(state: State) -> str:
    last = state["messages"][-1]
    if _tool_calls(last):
        log.debug("worker_router → tools")
        return "tools"
    log.debug("worker_router → evaluator")
    return "evaluator"


async def evaluator(state: State) -> Dict[str, Any]:
    log.info("evaluator() — running structured eval...")

    last_msg = state["messages"][-1]
    last_response = content_to_text(
        last_msg.get("content", "") if isinstance(last_msg, dict) else getattr(last_msg, "content", None)
    )

    sys_prompt = (
        "You are a strict but fair evaluator for an AI assistant.\n"
        "Assess whether the Assistant's latest response satisfies the success criteria.\n"
        "Be concise and specific in your feedback."
    )

    user_parts = [
        format_conversation(state["messages"]),
        f"\n## Success Criteria\n{state['success_criteria']}",
        f"\n## Assistant's Latest Response\n{last_response}",
        "\nEvaluate: is the criteria met? Is user input needed?",
    ]
    if state.get("feedback_on_work"):
        user_parts.append(
            f"\nNote — prior feedback was:\n{state['feedback_on_work']}\n"
            "If the assistant is repeating the same mistake, set user_input_needed=True."
        )

    eval_messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "\n".join(user_parts)},
    ]

    result: EvaluatorOutput = await evaluator_llm_with_output.ainvoke(eval_messages)
    retry = int(state.get("retry_count", 0)) + 1

    log.info(
        "evaluator() — met=%s, user_input=%s, retry=%d/%d",
        result.success_criteria_met, result.user_input_needed, retry, MAX_RETRIES,
    )

    return {
        "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {result.feedback}"}],
        "feedback_on_work": result.feedback,
        "success_criteria_met": result.success_criteria_met,
        "user_input_needed": result.user_input_needed,
        "retry_count": retry,
    }


async def route_based_on_evaluation(state: State) -> str:
    retry = state.get("retry_count", 0)
    if retry >= MAX_RETRIES:
        log.warning("Max retries (%d) reached — forcing END.", MAX_RETRIES)
        return "END"
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    return "worker"


# ─────────────────────────────── Graph Build ─────────────────────────────────

graph: Optional[Any] = None
memory: Optional[Any] = None  # AsyncSqliteSaver, assigned in lifespan


def build_graph() -> None:
    global graph
    if memory is None:
        raise RuntimeError("checkpointer not initialised — call build_graph() only inside lifespan.")
    builder = StateGraph(State)
    builder.add_node("worker", worker)
    builder.add_node("tools", ToolNode(tools=tools))
    builder.add_node("evaluator", evaluator)

    builder.add_conditional_edges(
        "worker", worker_router, {"tools": "tools", "evaluator": "evaluator"}
    )
    builder.add_edge("tools", "worker")
    builder.add_conditional_edges(
        "evaluator", route_based_on_evaluation, {"worker": "worker", "END": END}
    )
    builder.add_edge(START, "worker")

    graph = builder.compile(checkpointer=memory)
    log.info("Graph compiled. Nodes: worker → tools ↺ | worker → evaluator → [worker|END]")


# ─────────────────────────────── FastAPI Lifespan ────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _playwright_context, async_browser, toolkit, tools, worker_llm_with_tools, memory, graph

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    init_rag_db()
    load_rag_from_disk()

    log.info("Launching Playwright browser...")
    try:
        _playwright_context = await async_playwright().start()
        async_browser = await asyncio.wait_for(
            _playwright_context.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
            ),
            timeout=PLAYWRIGHT_TIMEOUT,
        )
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
        pw_tools = toolkit.get_tools()
        tools = pw_tools + [tavily_search]
        log.info("Playwright ready. Tools: %s", [t.name for t in tools])
    except Exception:
        log.exception(
            "Playwright failed — falling back to tavily_search only. "
            "Browser-based tools will be unavailable this session."
        )
        tools = [tavily_search]

    worker_llm_with_tools = worker_llm.bind_tools(tools)

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        memory = checkpointer
        build_graph()
        log.info(
            "✅ Sidekick ready at http://localhost:8000 (RAG DB: %s, checkpoints: %s)",
            RAG_DB_PATH,
            CHECKPOINT_DB_PATH,
        )
        yield

        graph = None
        memory = None

    log.info("Shutting down Playwright...")
    if async_browser:
        await async_browser.close()
    if _playwright_context:
        await _playwright_context.stop()


# ─────────────────────────────── FastAPI App ─────────────────────────────────

app = FastAPI(title="Sidekick AI", lifespan=lifespan)
app.mount("/public", StaticFiles(directory="public"), name="public")


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    with open("public/index.html", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ─────────────────────────────── Upload ──────────────────────────────────────


@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    thread_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    resolved_tid = thread_id or str(uuid.uuid4())
    if resolved_tid not in thread_documents:
        thread_documents[resolved_tid] = []

    added_files: List[str] = []
    added_chunks = 0
    ignored_files: List[Dict[str, str]] = []

    for upload in files:
        filename = upload.filename or "uploaded_file"
        suffix = Path(filename).suffix.lower()

        raw = await upload.read()
        if not raw or len(raw) > MAX_FILE_BYTES:
            ignored_files.append(
                {"filename": filename, "reason": "empty file or exceeds 5 MB limit"}
            )
            continue

        # 1) Explicit ignores (images/archives/binaries).
        if suffix in IGNORED_EXTENSIONS:
            ignored_files.append(
                {"filename": filename, "reason": f"ignored file type: {suffix}"}
            )
            continue

        # 1b) Enforce allowlist so unsupported binary/unknown types are rejected.
        if suffix not in ALLOWED_EXTENSIONS:
            ignored_files.append(
                {"filename": filename, "reason": f"unsupported file type: {suffix or '[no extension]'}"}
            )
            continue

        # 2) Extract text by file type.
        text = ""
        try:
            if suffix == ".pdf":
                # PDF text extraction (best-effort).
                from io import BytesIO
                from pypdf import PdfReader

                reader = PdfReader(BytesIO(raw))
                parts: List[str] = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        parts.append(page_text)
                text = "\n".join(parts)

            elif suffix == ".docx":
                from io import BytesIO

                try:
                    import importlib
                    docx_mod = importlib.import_module("docx")  # python-docx
                except Exception:
                    ignored_files.append(
                        {
                            "filename": filename,
                            "reason": "DOCX parsing requires python-docx (not installed)",
                        }
                    )
                    continue

                document = docx_mod.Document(BytesIO(raw))
                parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
                text = "\n".join(parts)

            else:
                # Default: treat as text-like and decode.
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1", errors="ignore")

        except Exception as e:
            ignored_files.append(
                {"filename": filename, "reason": f"text extraction failed: {e}"}
            )
            continue

        if not looks_like_meaningful_text(text):
            ignored_files.append(
                {"filename": filename, "reason": "no meaningful text extracted"}
            )
            continue

        chunks = chunk_text(text)
        if not chunks:
            ignored_files.append(
                {"filename": filename, "reason": "no chunks produced after processing"}
            )
            continue

        for chunk in chunks:
            thread_documents[resolved_tid].append({"source": filename, "text": chunk})
        persist_rag_chunks(resolved_tid, filename, chunks)

        added_files.append(filename)
        added_chunks += len(chunks)
        log.info("Indexed '%s' → %d chunks (thread=%s)", filename, len(chunks), resolved_tid)

    return {
        "thread_id": resolved_tid,
        "files": added_files,
        "chunks_added": added_chunks,
        "total_chunks_for_thread": len(thread_documents.get(resolved_tid, [])),
        "ignored_files": ignored_files,
    }


# ─────────────────────────────── Chat Stream ─────────────────────────────────


@app.post("/api/chat")
async def chat_stream(request: ChatRequest) -> EventSourceResponse:
    thread_id = request.thread_id or str(uuid.uuid4())
    resolved_success_criteria = (request.success_criteria or "").strip() or DEFAULT_SUCCESS_CRITERIA

    async def event_generator() -> AsyncGenerator[Dict[str, str], None]:

        # ── 1. Immediate meta ─────────────────────────────────────────────────
        yield {"event": "meta", "data": json.dumps({"thread_id": thread_id})}

        # ── 2. Server-ready guard ─────────────────────────────────────────────
        if graph is None or worker_llm_with_tools is None:
            yield {
                "event": "error",
                "data": json.dumps({"message": "Server not ready — please retry in a moment."}),
            }
            yield {
                "event": "done",
                "data": json.dumps({"assistant": "", "evaluator": "", "thread_id": thread_id}),
            }
            return

        # ── 3. RAG retrieval ──────────────────────────────────────────────────
        rag_context = retrieve_thread_context(thread_id, request.message)
        config = {"configurable": {"thread_id": thread_id}}

        # ── 4. New vs. continuing thread ──────────────────────────────────────
        is_new_thread = True
        try:
            current = await asyncio.wait_for(graph.aget_state(config), timeout=STATE_TIMEOUT)
            is_new_thread = not current or not current.values
        except Exception:
            log.exception("aget_state() failed — treating as new thread.")

        base_input: Dict[str, Any] = {
            "messages": [{"role": "user", "content": request.message}],
            "success_criteria": resolved_success_criteria,
            "rag_context": rag_context,
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
            "retry_count": 0,
        }
        if not is_new_thread:
            # On continuing threads we still reset the per-turn mutable fields
            # (retry_count, feedback, flags) so the new turn starts fresh.
            invoke_input = base_input
        else:
            invoke_input = base_input

        # ── 5. Stream the graph ───────────────────────────────────────────────
        latest_assistant_text = ""

        try:
            async for chunk in graph.astream(
                invoke_input, config=config, stream_mode="updates"
            ):
                for node_name, node_output in chunk.items():

                    if node_name == "worker":
                        last_msg = node_output.get("messages", [{}])[-1]
                        calls = _tool_calls(last_msg)

                        if calls:
                            for tc in calls:
                                tn = tc.get("name", "")
                                if any(k in tn.lower() for k in ("search", "tavily")):
                                    status = "🔍 Searching the web..."
                                elif any(k in tn.lower() for k in ("navigate", "click", "page", "browser", "playwright")):
                                    status = "🌐 Browsing the web..."
                                else:
                                    status = f"⚙️ Running tool: {tn}..."
                                yield {"event": "status", "data": json.dumps({"text": status})}
                        else:
                            yield {"event": "status", "data": json.dumps({"text": "🧠 Thinking..."})}
                            txt = content_to_text(
                                last_msg.get("content", "")
                                if isinstance(last_msg, dict)
                                else getattr(last_msg, "content", None)
                            )
                            if txt and not txt.startswith("Evaluator Feedback:"):
                                formatted = format_assistant_markdown(txt)
                                latest_assistant_text = formatted
                                yield {"event": "assistant", "data": json.dumps({"text": formatted})}

                    elif node_name == "tools":
                        yield {"event": "status", "data": json.dumps({"text": "📡 Processing tool results..."})}

                    elif node_name == "evaluator":
                        yield {"event": "status", "data": json.dumps({"text": "📋 Evaluating response..."})}
                        sc_met = node_output.get("success_criteria_met", False)
                        ui_needed = node_output.get("user_input_needed", False)
                        retry = node_output.get("retry_count", 0)

                        if sc_met:
                            yield {"event": "status", "data": json.dumps({"text": "✅ Criteria met!"})}
                        elif ui_needed:
                            yield {"event": "status", "data": json.dumps({"text": "❓ Clarification needed from you"})}
                        else:
                            yield {
                                "event": "status",
                                "data": json.dumps({"text": f"🔁 Not quite — retrying ({retry}/{MAX_RETRIES})..."}),
                            }

        except Exception:
            log.exception("Error inside graph.astream()")
            yield {
                "event": "error",
                "data": json.dumps({"message": "An internal error occurred. Check server logs."}),
            }
            # Always emit done so the client unblocks
            yield {
                "event": "done",
                "data": json.dumps({
                    "assistant": format_assistant_markdown(latest_assistant_text)
                    if latest_assistant_text
                    else "",
                    "evaluator": "",
                    "thread_id": thread_id,
                }),
            }
            return

        # ── 6. Resolve final answer from checkpoint ───────────────────────────
        assistant_text = ""
        evaluator_text = ""
        try:
            final_state = await asyncio.wait_for(
                graph.aget_state(config), timeout=STATE_TIMEOUT
            )
            if final_state and final_state.values:
                for msg in reversed(final_state.values.get("messages", [])):
                    raw = (
                        msg.get("content", "") if isinstance(msg, dict)
                        else getattr(msg, "content", None)
                    )
                    txt = content_to_text(raw)
                    if not txt:
                        continue
                    if txt.startswith("Evaluator Feedback:") and not evaluator_text:
                        evaluator_text = txt
                    elif not txt.startswith("Evaluator Feedback:") and not assistant_text:
                        assistant_text = txt
                    if assistant_text and evaluator_text:
                        break
        except Exception:
            log.exception("aget_state() failed after stream — using streamed fallback.")

        if not assistant_text:
            assistant_text = latest_assistant_text

        assistant_text = format_assistant_markdown(assistant_text)

        yield {
            "event": "done",
            "data": json.dumps({
                "assistant": assistant_text,
                "evaluator": evaluator_text,
                "thread_id": thread_id,
            }),
        }

    return EventSourceResponse(event_generator())


# ─────────────────────────────── Reset ───────────────────────────────────────


@app.post("/api/reset")
async def reset_session(thread_id: Optional[str] = None) -> Dict[str, str]:
    """
    Issue a new thread_id. Optionally pass the old one to purge its
    RAG index, LangGraph checkpoints, and in-memory cache.
    """
    if thread_id:
        delete_rag_for_thread(thread_id)
        purge_checkpoint_thread(thread_id)
        log.info("Purged RAG + checkpoints for thread %s", thread_id)
    new_thread = str(uuid.uuid4())
    return {"thread_id": new_thread}