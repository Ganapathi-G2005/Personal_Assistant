import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import json
import os
import sys
import uuid
import traceback
from contextlib import asynccontextmanager
from typing import Annotated, Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.tools import tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from tavily import TavilyClient
from typing_extensions import NotRequired, TypedDict

load_dotenv(override=True)

# ─────────────────────────────── Pydantic Schemas ────────────────────────────

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(
        description="True if more input is needed from the user, or clarifications, or the assistant is stuck"
    )


class ChatRequest(BaseModel):
    message: str
    success_criteria: str
    thread_id: Optional[str] = None


# ─────────────────────────────── LangGraph State ─────────────────────────────

class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: NotRequired[Optional[str]]
    success_criteria_met: bool
    user_input_needed: bool


# ─────────────────────────────── Tools Setup ─────────────────────────────────

# Playwright browser is initialized at startup (see lifespan)
_playwright_context = None
async_browser = None
toolkit = None
tools = []

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def tavily_search(query: str) -> str:
    """Search the web for information using Tavily."""
    response = tavily_client.search(query=query, search_depth="basic")
    return str(response["results"])

# ─────────────────────────────── LLMs ────────────────────────────────────────

worker_llm = ChatOpenAI(model="gpt-4o-mini")
# worker_llm_with_tools is re-bound after tools are initialized at startup
worker_llm_with_tools = None

evaluator_llm = ChatOpenAI(model="gpt-4.1-nano")
evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

# ─────────────────────────────── Graph Nodes ─────────────────────────────────

async def worker(state: State) -> Dict[str, Any]:
    print(f"--> [DEBUG] Entered worker() node. Messages count: {len(state.get('messages', []))}")
    system_message = f"""You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
This is the success criteria:
{state['success_criteria']}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""
    if state.get("feedback_on_work"):
        system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state['feedback_on_work']}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user."""

    found_system_message = False
    messages = state["messages"]
    for message in messages:
        # LangGraph message format adaptation handling both objects and dicts
        is_sys = getattr(message, "type", "") == "system" or (isinstance(message, dict) and message.get("type") == "system")
        if is_sys:
            if hasattr(message, "content"):
                message.content = system_message
            else:
                message["content"] = system_message
            found_system_message = True

    if not found_system_message:
        messages = [{"role": "system", "content": system_message}] + messages

    print("--> [DEBUG] Calling worker_llm_with_tools.ainvoke()...")
    response = await worker_llm_with_tools.ainvoke(messages)
    print("--> [DEBUG] worker_llm_with_tools.ainvoke() completed.")
    return {"messages": [response]}


async def worker_router(state: State) -> str:
    print("--> [DEBUG] Entered worker_router()...")
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "evaluator"


def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        m_type = getattr(message, "type", "") or message.get("type", "") if isinstance(message, dict) else ""
        content = getattr(message, "content", "") or (message.get("content", "") if isinstance(message, dict) else "")

        if m_type == "human":
            conversation += f"User: {content}\n"
        elif m_type == "ai":
            text = content or "[Tool use]"
            conversation += f"Assistant: {text}\n"
    return conversation


async def evaluator(state: State) -> State:
    print("--> [DEBUG] Entered evaluator() node...")
    last_message = state["messages"][-1]
    last_response = getattr(last_message, "content", "") or (last_message.get("content", "") if isinstance(last_message, dict) else "")

    system_message = """You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""

    user_message = f"""You are evaluating a conversation between the User and Assistant.

The entire conversation is:
{format_conversation(state['messages'])}

The success criteria for this assignment is:
{state['success_criteria']}

The final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met.
Also, decide if more user input is required.
"""
    if state.get("feedback_on_work"):
        user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
        user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

    evaluator_messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    print("--> [DEBUG] Calling evaluator_llm_with_output.ainvoke()...")
    eval_result = await evaluator_llm_with_output.ainvoke(evaluator_messages)
    print(f"--> [DEBUG] evaluator_llm_with_output.ainvoke() completed: feedback={eval_result.feedback}, met={eval_result.success_criteria_met}")

    return {
        "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {eval_result.feedback}"}],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed,
    }


async def route_based_on_evaluation(state: State) -> str:
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    return "worker"


# ─────────────────────────────── Build Graph (deferred compile) ──────────────

graph = None
memory = MemorySaver()


def build_graph():
    global graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("worker", worker)
    graph_builder.add_node("tools", ToolNode(tools=tools))
    graph_builder.add_node("evaluator", evaluator)
    graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
    graph_builder.add_edge("tools", "worker")
    graph_builder.add_conditional_edges("evaluator", route_based_on_evaluation, {"worker": "worker", "END": END})
    graph_builder.add_edge(START, "worker")
    graph = graph_builder.compile(checkpointer=memory)

# ─────────────────────────────── FastAPI Lifespan ────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Playwright browser and build graph on startup."""
    global _playwright_context, async_browser, toolkit, tools, worker_llm_with_tools
    _playwright_context = await async_playwright().start()
    async_browser = await asyncio.wait_for(
        _playwright_context.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        ),
        timeout=30
    )
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools() + [tavily_search]
    worker_llm_with_tools = worker_llm.bind_tools(tools)
    build_graph()
    print("✅ Sidekick ready at http://localhost:8000")
    yield
    # Cleanup
    if async_browser:
        await async_browser.close()
    if _playwright_context:
        await _playwright_context.stop()


# ─────────────────────────────── FastAPI App ─────────────────────────────────

app = FastAPI(title="Sidekick AI", lifespan=lifespan)
app.mount("/public", StaticFiles(directory="public"), name="public")


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    with open("public/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/api/chat")
async def chat_stream(request: ChatRequest):
    thread_id = request.thread_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[dict, None]:
        def _content_to_text(raw: Any) -> str:
            """Normalize LangChain message content into plain text."""
            if raw is None:
                return ""
            if isinstance(raw, str):
                return raw
            if isinstance(raw, list):
                parts: List[str] = []
                for item in raw:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        if item.get("type") == "text":
                            parts.append(str(item.get("text", "")))
                        elif "text" in item:
                            parts.append(str(item.get("text", "")))
                return "\n".join(p for p in parts if p).strip()
            return str(raw)

        config = {"configurable": {"thread_id": thread_id}}
        latest_assistant_text = ""

        # Send the thread_id back immediately so the frontend can track it
        yield {"event": "meta", "data": json.dumps({"thread_id": thread_id})}

        if graph is None or worker_llm_with_tools is None:
            yield {"event": "error", "data": json.dumps({"message": "Server not ready or Playwright failed to initialize."})}
            return

        try:
            current_state = await asyncio.wait_for(graph.aget_state(config), timeout=10)
            if not current_state or not current_state.values:
                invoke_input = {
                    "messages": [{"role": "user", "content": request.message}],
                    "success_criteria": request.success_criteria,
                    "feedback_on_work": None,
                    "success_criteria_met": False,
                    "user_input_needed": False,
                }
            else:
                invoke_input = {
                    "messages": [{"role": "user", "content": request.message}],
                    "success_criteria": request.success_criteria,
                }
        except Exception as e:
            traceback.print_exc()
            yield {"event": "error", "data": json.dumps({"message": "Failed to load state: " + str(e)})}
            return

        try:
            async for chunk in graph.astream(invoke_input, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    if node_name == "worker":
                        last_msg = node_output.get("messages", [{}])[-1]
                        # Check if worker is about to call tools
                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                            tool_names = [tc["name"] for tc in last_msg.tool_calls]
                            for tn in tool_names:
                                if "search" in tn.lower() or "tavily" in tn.lower():
                                    yield {"event": "status", "data": json.dumps({"text": "🔍 Searching the web..."})}
                                elif "navigate" in tn.lower() or "browser" in tn.lower() or "playwright" in tn.lower() or "click" in tn.lower() or "page" in tn.lower():
                                    yield {"event": "status", "data": json.dumps({"text": "🌐 Browsing the web..."})}
                                else:
                                    yield {"event": "status", "data": json.dumps({"text": f"⚙️ Running tool: {tn}..."})}
                        else:
                            yield {"event": "status", "data": json.dumps({"text": "🧠 Thinking..."})}
                            text = _content_to_text(getattr(last_msg, "content", None))
                            if text:
                                latest_assistant_text = text
                                yield {"event": "assistant", "data": json.dumps({"text": text})}

                    elif node_name == "tools":
                        yield {"event": "status", "data": json.dumps({"text": "📡 Processing tool results..."})}

                    elif node_name == "evaluator":
                        yield {"event": "status", "data": json.dumps({"text": "📋 Evaluating response..."})}
                        # Extract evaluation result
                        msgs = node_output.get("messages", [])
                        sc_met = node_output.get("success_criteria_met", False)
                        ui_needed = node_output.get("user_input_needed", False)
                        feedback = node_output.get("feedback_on_work", "")

                        if sc_met:
                            yield {"event": "status", "data": json.dumps({"text": "✅ Criteria met!"})}
                        elif ui_needed:
                            yield {"event": "status", "data": json.dumps({"text": "❓ Clarification needed from you"})}
                        else:
                            yield {"event": "status", "data": json.dumps({"text": "🔁 Not quite right — retrying..."})}

            # Get the final state
            final_state = await asyncio.wait_for(graph.aget_state(config), timeout=10)
            if final_state and final_state.values:
                all_messages = final_state.values.get("messages", [])
                # Assistant response = second to last, evaluator = last
                assistant_text = ""
                evaluator_text = ""
                for msg in reversed(all_messages):
                    content: Any = ""
                    if hasattr(msg, "content") and msg.content:
                        content = msg.content
                    elif isinstance(msg, dict):
                        content = msg.get("content", "")
                    content = _content_to_text(content)

                    if not content:
                        continue

                    if content.startswith("Evaluator Feedback:") and not evaluator_text:
                        evaluator_text = content
                    elif not content.startswith("Evaluator Feedback:") and not assistant_text:
                        assistant_text = content

                    if assistant_text and evaluator_text:
                        break

                if not assistant_text and latest_assistant_text:
                    assistant_text = latest_assistant_text

                yield {
                    "event": "done",
                    "data": json.dumps({
                        "assistant": assistant_text,
                        "evaluator": evaluator_text,
                        "thread_id": thread_id,
                    }),
                }
        except Exception as e:
            traceback.print_exc()
            yield {"event": "error", "data": json.dumps({"message": str(e)})}

    return EventSourceResponse(event_generator())


@app.post("/api/reset")
async def reset_session():
    new_thread = str(uuid.uuid4())
    return {"thread_id": new_thread}