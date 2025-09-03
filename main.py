from typing import Annotated, TypedDict, List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import gradio as gr
import uuid
from dotenv import load_dotenv
from tavily import TavilyClient
import os
from langchain.tools import tool

load_dotenv(override=True)

# ### For structured outputs, we define a Pydantic object for the Schema
# First define a structured output

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")


class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool

# Get our async Playwright tools

import nest_asyncio
nest_asyncio.apply()
async_browser =  create_async_playwright_browser(headless=False)  # headful mode
toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def tavily_search(query: str) -> str:
    """Search the web for information using Tavily."""
    print(f"--- Calling Tavily Search Tool for query: {query} ---")
    # FIX 2: Return a string, not the whole dictionary object
    response = tavily_client.search(query=query, search_depth="basic")
    return str(response['results'])

tools = toolkit.get_tools() + [tavily_search]

# Initialize the LLMs

worker_llm = ChatOpenAI(model="gpt-4o-mini")
worker_llm_with_tools = worker_llm.bind_tools(tools)

evaluator_llm = ChatOpenAI(model="gpt-4o-mini")
evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

# The worker node

def worker(state: State) -> Dict[str, Any]:
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
    
    # Add in the system message

    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True
    
    if not found_system_message:
        messages = [SystemMessage(content=system_message)] + messages
    
    # Invoke the LLM with tools
    response = worker_llm_with_tools.invoke(messages)
    
    # Return updated state
    return {
        "messages": [response],
    }

def worker_router(state: State) -> str:
    last_message = state["messages"][-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    else:
        return "evaluator"

def format_conversation(messages: List[Any]) -> str:
    conversation = "Conversation history:\n\n"
    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"User: {message.content}\n"
        elif isinstance(message, AIMessage):
            text = message.content or "[Tools use]"
            conversation += f"Assistant: {text}\n"
    return conversation

def evaluator(state: State) -> State:
    last_response = state["messages"][-1].content

    system_message = f"""You are an evaluator that determines if a task has been completed successfully by an Assistant.
Assess the Assistant's last response based on the given criteria. Respond with your feedback, and with your decision on whether the success criteria has been met,
and whether more input is needed from the user."""
    
    user_message = f"""You are evaluating a conversation between the User and Assistant. You decide what action to take based on the last response from the Assistant.

The entire conversation with the assistant, with the user's original request and all replies, is:
{format_conversation(state['messages'])}

The success criteria for this assignment is:
{state['success_criteria']}

And the final response from the Assistant that you are evaluating is:
{last_response}

Respond with your feedback, and decide if the success criteria is met by this response.
Also, decide if more user input is required, either because the assistant has a question, needs clarification, or seems to be stuck and unable to answer without help.
"""
    if state.get("feedback_on_work"):
        user_message += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state['feedback_on_work']}\n"
        user_message += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."
    
    evaluator_messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]

    eval_result = evaluator_llm_with_output.invoke(evaluator_messages)
    new_state = {
        "messages": [{"role": "assistant", "content": f"Evaluator Feedback on this answer: {eval_result.feedback}"}],
        "feedback_on_work": eval_result.feedback,
        "success_criteria_met": eval_result.success_criteria_met,
        "user_input_needed": eval_result.user_input_needed
    }
    return new_state

def route_based_on_evaluation(state: State) -> str:
    if state["success_criteria_met"] or state["user_input_needed"]:
        return "END"
    else:
        return "worker"

# Set up Graph Builder with State
graph_builder = StateGraph(State)

# Add nodes
graph_builder.add_node("worker", worker)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_node("evaluator", evaluator)

# Add edges
graph_builder.add_conditional_edges("worker", worker_router, {"tools": "tools", "evaluator": "evaluator"})
graph_builder.add_edge("tools", "worker")
graph_builder.add_conditional_edges("evaluator", route_based_on_evaluation, {"worker": "worker", "END": END})
graph_builder.add_edge(START, "worker")

# Compile the graph
memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

# display(Image(graph.get_graph().draw_mermaid_png()))

# ### Next comes the gradio Callback to kick off a super-step
def make_thread_id() -> str:
    return str(uuid.uuid4())


async def process_message(message, success_criteria, history, thread):

    config = {"configurable": {"thread_id": thread}}

    state = {
        "messages": [HumanMessage(content=message)],
        "success_criteria": success_criteria,
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False
    }

    # Since the initial state might not have all keys, we get the current state first
    # and then update it with the new message.
    current_state = await graph.aget_state(config)
    
    # We create a dictionary to pass to ainvoke. If it's the first message,
    # we initialize the state. Otherwise, we just pass the new message.
    if not current_state:
        invoke_input = state
    else:
        # For subsequent messages, we need to provide both the new message AND the success_criteria
        # because the worker function needs access to success_criteria
        invoke_input = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": success_criteria
        }


    result = await graph.ainvoke(invoke_input, config=config)
    
    user_message_for_history = (message, None)

    # The actual response from the assistant is the second to last message.
    # The last one is the evaluator's feedback.
    assistant_response = result["messages"][-2].content if len(result["messages"]) >= 2 else "No response."
    
    # We add both the assistant's response and the evaluator's feedback to the chat history
    # for full transparency.
    full_response = f"{assistant_response}\n\n---\n*Evaluator Feedback:*\n{result['messages'][-1].content}"
    
    assistant_message_for_history = (None, full_response)

    # The history format for gradio chatbot is a list of tuples [(user, bot), (user, bot)]
    # We update the history this way.
    history.append(user_message_for_history)
    history.append(assistant_message_for_history)

    return history


async def reset():
    return [], "", "", make_thread_id()


# ### And now launch our Sidekick UI

with gr.Blocks(theme=gr.themes.Default(primary_hue="emerald")) as demo:
    gr.Markdown("## Sidekick Personal Co-worker")
    thread = gr.State(make_thread_id)
    
    chatbot = gr.Chatbot(label="Sidekick", height=500)
    
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(show_label=False, placeholder="Your request to the Agent", scale=4)
        with gr.Row():
            success_criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?", scale=4)
    with gr.Row():
        reset_button = gr.Button("Reset", variant="stop")
        go_button = gr.Button("Go!", variant="primary")
    
    # Define a new function to handle the gradio interaction logic
    async def handle_submit(msg, criteria, chat_history, thread_id):
        # Add user message to history immediately for better UX
        chat_history.append((msg, None))
        yield chat_history, "", "" # Return updated history and clear textboxes
        
        # Start processing the message
        final_history = await process_message(msg, criteria, chat_history, thread_id)
        yield final_history, "", ""

    # Connect the UI components to the handler function
    go_button.click(
        handle_submit, 
        [message, success_criteria, chatbot, thread], 
        [chatbot, message, success_criteria]
    )
    message.submit(
        handle_submit, 
        [message, success_criteria, chatbot, thread], 
        [chatbot, message, success_criteria]
    )
    success_criteria.submit(
        handle_submit, 
        [message, success_criteria, chatbot, thread], 
        [chatbot, message, success_criteria]
    )

    # The reset function needs to clear the chatbot as well
    async def reset_all():
        return [], "", "", make_thread_id()

    reset_button.click(
        reset_all, 
        [], 
        [chatbot, message, success_criteria, thread]
    )


demo.launch()