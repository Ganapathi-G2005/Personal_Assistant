import asyncio
from main import graph, invoke_input

async def test_graph():
    print("Testing graph execution directly...")
    
    config = {"configurable": {"thread_id": "test_thread_123"}}
    inp = {
        "messages": [{"role": "user", "content": "search for information on async playwright"}],
        "success_criteria": "just give a summary",
        "feedback_on_work": None,
        "success_criteria_met": False,
        "user_input_needed": False,
    }
    
    try:
        async for chunk in graph.astream(inp, config=config, stream_mode="updates"):
            print("CHUNK REC:", chunk)
    except Exception as e:
        print("ERROR:", e)
        import traceback
        traceback.print_exc()
        
asyncio.run(test_graph())
