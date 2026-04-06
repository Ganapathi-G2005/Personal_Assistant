import asyncio
from main import evaluator_llm_with_output, EvaluatorOutput

async def test():
    messages = [
        {"role": "system", "content": "You are an evaluator..."},
        {"role": "user", "content": "Test conversation..."}
    ]
    print("Invoking evaluator...")
    try:
        # Tenacity might log inside here
        res = await evaluator_llm_with_output.ainvoke(messages)
        print("Result:", res)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
