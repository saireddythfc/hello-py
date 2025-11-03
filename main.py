import asyncio
import json
from io import StringIO
from contextlib import redirect_stdout
from typing import Any, Callable, TypedDict
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam

from tools import train_and_eval

# -------------------- Tool Handlers --------------------

class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None

class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool

def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """Execute Python expressions safely and capture stdout."""
    try:
        ns = {}
        buf = StringIO()
        with redirect_stdout(buf):
            exec(expression, ns, ns)
        return {"result": buf.getvalue(), "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}

def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Submit the final tuned accuracy result."""
    return {"answer": answer, "submitted": True}

# -------------------- RL Task Prompt --------------------

BASELINE = train_and_eval({"lr": 0.01, "weight_decay": 0.01})

PROMPT = f"""
You are an ML engineer tuning hyperparameters of a logistic regression model.

A baseline model achieves validation accuracy = {BASELINE:.3f}.

You can use the `python_expression` tool to run Python code.
The function `train_and_eval(params)` is available; call it with a dictionary
containing 'lr' (learning rate) and 'weight_decay' values, e.g.:

    acc = train_and_eval({{"lr": 0.05, "weight_decay": 0.001}})

Your goal is to find parameters giving accuracy at least 0.02 higher than baseline.
When done, call `submit_answer` with the accuracy you achieved.
Return only the numeric accuracy.
"""

# -------------------- Agent Loop --------------------

async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 10,
    model: str = "claude-3-5-haiku-latest",
) -> Any | None:
    client = AsyncAnthropic()
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for _ in range(max_steps):
        resp = await client.messages.create(model=model, max_tokens=1000, tools=tools, messages=messages)
        has_tool = False
        submitted = None
        tool_results = []

        for c in resp.content:
            if c.type == "tool_use":
                has_tool = True
                name = c.name
                handler = tool_handlers.get(name)
                inp = c.input
                if name == "python_expression":
                    result = handler(inp["expression"])
                elif name == "submit_answer":
                    result = handler(inp["answer"])
                    submitted = result["answer"]
                else:
                    result = handler(**inp)
                tool_results.append({"type": "tool_result", "tool_use_id": c.id, "content": json.dumps(result)})

        if has_tool:
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted is not None:
                return submitted
        else:
            break
    return None

# -------------------- Test Harness --------------------

async def run_single_test(run_id: int, num_runs: int, expected_min: float) -> tuple[int, bool, Any]:
    tools = [
        {
            "name": "python_expression",
            "description": "Evaluates a Python expression",
            "input_schema": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the final tuned accuracy",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "Accuracy value"}},
                "required": ["answer"],
            },
        },
    ]
    handlers = {"python_expression": python_expression_tool, "submit_answer": submit_answer_tool}

    print(f"--- Run {run_id}/{num_runs} ---")
    acc = await run_agent_loop(prompt=PROMPT, tools=tools, tool_handlers=handlers, max_steps=6)
    try:
        acc_val = float(acc)
    except (TypeError, ValueError):
        acc_val = None
    success = acc_val is not None and acc_val >= expected_min
    print(f"{'✓' if success else '✗'} Accuracy={acc}, threshold={expected_min:.3f}")
    return run_id, success, acc

async def main(concurrent=True):
    num_runs = 10
    expected_min = BASELINE + 0.02
    print(f"Baseline accuracy: {BASELINE:.3f}")
    print(f"Target accuracy:   {expected_min:.3f}")
    print("=" * 50)

    coros = [run_single_test(i + 1, num_runs, expected_min) for i in range(num_runs)]
    results = await asyncio.gather(*coros) if concurrent else [await c for c in coros]
    successes = sum(1 for _, s, _ in results if s)
    rate = (successes / num_runs) * 100
    print(f"\nPass Rate: {rate:.1f}% ({successes}/{num_runs})")

if __name__ == "__main__":
    asyncio.run(main(concurrent=True))
