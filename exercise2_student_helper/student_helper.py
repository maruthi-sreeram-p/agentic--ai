"""
Exercise 2 - Student Helper Agent

The LLM reads the student's query and decides which tool to use
(math, python_help or english_grammar). We call that tool and print its response.
"""
import ast
import json
import operator
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

ROUTER_PROMPT = """You are a student helper. For each request, choose one action:
- math: arithmetic. action_input must be a plain arithmetic expression, e.g. "25% of 480" becomes "0.25 * 480"
- python_help: questions about Python syntax or errors. action_input is the question.
- english_grammar: correct an English sentence. action_input is the sentence.
- none: anything else.

Reply with JSON only:
{"thought": "<one short sentence>", "action": "math" or "python_help" or "english_grammar" or "none", "action_input": "<input>"}"""


def ask_llm(system, user, json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        **kwargs,
    )
    return response.choices[0].message.content.strip()


# ---- tools ----
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    raise ValueError("unsupported expression")


def math_tool(expression):
    # only plain arithmetic is allowed, no eval()
    result = _eval(ast.parse(expression, mode="eval").body)
    return f"{expression} = {result}"


def python_help_tool(question):
    return ask_llm(
        "You are a Python tutor. Answer in at most 4 short lines and include a tiny code example if useful.",
        question,
    )


def english_grammar_tool(sentence):
    corrected = ask_llm(
        "Correct the grammar of the user's text. Reply with only the corrected text.",
        sentence,
    )
    return f"Corrected: {corrected}"


TOOLS = {
    "math": math_tool,
    "python_help": python_help_tool,
    "english_grammar": english_grammar_tool,
}


def decide(query):
    reply = ask_llm(ROUTER_PROMPT, query, json_mode=True)
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return "", "none", query
    return data.get("thought", ""), data.get("action", "none"), data.get("action_input", query)


def run_agent(query):
    thought, tool, tool_input = decide(query)
    print(f"Thought: {thought}")

    if tool not in TOOLS:
        print("Action: none")
        print("Observation: I can only help with math, Python and English grammar.")
        return

    print(f"Action: {tool}({tool_input!r})")
    try:
        observation = TOOLS[tool](tool_input)
    except Exception as e:
        observation = f"The {tool} tool failed: {e}"
    print(f"Observation: {observation}")


def main():
    print("Student Helper Agent - math / Python help / English grammar (type 'exit' to quit)\n")
    while True:
        query = input("Student: ").strip()
        if query.lower() == "exit":
            break
        if not query:
            continue
        try:
            run_agent(query)
        except Exception as e:
            print(f"Something went wrong: {e}")
        print()


if __name__ == "__main__":
    main()
