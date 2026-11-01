"""
Exercise 4 - Self-Correction Coding Agent (LangGraph)

generate -> check -> (error found?) -> revise -> check -> ... -> done

check runs the generated code (only after the user says yes) and looks for
"Traceback" or "Error" in what it prints. If found, the agent revises the code
using that error message. It stops after MAX_REVISIONS revisions.
"""
import re
import subprocess
import sys
import tempfile
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

load_dotenv()
llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

MAX_REVISIONS = 2

GENERATE_PROMPT = (
    "You are a Python coding assistant. Write a short, complete Python script for the task below. "
    "The script must run on its own with no user input and print its result. "
    "If it needs a file, create a small sample file first. "
    "Reply with only one ```python code block."
)

REVISE_PROMPT = (
    "The Python script below was run and printed an error. Fix it. "
    "Reply with only one corrected ```python code block."
)


class State(TypedDict):
    question: str
    code: str
    output: str
    has_error: bool
    revisions: int


def extract_code(text):
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


def generate(state):
    print("\n[generate] writing code...")
    reply = llm.invoke([("system", GENERATE_PROMPT), ("human", state["question"])])
    return {"code": extract_code(reply.content), "revisions": 0}


def check(state):
    print("\n--- code ---")
    print(state["code"])
    print("------------")
    answer = input("Run this code to test it? (y/n): ").strip().lower()
    if answer != "y":
        print("[check] skipped, not running the code")
        return {"output": "(not run)", "has_error": False}

    # run in a temp folder so any sample files it creates don't end up in the project
    with tempfile.TemporaryDirectory() as tmp:
        script = f"{tmp}/script.py"
        with open(script, "w", encoding="utf-8") as f:
            f.write(state["code"])
        try:
            run = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=10, cwd=tmp,
            )
            output = (run.stdout + run.stderr).strip()
        except subprocess.TimeoutExpired:
            output = "Error: the script took more than 10 seconds"

    has_error = "Traceback" in output or "Error" in output
    print(f"[check] output:\n{output}")
    print("[check] error found" if has_error else "[check] no error found")
    return {"output": output, "has_error": has_error}


def revise(state):
    print("\n[revise] fixing the code...")
    message = f"Task: {state['question']}\n\nScript:\n{state['code']}\n\nError output:\n{state['output']}"
    reply = llm.invoke([("system", REVISE_PROMPT), ("human", message)])
    return {"code": extract_code(reply.content), "revisions": state["revisions"] + 1}


def after_check(state):
    if state["has_error"] and state["revisions"] < MAX_REVISIONS:
        return "revise"
    return END


graph = StateGraph(State)
graph.add_node("generate", generate)
graph.add_node("check", check)
graph.add_node("revise", revise)
graph.add_edge(START, "generate")
graph.add_edge("generate", "check")
graph.add_conditional_edges("check", after_check, {"revise": "revise", END: END})
graph.add_edge("revise", "check")
agent = graph.compile()


def main():
    print("Self-Correction Coding Agent (type 'exit' to quit)\n")
    while True:
        question = input("Coding question: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        try:
            result = agent.invoke({"question": question, "code": "", "output": "", "has_error": False, "revisions": 0})
        except Exception as e:
            print(f"Something went wrong: {e}")
            continue

        print("\n=== Final answer ===")
        print(result["code"])
        if result["has_error"]:
            print(f"\nStill showing an error after {result['revisions']} revision(s):\n{result['output']}")
        elif result["revisions"]:
            print(f"\n(Corrected after {result['revisions']} revision(s))")
        print()


if __name__ == "__main__":
    main()
