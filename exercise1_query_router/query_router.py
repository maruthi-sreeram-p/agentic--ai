"""
Exercise 1 - Campus Query Router Agent

Takes a student's question, asks the LLM to classify it as Academic / Exam / General,
then calls the matching stub function and prints what it returns.
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads GROQ_API_KEY from a .env file (this folder or any parent folder)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

ROUTER_PROMPT = """You are the query router of a college helpdesk.
Classify the student's question into exactly one intent:
- Academic: courses, syllabus, subjects, faculty, timetable, registration, attendance
- Exam: exam dates, hall tickets, results, revaluation, exam rules
- General: hostel, fees, library, transport, canteen, anything else

Reply with JSON only, like this:
{"thought": "<one short sentence of reasoning>", "intent": "Academic" or "Exam" or "General"}"""


# ---- stub tools (one per intent) ----
def academic_tool(query):
    return f"[Academic Office] Your question '{query}' has been forwarded to the syllabus and course registration desk."


def exam_tool(query):
    return f"[Exam Cell] Your question '{query}' has been forwarded to the timetable and hall ticket desk."


def general_tool(query):
    return f"[General Helpdesk] Your question '{query}' has been forwarded to the campus information desk."


TOOLS = {
    "Academic": academic_tool,
    "Exam": exam_tool,
    "General": general_tool,
}


def classify(query):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": ROUTER_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        data = {}
    thought = data.get("thought", "")
    intent = data.get("intent", "General")
    if intent not in TOOLS:
        intent = "General"
    return thought, intent


def run_agent(query):
    # ReAct style: Thought -> Action -> Observation
    thought, intent = classify(query)
    print(f"Thought: {thought}")
    print(f"Action: {intent.lower()}_tool")
    observation = TOOLS[intent](query)
    print(f"Observation: {observation}")


def main():
    print("Campus Query Router Agent (type 'exit' to quit)\n")
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
