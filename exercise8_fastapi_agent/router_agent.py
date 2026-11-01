"""
Agent logic for Exercise 8 - this is the Exercise 1 query router
without the console loop, so the API can call it.
"""
import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

ROUTER_PROMPT = """You are the query router of a college helpdesk.
Classify the student's question into exactly one intent:
- Academic: courses, syllabus, subjects, faculty, timetable, registration, attendance
- Exam: exam dates, hall tickets, results, revaluation, exam rules
- General: hostel, fees, library, transport, canteen, anything else

Reply with JSON only, like this:
{"intent": "Academic" or "Exam" or "General"}"""


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
        intent = json.loads(response.choices[0].message.content).get("intent", "General")
    except json.JSONDecodeError:
        intent = "General"
    return intent if intent in TOOLS else "General"


def route(query):
    """Returns (intent, answer)."""
    intent = classify(query)
    return intent, TOOLS[intent](query)
