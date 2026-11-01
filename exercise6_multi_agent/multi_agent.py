"""
Exercise 6 - Two-agent pipeline

Research Agent  : topic -> bullet points
Refiner Agent   : bullet points -> short explanation
The refined explanation is what the user gets at the end.
"""
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

RESEARCH_PROMPT = (
    "You are a Research Agent. For the given topic, list 5 to 6 short factual bullet points. "
    "Output only the bullets, one per line, each starting with '- '."
)

REFINER_PROMPT = (
    "You are a Refiner Agent. You get a topic and raw bullet points from a researcher. "
    "Turn them into one short, well-structured explanation of 4 to 5 sentences in plain prose. "
    "Do not use bullet points and do not add facts that are not in the bullets."
)


def ask(system, user):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def research_agent(topic):
    return ask(RESEARCH_PROMPT, topic)


def refiner_agent(topic, bullets):
    return ask(REFINER_PROMPT, f"Topic: {topic}\n\nBullet points:\n{bullets}")


def run_pipeline(topic):
    print("\n[Research Agent] working...")
    bullets = research_agent(topic)
    print(bullets)

    print("\n[Refiner Agent] working...")
    explanation = refiner_agent(topic, bullets)

    print("\nFinal answer:")
    print(explanation)


def main():
    print("Research Agent -> Refiner Agent (type 'exit' to quit)\n")
    while True:
        topic = input("Topic: ").strip()
        if topic.lower() == "exit":
            break
        if not topic:
            continue
        try:
            run_pipeline(topic)
        except Exception as e:
            print(f"Something went wrong: {e}")
        print()


if __name__ == "__main__":
    main()
