"""
Exercise 7 - API-enabled agent (ReAct loop + Open-Meteo public API)

Case study: a student asks about the weather before a campus event
("Is it ok to hold the sports day in Hyderabad today?").

The agent follows the ReAct cycle:
    Thought (LLM reasons) -> Action (LLM picks an API function) -> Observation (we call the API)
and repeats until the LLM can answer. Open-Meteo needs no API key.
At the end we print the raw JSON from the weather API and the explanation.
"""
import json
import os

import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You answer weather questions for college students. Use the functions to get real data, "
    "never guess. When you have the data, reply with a short, friendly explanation that includes "
    "the key numbers (temperature, humidity, rain, wind) and a clear recommendation."
)


# ---- functions that call the public APIs and return parsed JSON ----
def get_coordinates(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    data = requests.get(url, params={"name": city, "count": 1}, timeout=10).json()
    if not data.get("results"):
        return {"error": f"city '{city}' not found"}
    place = data["results"][0]
    return {
        "name": place["name"],
        "country": place.get("country"),
        "latitude": place["latitude"],
        "longitude": place["longitude"],
    }


def get_weather(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


FUNCTIONS = {
    "get_coordinates": get_coordinates,
    "get_weather": get_weather,
}

# how the LLM sees the functions
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_coordinates",
            "description": "Look up the latitude and longitude of a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather at a latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                },
                "required": ["latitude", "longitude"],
            },
        },
    },
]


def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    raw_weather = None

    for _ in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS, temperature=0
        )
        message = response.choices[0].message

        thought = getattr(message, "reasoning", None)
        if thought:
            print(f"Thought: {thought.strip()[:300]}")

        # no more actions -> this is the final answer
        if not message.tool_calls:
            if raw_weather:
                print("\n--- Raw API response ---")
                print(json.dumps(raw_weather, indent=2))
            print("\n--- Explanation ---")
            print(message.content.strip())
            return

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.function.name, "arguments": call.function.arguments},
                }
                for call in message.tool_calls
            ],
        })

        for call in message.tool_calls:
            name = call.function.name
            print(f"Action: {name}({call.function.arguments})")
            try:
                args = json.loads(call.function.arguments)
                observation = FUNCTIONS[name](**args)
                if name == "get_weather":
                    raw_weather = observation
            except Exception as e:
                observation = {"error": str(e)}
            print(f"Observation: {json.dumps(observation)}\n")
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(observation),
            })

    print("Stopped: too many steps without a final answer.")


def main():
    print("Campus Weather Agent (type 'exit' to quit)\n")
    while True:
        question = input("Student: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        print()
        try:
            run_agent(question)
        except Exception as e:
            print(f"Something went wrong: {e}")
        print()


if __name__ == "__main__":
    main()
